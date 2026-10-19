"""Validacion cruzada temporal y test final congelado (Tarea 3).

Con series de tiempo no se usa k-fold aleatorio (mezclaria futuro y pasado): se usa CV de
origen rodante con ventana expansiva sobre los origenes de train. Flujo:

    study = run_cv_study(feat, {**ridge_specs(), **enet_specs(), **lgbm_specs()})
    study.table()           # WAPE de CV por configuracion
    study.block_table()     # dispersion en bloques de origenes
    study.decision()        # reglas de decision (solo con train)
    study.holdout()         # test final: modelos congelados, se mira una sola vez

Claves de `specs`: (familia, config), con familia en ridge | enet | lgbm.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from logforecast import features as F
from logforecast.evaluation import (
    ORIG_TEST,
    ORIG_TRAIN,
    SPEC_BASE,
    REF_NAME,
    SPEC_RATIO_CD,
    SPEC_REF,
    backtest,
    make_estimator,
    wape,
    wape_table,
)

FAMILIAS = ("ridge", "enet", "lgbm")
ALPHAS_RIDGE = (0.1, 1, 10, 100, 1000)
ALPHAS_ENET = (1e-4, 1e-3, 1e-2, 1e-1)  # el objetivo esta en escala log1p, por eso son chicos
L1_RATIOS = (0.1, 0.5, 0.9, 1.0)
LGBM_GRID = {"num_leaves": (4, 8), "n_estimators": (100, 300), "min_child_samples": (20, 50)}
LGBM_BASE = dict(learning_rate=0.05, reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
N_BLOQUES = 4
MIN_BLOQUES = 3  # el candidato debe ganar en al menos 3 de 4 bloques
# Modelo elegido en 04_validacion_cruzada_t3 (familia, config). El notebook 04 verifica que coincida con la CV.
ELEGIDO = ("enet", (0.1, 0.1))  # ElasticNet: l1_ratio=0.1, alpha=0.1


def default_cols() -> list[str]:
    return list(F.CORE + F.EXTRA)


# ----------------------------------------------------------------------------- grillas
def ridge_specs(cols=None, alphas=ALPHAS_RIDGE) -> dict:
    cols = list(cols or default_cols())
    return {("ridge", (a,)): dict(kind="ridge", cols=cols, alpha=a) for a in alphas}


def enet_specs(cols=None, alphas=ALPHAS_ENET, l1_ratios=L1_RATIOS) -> dict:
    cols = list(cols or default_cols())
    return {("enet", (l1, a)): dict(kind="enet", cols=cols, alpha=a, l1_ratio=l1)
            for l1, a in product(l1_ratios, alphas)}


def lgbm_specs(cols=None, grid=LGBM_GRID, base=LGBM_BASE) -> dict:
    cols = list(cols or default_cols())
    out = {}
    for vals in product(*grid.values()):
        params = dict(base, **dict(zip(grid, vals)))
        out[("lgbm", vals)] = dict(kind="lgbm", cols=cols, params=params)
    return out


def chosen_spec(cols=None) -> dict:
    """Spec del modelo elegido (ELEGIDO), con las features por defecto."""
    grillas = {"ridge": ridge_specs, "enet": enet_specs, "lgbm": lgbm_specs}
    return grillas[ELEGIDO[0]](cols)[ELEGIDO]


def label(familia: str, cfg: tuple) -> str:
    if familia == "ridge":
        return f"alpha={cfg[0]:g}"
    if familia == "enet":
        return f"l1_ratio={cfg[0]:g} alpha={cfg[1]:g}"
    if familia == "lgbm":
        return f"leaves={cfg[0]} n={cfg[1]} min_child={cfg[2]}"
    return str(cfg)


# ----------------------------------------------------------------------------- bloques
def make_blocks(origenes: pd.DatetimeIndex, n: int = N_BLOQUES) -> list[pd.DatetimeIndex]:
    """Divide los origenes en n bloques contiguos."""
    return [origenes[i] for i in np.array_split(np.arange(len(origenes)), n)]


def wape_by_block(d: pd.DataFrame, bloques: list) -> list[float]:
    return [wape(d[d["origen"].isin(b)]) for b in bloques]


# ----------------------------------------------------------------------------- estudio
@dataclass
class CVStudy:
    """Resultado de una CV: backtests por configuracion, referencias, mejores y bloques."""

    feat: pd.DataFrame
    avail_lag: int
    orig_train: pd.DatetimeIndex
    orig_test: pd.DatetimeIndex
    specs: dict
    res: dict
    refs: dict
    scores: dict
    best: dict
    blocks: list

    @property
    def lag(self) -> pd.Timedelta:
        return pd.Timedelta(days=self.avail_lag)

    @property
    def train_fijo(self) -> pd.Timestamp:
        """Corte de entrenamiento de los modelos finales: lo conocido al inicio de test."""
        return self.orig_test[0] - self.lag

    def info(self) -> str:
        n = lambda t: int((self.feat[F.DESP] <= t - self.lag).sum())
        return (
            f"CV (train): {len(self.orig_train)} origenes | entrena con {n(self.orig_train[0])} filas (primero) "
            f"a {n(self.orig_train[-1])} (ultimo) | filas de CV: {len(self.refs['base'])}\n"
            f"Test: {len(self.orig_test)} origenes | modelo unico entrenado con "
            f"{int((self.feat[F.DESP] <= self.train_fijo).sum())} filas\n"
            f"base: {wape(self.refs['base']) * 100:.2f}% | {REF_NAME}: {wape(self.refs['ref']) * 100:.2f}%"
        )

    # --- tablas de CV
    def table(self) -> pd.DataFrame:
        """WAPE de CV (%) de todas las configuraciones, de mejor a peor."""
        t = pd.DataFrame(
            [{"modelo": k[0], "config": label(*k), "WAPE CV (%)": round(v * 100, 2)} for k, v in self.scores.items()]
        )
        return t.sort_values("WAPE CV (%)").reset_index(drop=True)

    def best_table(self) -> pd.DataFrame:
        """Mejor configuracion de cada familia."""
        return pd.DataFrame(
            {f: {"mejor config": label(*k), "WAPE CV (%)": round(self.scores[k] * 100, 2)} for f, k in self.best.items()}
        ).T

    def block_table(self) -> pd.DataFrame:
        """WAPE (%) por bloque de origenes: referencia (REF_NAME), base y la mejor config de cada familia."""
        idx = [f"{b[0]:%d-%b}..{b[-1]:%d-%b}" for b in self.blocks]
        filas = {REF_NAME: wape_by_block(self.refs["ref"], self.blocks),
                 "base": wape_by_block(self.refs["base"], self.blocks)}
        for f, k in self.best.items():
            filas[f"{f} ({label(*k)})"] = wape_by_block(self.res[k], self.blocks)
        t = pd.DataFrame(filas, index=idx).T
        t["media"] = t.mean(axis=1)
        t["desv"] = t.drop(columns="media").std(axis=1)
        return t * 100

    # --- decision (solo con train)
    def decision(self, ref: str = "ridge", min_bloques: int = MIN_BLOQUES) -> pd.DataFrame:
        """Cada familia reemplaza a la referencia (REF_NAME) si mejora su WAPE de CV y gana en >= min_bloques bloques.

        Si `ref` (ridge) esta entre las familias, las demas se miden tambien contra su mejor config.
        """
        w_ratio = wape(self.refs["ref"])
        b_ratio = np.array(wape_by_block(self.refs["ref"], self.blocks))
        k_ref = self.best.get(ref)
        b_ref = np.array(wape_by_block(self.res[k_ref], self.blocks)) if k_ref else None
        filas = {}
        for f, k in self.best.items():
            b = np.array(wape_by_block(self.res[k], self.blocks))
            gana = int((b < b_ratio).sum())
            fila = {
                "config": label(*k),
                "WAPE CV (%)": round(self.scores[k] * 100, 2),
                f"{REF_NAME} (%)": round(w_ratio * 100, 2),
                f"bloques vs {REF_NAME}": f"{gana}/{len(self.blocks)}",
                f"reemplaza a {REF_NAME}": bool(self.scores[k] < w_ratio and gana >= min_bloques),
            }
            if k_ref and f != ref:
                gana_r = int((b < b_ref).sum())
                fila["bloques vs ridge"] = f"{gana_r}/{len(self.blocks)}"
                fila["mejor que ridge"] = bool(self.scores[k] < self.scores[k_ref] and gana_r >= min_bloques)
            filas[f] = fila
        return pd.DataFrame(filas).T

    @property
    def ref_name(self) -> str:
        return REF_NAME

    def passing(self, min_bloques: int = MIN_BLOQUES) -> list[str]:
        """Familias que reemplazan a la referencia segun la regla de decision."""
        d = self.decision(min_bloques=min_bloques)
        return [f for f in d.index if d.loc[f, f"reemplaza a {REF_NAME}"]]

    def winner(self, min_bloques: int = MIN_BLOQUES) -> str | None:
        """Familia ganadora de la regla (menor WAPE de CV entre las que reemplazan a la referencia), o None."""
        pasan = self.passing(min_bloques)
        return min(pasan, key=lambda f: self.scores[self.best[f]]) if pasan else None

    # --- test final congelado
    def holdout(self):
        """Test final: cada modelo se entrena UNA vez con lo conocido al inicio de test.

        Devuelve (HO, tabla, dif): backtests, WAPE global y por CD (%), y diferencia en puntos
        de WAPE vs la referencia REF_NAME (y vs ridge si esta) de cada modelo `*_cv`. Incluye ratio_cd
        como baseline adicional.
        """
        specs = {"base": SPEC_BASE, "ratio_cd": SPEC_RATIO_CD, REF_NAME: SPEC_REF,
                 **{f"{f}_cv": self.specs[k] for f, k in self.best.items()}}
        ho = {n: backtest(self.feat, s, self.avail_lag, self.orig_test, "test", train_hasta=self.train_fijo)
              for n, s in specs.items()}
        tab = wape_table(ho)
        cv_cols = [c for c in tab.columns if c.endswith("_cv")]
        dif = {f"{c} - {REF_NAME}": tab[c] - tab[REF_NAME] for c in cv_cols}
        if "ridge_cv" in tab:
            dif.update({f"{c} - ridge_cv": tab[c] - tab["ridge_cv"] for c in cv_cols if c != "ridge_cv"})
        return ho, tab, pd.DataFrame(dif) * 100

    # --- que aprendio cada modelo (entrenado al inicio de test)
    def _train_set(self):
        tr = self.feat[self.feat[F.DESP] <= self.train_fijo]
        return tr

    def coefficients(self, familias=("ridge", "enet")) -> pd.DataFrame:
        """Coeficientes estandarizados de los modelos lineales elegidos (una columna por familia)."""
        tr = self._train_set()
        out = {}
        for f in familias:
            if f not in self.best:
                continue
            spec = self.specs[self.best[f]]
            X = F.design_matrix(tr, spec["cols"])
            m = make_estimator(spec).fit(X, tr["y"])
            out[f] = pd.Series(m[-1].coef_, index=X.columns)
        return pd.DataFrame(out)

    def importance(self, familia: str = "lgbm") -> pd.Series:
        """Importancia por ganancia (%) del LightGBM elegido, de menor a mayor."""
        tr = self._train_set()
        spec = self.specs[self.best[familia]]
        X = F.design_matrix(tr, spec["cols"])
        spec = dict(spec, params=dict(spec["params"], importance_type="gain"))
        m = make_estimator(spec).fit(X, tr["y"])
        imp = pd.Series(m.feature_importances_, index=X.columns)
        return (imp / imp.sum() * 100).sort_values()


def run_cv_study(
    feat: pd.DataFrame,
    specs: dict,
    avail_lag: int = 1,
    orig_train: pd.DatetimeIndex = ORIG_TRAIN,
    orig_test: pd.DatetimeIndex = ORIG_TEST,
    n_bloques: int = N_BLOQUES,
) -> CVStudy:
    """Corre la CV de todas las configuraciones de `specs` sobre los origenes de train.

    Los objetivos evaluados llegan hasta orig_test[0]; ninguno cae despues, asi que train no ve test.
    """
    desp_max = orig_test[0]
    res = {k: backtest(feat, s, avail_lag, orig_train, "cv", desp_max=desp_max) for k, s in specs.items()}
    refs = {"base": backtest(feat, SPEC_BASE, avail_lag, orig_train, "cv", desp_max=desp_max),
            "ratio": backtest(feat, SPEC_RATIO_CD, avail_lag, orig_train, "cv", desp_max=desp_max),
            "ref": backtest(feat, SPEC_REF, avail_lag, orig_train, "cv", desp_max=desp_max)}
    scores = {k: wape(d) for k, d in res.items()}
    best = {}
    for f in dict.fromkeys(k[0] for k in specs):  # orden de aparicion
        best[f] = min((k for k in scores if k[0] == f), key=scores.get)
    return CVStudy(feat, avail_lag, orig_train, orig_test, specs, res, refs, scores, best,
                   make_blocks(orig_train, n_bloques))
