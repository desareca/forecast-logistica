"""Motor de backtest de origen rodante (Tarea 3).

Separacion train / test (sin mezclar):
  - train: origenes ORIG_TRAIN. Todos los ajustes (modelos, hiperparametros, reglas) se hacen aqui.
    Los objetivos evaluados llegan hasta ORIG_TEST[0] (el 14-feb), nunca mas alla.
  - test: origenes ORIG_TEST. El modelo se entrena UNA vez con lo conocido al inicio de test
    (`train_hasta`) y no se reajusta.

Regla de disponibilidad: al emitir en t, el real conocido llega hasta t - avail_lag, asi que
el entrenamiento de ese origen usa filas con fecha_despacho <= t - avail_lag.

Un `spec` es un dict con la clave `kind`:
    base | snaive | ratio (by=[...]) | ratio_mov (window=N) | ridge | enet | lgbm
y opcionalmente `min_rows` (con menos filas de entrenamiento se devuelve el pronostico base).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from logforecast import features as F
from logforecast import metrics

ORIG_TRAIN = pd.date_range("2026-01-22", "2026-02-13")
ORIG_TEST = pd.date_range("2026-02-14", "2026-02-28")
ZONAS = ("train", "test")

SPEC_BASE = dict(kind="base")
SPEC_RATIO_CD = dict(kind="ratio", by=[F.CD])
SPEC_RATIO_MOV14 = dict(kind="ratio_mov", window=14)

# Baseline de referencia que deben superar los modelos (mejor baseline en train con rezago 1, notebook 03)
REF_NAME = "ratio_mov14"
SPEC_REF = SPEC_RATIO_MOV14


def make_estimator(spec: dict):
    """Estimador sklearn/lightgbm para los kinds ridge, enet y lgbm."""
    kind = spec["kind"]
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=spec.get("alpha", 10.0)))
    if kind == "enet":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=spec["alpha"], l1_ratio=spec["l1_ratio"], max_iter=20000),
        )
    if kind == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:  # pragma: no cover
            raise ImportError("Falta lightgbm: esta en requirements.txt (docker compose build)") from e
        return LGBMRegressor(**spec["params"], random_state=0, n_jobs=1, verbose=-1)
    raise ValueError(f"kind sin estimador: {kind}")


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, spec: dict) -> np.ndarray:
    """Entrena con `tr` (si el modelo lo requiere) y devuelve el pronostico corregido para `te`."""
    kind = spec["kind"]
    base = te[F.PRON].to_numpy(float)
    if kind == "base":
        return base
    if kind == "snaive":  # real del mismo dia de la semana anterior; si no esta disponible, el base
        return te["real_lag7"].fillna(te[F.PRON]).to_numpy(float)
    if len(tr) < spec.get("min_rows", 1):  # muy pocas filas: fallback al base
        return base
    if kind == "ratio":  # factor multiplicativo = sum(real) / sum(pron) por grupo
        by = list(spec.get("by", [F.CD]))
        s = tr.groupby(by)[[F.REAL, F.PRON]].sum()
        fac = (s[F.REAL] / s[F.PRON]).rename("fac")
        f = te[by].merge(fac, left_on=by, right_index=True, how="left")["fac"].fillna(1.0)
        return base * f.to_numpy()
    if kind == "ratio_mov":  # igual que ratio por CD, pero solo con los ultimos `window` dias
        trw = tr[tr[F.DESP] > tr[F.DESP].max() - pd.Timedelta(days=spec.get("window", 14))]
        s = trw.groupby(F.CD)[[F.REAL, F.PRON]].sum()
        f = te[F.CD].map(s[F.REAL] / s[F.PRON]).fillna(1.0)
        return base * f.to_numpy()
    if kind in ("ridge", "enet", "lgbm"):
        m = make_estimator(spec)
        m.fit(F.design_matrix(tr, spec["cols"]), tr["y"])
        return F.apply_delta(te[F.PRON], m.predict(F.design_matrix(te, spec["cols"])))
    raise ValueError(kind)


def backtest(
    feat: pd.DataFrame,
    spec: dict,
    avail_lag: int,
    origenes: Iterable,
    zona: str = "cv",
    desp_max: pd.Timestamp | None = None,
    train_hasta: pd.Timestamp | None = None,
    ventana: int | None = None,
) -> pd.DataFrame:
    """Backtest de origen rodante. Devuelve las filas evaluadas con `pred`, `origen` y `zona`.

    desp_max: ignora objetivos con despacho posterior (evita usar reales de test en train).
    train_hasta: si se da, el entrenamiento es fijo (despacho <= train_hasta) en todos los origenes.
    ventana: si se da, entrena solo con los ultimos N dias de reales conocidos (ventana movil).
    """
    lag = pd.Timedelta(days=avail_lag)
    out = []
    for t in origenes:
        corte = train_hasta if train_hasta is not None else t - lag
        tr = feat[feat[F.DESP] <= corte]
        if ventana:
            tr = tr[tr[F.DESP] > corte - pd.Timedelta(days=ventana)]
        te = feat[feat[F.EMI] == t]
        if desp_max is not None:
            te = te[te[F.DESP] <= desp_max]
        if te.empty:
            continue
        out.append(te.assign(pred=fit_predict(tr, te, spec), origen=t, zona=zona))
    if not out:
        raise ValueError("ningun origen tiene filas para evaluar")
    return pd.concat(out, ignore_index=True)


def wape(d: pd.DataFrame, col: str = "pred") -> float:
    """WAPE de un frame de backtest (columna de prediccion configurable)."""
    return metrics.wape(d[F.REAL], d[col])


def wape_table(res: dict[str, pd.DataFrame], zona: str | None = None) -> pd.DataFrame:
    """WAPE global y por CD, una columna por modelo. Si `zona`, filtra la columna `zona`."""
    out = {}
    for n, d in res.items():
        dz = d if zona is None else d[d["zona"] == zona]
        out[n] = {"GLOBAL": wape(dz), **{cd: wape(g) for cd, g in dz.groupby(F.CD)}}
    return pd.DataFrame(out)


def run_models(
    frame: pd.DataFrame,
    modelos: dict[str, dict],
    avail_lag: int = 1,
    orig_train: pd.DatetimeIndex = ORIG_TRAIN,
    orig_test: pd.DatetimeIndex = ORIG_TEST,
):
    """Features + train (rodante) + test (congelado) de todos los modelos.

    Devuelve (feat, res, {zona: tabla}); `res[modelo]` junta las filas de train y de test.
    """
    feat = F.build_features(frame, avail_lag=avail_lag)
    lag = pd.Timedelta(days=avail_lag)
    res = {}
    for n, s in modelos.items():
        d_tr = backtest(feat, s, avail_lag, orig_train, "train", desp_max=orig_test[0])
        d_te = backtest(feat, s, avail_lag, orig_test, "test", train_hasta=orig_test[0] - lag)
        res[n] = pd.concat([d_tr, d_te], ignore_index=True)
    return feat, res, {z: wape_table(res, z) for z in ZONAS}


def run_lags(frame: pd.DataFrame, modelos: dict[str, dict], lags: Iterable[int] = (1, 2, 3, 4), **kw) -> dict:
    """`run_models` para varios rezagos del real. Devuelve {lag: (feat, res, tablas)}."""
    return {L: run_models(frame, modelos, L, **kw) for L in lags}


def describe_runs(corridas: dict, orig_test: pd.DatetimeIndex = ORIG_TEST) -> pd.DataFrame:
    """Filas de train, de test y con que se entrena el modelo de test, por rezago."""
    rows = {}
    for L, (feat, res, _) in corridas.items():
        b = next(iter(res.values()))
        rows[L] = {
            "filas train": int((b["zona"] == "train").sum()),
            "filas test": int((b["zona"] == "test").sum()),
            "filas con que se entrena el modelo de test": int(
                (feat[F.DESP] <= orig_test[0] - pd.Timedelta(days=L)).sum()
            ),
        }
    return pd.DataFrame(rows).T.rename_axis("AVAIL_LAG")


def summarize_runs(corridas: dict, zonas: Iterable[str] = ZONAS, ref: str = REF_NAME):
    """Tablas resumen de `run_lags`.

    Devuelve (glob, por_cd, dif):
      glob   WAPE global por zona y rezago, una columna por modelo.
      por_cd WAPE por zona, rezago y CD (incluye la fila GLOBAL).
      dif    diferencia vs `ref` (por defecto REF_NAME) en puntos de WAPE (negativo = mejora).
    """
    zonas = list(zonas)
    glob = pd.concat(
        {z: pd.DataFrame({L: c[2][z].loc["GLOBAL"] for L, c in corridas.items()}).T.rename_axis("AVAIL_LAG")
         for z in zonas},
        names=["zona", "AVAIL_LAG"],
    )
    por_cd = pd.concat(
        {z: pd.concat({L: c[2][z] for L, c in corridas.items()}, names=["AVAIL_LAG", "CD"]) for z in zonas},
        names=["zona", "AVAIL_LAG", "CD"],
    )
    dif = por_cd.sub(por_cd[ref], axis=0) * 100
    return glob, por_cd, dif


def unit_gaps(res: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Brechas en unidades por modelo: faltante (real > pred), sobrante (pred > real) y error absoluto.

    Faltante = demanda que el pronostico no cubrio (riesgo de flete de emergencia); sobrante = capacidad
    reservada de mas. Son proxies en unidades: no incluyen costos.
    """
    filas = {}
    for nombre, d in res.items():
        e = d[F.REAL] - d["pred"]
        filas[nombre] = {"real total": d[F.REAL].sum(), "faltante": e.clip(lower=0).sum(),
                         "sobrante": (-e).clip(lower=0).sum(), "error absoluto": e.abs().sum()}
    t = pd.DataFrame(filas).T
    t["faltante (% del real)"] = t["faltante"] / t["real total"] * 100
    t["sobrante (% del real)"] = t["sobrante"] / t["real total"] * 100
    return t.round(1)
