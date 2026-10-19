"""Tarea 4: estudio del gate de historia minima (demostracion y justificacion del umbral).

El gate en si vive en `logforecast.gate`; aqui estan las funciones que lo demuestran y que
miden, con numeros, que pasa al mover el umbral:
  - estado del gate en el tiempo y casos borde,
  - cold start simulado con `GatedCorrector`,
  - curva de aprendizaje (WAPE segun dias de historia de entrenamiento),
  - error segun dias de historia disponibles y simulacion del sistema con gate por umbral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from logforecast import features as F
from logforecast.cv import ELEGIDO, chosen_spec
from logforecast.evaluation import REF_NAME, SPEC_BASE, SPEC_REF, backtest, wape
from logforecast.gate import GateStatus, evaluate_gate
from logforecast.model import ElasticNetCorrector, GatedCorrector

TRAMOS = [0, 20, 27, 34, 44, 999]
ETIQUETAS = ["14-20", "21-27", "28-34", "35-44", "45+"]


# ----------------------------------------------------------------------------- historia y gate
def history_frame(orders: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (CD, dia con datos): es lo que mira el gate."""
    hist = orders[[F.CD, F.DESP]].copy()
    hist[F.DESP] = pd.to_datetime(hist[F.DESP])
    return hist.drop_duplicates().reset_index(drop=True)


def history_summary(hist: pd.DataFrame) -> pd.DataFrame:
    r = hist.groupby(F.CD)[F.DESP].agg(desde="min", hasta="max", dias_con_datos="nunique")
    r["dias_calendario"] = (r["hasta"] - r["desde"]).dt.days + 1
    r["huecos"] = r["dias_calendario"] - r["dias_con_datos"]
    return r


def gate_state(hist: pd.DataFrame, as_of, centers, min_days: int = 45) -> pd.DataFrame:
    """Dias de historia continua y estado del gate por CD a la fecha `as_of`."""
    d = evaluate_gate(hist, min_days, centers=list(centers), as_of=pd.Timestamp(as_of))
    return pd.DataFrame({cd: {"dias": x.continuous_days, "estado": x.status.value} for cd, x in d.items()}).T


def first_open_date(hist, centers, min_days, lag_days, start, end):
    """Primer dia de emision con el gate abierto en todos los CD (None si no se abre)."""
    lag = pd.Timedelta(days=lag_days)
    for t in pd.date_range(start, end):
        d = evaluate_gate(hist, min_days, centers=list(centers), as_of=t - lag)
        if d and all(x.status == GateStatus.TRAIN for x in d.values()):
            return t
    return None


def edge_cases(hist: pd.DataFrame, centers, as_of, min_days: int = 45) -> pd.DataFrame:
    """Casos borde: el gate no debe lanzar excepciones y debe caer al fallback."""
    centers = list(centers)
    hueco = hist[(hist[F.CD] == centers[1]) & (hist[F.DESP] == pd.Timestamp("2026-02-10"))].index
    casos = {
        "historia vacía": pd.DataFrame({F.CD: pd.Series(dtype=str), F.DESP: pd.to_datetime([])}),
        f"CD sin datos ({centers[0]})": hist[hist[F.CD] != centers[0]],
        f"hueco el 10-feb en {centers[1]}": hist.drop(hueco),
        "filas duplicadas": pd.concat([hist, hist], ignore_index=True),
        "columna de fecha faltante": hist.rename(columns={F.DESP: "otra_columna"}),
    }
    filas = {}
    for nombre, h in casos.items():
        try:
            d = evaluate_gate(h, min_days, centers=centers, as_of=pd.Timestamp(as_of))
            if not d:
                filas[nombre] = {"excepción": "no", "decisiones": 0, "en TRAIN": 0,
                                 "en FALLBACK": "todos (sin decisión)"}
            else:
                fb = ", ".join(f"{cd} ({x.continuous_days}d)" for cd, x in d.items()
                               if x.status == GateStatus.FALLBACK) or "-"
                filas[nombre] = {"excepción": "no", "decisiones": len(d),
                                 "en TRAIN": sum(x.status == GateStatus.TRAIN for x in d.values()),
                                 "en FALLBACK": fb}
        except Exception as e:  # si esto aparece, el gate no cumple el requisito
            filas[nombre] = {"excepción": f"SÍ: {e}", "decisiones": None, "en TRAIN": None, "en FALLBACK": None}
    return pd.DataFrame(filas).T


def cold_start_demo(frame, centers, cd_nuevo, inicio_nuevo, t_demo, min_days: int = 45, lag_days: int = 1,
                    model_factory=ElasticNetCorrector):
    """Simula que `cd_nuevo` empezo a operar en `inicio_nuevo` y predice lo emitido en `t_demo`.

    Devuelve (decisiones, chequeo): decisiones del gate por CD y, por CD, si la prediccion
    es igual al base (fallback) o fue corregida.
    """
    t_demo, inicio_nuevo = pd.Timestamp(t_demo), pd.Timestamp(inicio_nuevo)
    train = frame[frame[F.DESP] <= t_demo - pd.Timedelta(days=lag_days)]
    train = train[~((train[F.CD] == cd_nuevo) & (train[F.DESP] < inicio_nuevo))]
    test = frame[frame[F.EMI] == t_demo].copy()

    gm = GatedCorrector(model_factory, min_days=min_days).fit(train, centers=list(centers))
    test["pred"] = gm.predict(test)

    dec = pd.DataFrame({cd: d.to_dict() for cd, d in gm.decisions_.items()}).T
    filas = {}
    for cd, g in test.groupby(F.CD):
        filas[cd] = {
            "filas": len(g),
            "pred == base": bool(np.allclose(g["pred"], g[F.PRON])),
            "factor medio": float((g["pred"] / g[F.PRON].replace(0, np.nan)).mean()),
        }
    return dec[["status", "continuous_days", "min_days", "reason"]], pd.DataFrame(filas).T


# ----------------------------------------------------------------------------- curva de aprendizaje
def default_models(min_rows: int = 30) -> dict:
    """Modelos del estudio: la referencia (REF_NAME) y el modelo elegido en la CV (ELEGIDO), con
    configuracion fija. Con menos de `min_rows` filas de entrenamiento devuelven el base."""
    return {REF_NAME: dict(SPEC_REF, min_rows=min_rows), ELEGIDO[0]: dict(chosen_spec(), min_rows=min_rows)}


def learning_curve(feat, origenes, ventanas, avail_lag: int = 1, modelos: dict | None = None):
    """WAPE segun los ultimos N dias de reales usados para entrenar (None = toda la historia).

    Todas las ventanas se evaluan sobre las mismas filas. Devuelve (curva, n_filas).
    Nota: ratio_mov14 usa por construccion solo los ultimos 14 dias, asi que con N >= 14 no cambia.
    """
    specs = modelos or default_models()
    filas = {}
    for v in ventanas:
        filas["todo" if v is None else v] = {
            n: wape(backtest(feat, s, avail_lag, origenes, "cv", ventana=v)) for n, s in specs.items()
        }
    base = backtest(feat, SPEC_BASE, avail_lag, origenes, "cv")
    curva = pd.DataFrame(filas).T.rename_axis("días de historia (ventana)")
    curva.insert(0, "base", wape(base))
    return curva, len(base)


# ----------------------------------------------------------------------------- por dias de historia
def history_days(hist, centers, origenes, lag_days: int = 1, min_days: int = 45) -> pd.DataFrame:
    """Dias de historia continua por origen y CD, con el gate real."""
    lag = pd.Timedelta(days=lag_days)
    return pd.DataFrame([
        {"origen": t, F.CD: cd, "dias": x.continuous_days}
        for t in origenes
        for cd, x in evaluate_gate(hist, min_days, centers=list(centers), as_of=t - lag).items()
    ])


def backtest_by_history(feat, hist, centers, origenes, avail_lag: int = 1, modelos: dict | None = None):
    """Backtest con ventana expansiva y error agrupado por tramo de dias de historia.

    Devuelve (D, tabla): D tiene una fila por pronostico con pred_base, pred_<modelo>, `dias` y
    `tramo`; tabla resume el WAPE (%) por tramo.
    """
    specs = modelos or default_models()
    res = {k: backtest(feat, s, avail_lag, origenes, "cv") for k, s in {"base": SPEC_BASE, **specs}.items()}
    assert len({len(d) for d in res.values()}) == 1  # mismo orden de filas

    D = res["base"].rename(columns={"pred": "pred_base"})
    for n in specs:
        D[f"pred_{n}"] = res[n]["pred"].to_numpy()
    D = D.merge(history_days(hist, centers, origenes, avail_lag), on=["origen", F.CD], how="left")
    D["tramo"] = pd.cut(D["dias"], bins=TRAMOS, labels=ETIQUETAS)

    filas = {}
    for tramo, g in D.groupby("tramo", observed=True):
        filas[tramo] = {"orígenes": g["origen"].nunique(), "filas": len(g), "base": wape(g, "pred_base") * 100,
                        **{n: wape(g, f"pred_{n}") * 100 for n in specs}}
    tabla = pd.DataFrame(filas).T
    for n in specs:
        tabla[f"{n} - base (pp)"] = tabla[n] - tabla["base"]
    return D, tabla.rename_axis("días de historia")


# ----------------------------------------------------------------------------- simulacion por umbral
def open_status(hist, centers, origenes, umbrales, lag_days: int = 1) -> dict:
    """Estado del gate real por umbral, origen y CD: {T: DataFrame(origen, CD, abierto)}."""
    lag = pd.Timedelta(days=lag_days)
    return {
        T: pd.DataFrame([
            {"origen": t, F.CD: cd, "abierto": x.status == GateStatus.TRAIN}
            for t in origenes
            for cd, x in evaluate_gate(hist, T, centers=list(centers), as_of=t - lag).items()
        ])
        for T in umbrales
    }


def simulate_thresholds(D: pd.DataFrame, abierto: dict, modelo: str, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Sistema con gate por umbral: CD abiertos usan el modelo, el resto el base.

    modelo: nombre de un modelo de D (columna pred_<modelo>). Devuelve por umbral la cobertura, el WAPE con gate, su diferencia
    contra el base y contra el modelo sin gate, y un IC 90 % de la diferencia con el base
    (remuestreo de origenes con reemplazo; los origenes consecutivos son dependientes, es una guia).
    """
    rng = np.random.default_rng(seed)
    origs = np.array(sorted(D["origen"].unique()))
    idx_boot = [rng.integers(0, len(origs), len(origs)) for _ in range(n_boot)]
    out = {}
    for T, ab in abierto.items():
        g = D.merge(ab, on=["origen", F.CD], how="left")
        pred = np.where(g["abierto"], g[f"pred_{modelo}"], g["pred_base"])
        g = g.assign(err_g=(g[F.REAL] - pred).abs(), err_b=(g[F.REAL] - g["pred_base"]).abs())
        w_g = g["err_g"].sum() / g[F.REAL].sum()
        w_b = g["err_b"].sum() / g[F.REAL].sum()
        w_m = wape(g, f"pred_{modelo}")
        po = g.groupby("origen").agg(eg=("err_g", "sum"), eb=("err_b", "sum"), r=(F.REAL, "sum")).reindex(origs)
        eg, eb, r = po["eg"].to_numpy(), po["eb"].to_numpy(), po["r"].to_numpy()
        diffs = [(eg[i].sum() - eb[i].sum()) / r[i].sum() for i in idx_boot]
        lo, hi = np.percentile(diffs, [5, 95]) * 100
        out[T] = {"cobertura (%)": g["abierto"].mean() * 100, "WAPE gate (%)": w_g * 100,
                  "vs base (pp)": (w_g - w_b) * 100, "vs modelo sin gate (pp)": (w_g - w_m) * 100,
                  "IC90 vs base (pp)": f"[{lo:.2f}, {hi:.2f}]"}
    return pd.DataFrame(out).T.rename_axis("umbral T (días)")
