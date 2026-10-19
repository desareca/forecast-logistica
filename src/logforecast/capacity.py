"""Diagnostico de capacidad logistica: que tan ejecutable es el pronostico y cuanto afecta el tope.

La capacidad es el maximo de unidades que un CD puede despachar un dia. Aqui no se entrena nada:
se compara pronostico y real con ese tope y se mide el efecto de recortar (`min(pred, capacidad)`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from logforecast import features as F
from logforecast.metrics import wape

CAP = "capacidad"


def attach_capacity(frame: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    """Agrega la capacidad del dia de despacho y los indicadores de uso (pronostico/cap y real/cap)."""
    c = cap.rename(columns={"fecha": F.DESP, "capacidad_maxima_unidades": CAP})[[F.CD, F.DESP, CAP]]
    d = frame.merge(c, on=[F.CD, F.DESP], how="left")
    if d[CAP].isna().any():
        raise ValueError(f"{int(d[CAP].isna().sum())} filas sin capacidad para su CD y fecha de despacho")
    d["uso_pron"] = d[F.PRON] / d[CAP]
    d["uso_real"] = d[F.REAL] / d[CAP]
    return d


def capped(d: pd.DataFrame, col: str = F.PRON) -> pd.Series:
    """Prediccion recortada a la capacidad del dia."""
    return np.minimum(d[col], d[CAP])


def usage_by_cd(d: pd.DataFrame) -> pd.DataFrame:
    """Uso de la capacidad por CD: real (nivel y saturacion) y pronostico (cuanto y cuanto se pasa)."""
    def f(g):
        ex = g[g[F.PRON] > g[CAP]]
        return pd.Series({
            "cap. normal": g[CAP].min(), "cap. alta": g[CAP].max(),
            "uso real medio (%)": g["uso_real"].mean() * 100,
            "dias real >= 90% cap": int(g.drop_duplicates(F.DESP).eval("uso_real >= 0.9").sum()),
            "dias real > cap": int(g.drop_duplicates(F.DESP).eval("uso_real > 1").sum()),
            "pron > cap (% filas)": (g[F.PRON] > g[CAP]).mean() * 100,
            "pron/cap cuando excede": ex["uso_pron"].mean() if len(ex) else np.nan,
        })
    return d.groupby(F.CD).apply(f, include_groups=False).round(2)


def exceed_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """Cruce entre 'el pronostico supera la capacidad' y 'el real supera la capacidad' (filas)."""
    t = pd.crosstab(
        np.where(d[F.PRON] > d[CAP], "pron > cap", "pron <= cap"),
        np.where(d[F.REAL] > d[CAP], "real > cap", "real <= cap"),
    )
    t["total"] = t.sum(axis=1)
    return t


def excess_rows_profile(d: pd.DataFrame) -> pd.Series:
    """Que pasa en las filas donde el pronostico excede la capacidad."""
    x = d[d[F.PRON] > d[CAP]]
    return pd.Series({
        "filas": len(x), "% de las filas": len(x) / len(d) * 100,
        "pron/cap medio": x["uso_pron"].mean(), "real/cap medio": x["uso_real"].mean(),
        "% donde el real tambien excede": (x[F.REAL] > x[CAP]).mean() * 100,
        "sesgo (real - pron)/real (%)": (x[F.REAL].sum() - x[F.PRON].sum()) / x[F.REAL].sum() * 100,
        "unidades pronosticadas sobre el tope": (x[F.PRON] - x[CAP]).sum(),
    })


def cap_effect(d: pd.DataFrame, by: str | None = F.CD, col: str = F.PRON) -> pd.DataFrame:
    """WAPE (%) de `col` sin y con recorte a la capacidad, y filas afectadas."""
    d = d.assign(_cap=capped(d, col))

    def f(g):
        w0 = wape(g[F.REAL], g[col]) * 100
        w1 = wape(g[F.REAL], g["_cap"]) * 100
        return pd.Series({"WAPE sin cap (%)": w0, "WAPE con cap (%)": w1, "dif (pp)": w1 - w0,
                          "filas recortadas (%)": (g[col] > g[CAP]).mean() * 100})
    r = d.groupby(by).apply(f, include_groups=False) if by else f(d).to_frame("GLOBAL").T
    if by:
        r.loc["GLOBAL"] = f(d)
    return r.round(2)


def weekday_exceed(d: pd.DataFrame) -> pd.DataFrame:
    """% de filas que excede la capacidad por dia de la semana del despacho, y capacidad media (martes es el dia alto)."""
    dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
    g = d.assign(dow=d[F.DESP].dt.dayofweek).groupby("dow")
    r = pd.DataFrame({"pron > cap (%)": g.apply(lambda x: (x[F.PRON] > x[CAP]).mean() * 100, include_groups=False),
                      "uso real medio (%)": g["uso_real"].mean() * 100, "filas": g.size()})
    r.index = [dias[i] for i in r.index]
    return r.round(1)


def models_effect(res: dict[str, pd.DataFrame], cap: pd.DataFrame, zona: str | None = None) -> pd.DataFrame:
    """Efecto del recorte sobre las predicciones de cada modelo (res: nombre -> backtest con columna `pred`).

    Devuelve, por modelo, el WAPE global (%) sin y con cap, la diferencia y el % de filas recortadas.
    """
    c = cap.rename(columns={"fecha": F.DESP, "capacidad_maxima_unidades": CAP})[[F.CD, F.DESP, CAP]]
    filas = {}
    for nombre, r in res.items():
        x = r if zona is None else r[r["zona"] == zona]
        x = x.merge(c, on=[F.CD, F.DESP], how="left")
        w0 = wape(x[F.REAL], x["pred"]) * 100
        w1 = wape(x[F.REAL], np.minimum(x["pred"], x[CAP])) * 100
        filas[nombre] = {"WAPE sin cap (%)": w0, "WAPE con cap (%)": w1, "dif (pp)": w1 - w0,
                         "filas recortadas (%)": (x["pred"] > x[CAP]).mean() * 100}
    return pd.DataFrame(filas).T.round(2)


# ----------------------------------------------------------------------------- alerta para el endpoint
ALERTA_OK, ALERTA_CERCA, ALERTA_EXCEDE, ALERTA_SIN_CAP = "OK", "CERCA", "EXCEDE", "SIN_CAPACIDAD"


def capacity_alert(
    records: pd.DataFrame,
    cap: pd.DataFrame | None,
    pred_col: str = "unidades_corregidas",
    warn_ratio: float = 0.9,
) -> pd.DataFrame:
    """Agrega capacidad maxima, uso de la capacidad y alerta a las predicciones. Nunca lanza excepciones.

    Requiere `centro_distribucion` y `fecha_despacho` en `records` para ubicar la capacidad del dia.
    Alertas: OK (uso < warn_ratio), CERCA (warn_ratio <= uso <= 1), EXCEDE (uso > 1) y
    SIN_CAPACIDAD (no hay dato de capacidad para ese CD y fecha, o falta la fecha).
    """
    out = records.copy()
    out["capacidad_maxima"] = np.nan
    if cap is not None and not cap.empty and F.DESP in out.columns and F.CD in out.columns:
        c = cap.rename(columns={"fecha": F.DESP, "capacidad_maxima_unidades": CAP})
        c = c.assign(**{F.DESP: pd.to_datetime(c[F.DESP], errors="coerce")}).drop_duplicates([F.CD, F.DESP], keep="last")
        key = pd.MultiIndex.from_arrays([out[F.CD], pd.to_datetime(out[F.DESP], errors="coerce")])
        out["capacidad_maxima"] = pd.Series(c.set_index([F.CD, F.DESP])[CAP]).reindex(key).to_numpy()
    uso = out[pred_col].astype(float) / out["capacidad_maxima"]
    out["uso_capacidad"] = uso.round(3)
    out["alerta_capacidad"] = np.select(
        [uso.isna(), uso > 1.0, uso >= warn_ratio],
        [ALERTA_SIN_CAP, ALERTA_EXCEDE, ALERTA_CERCA],
        default=ALERTA_OK,
    )
    return out
