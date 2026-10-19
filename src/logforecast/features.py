"""Features causales para corregir el pronostico base (Tarea 3).

Regla de disponibilidad (parametro `avail_lag`, por defecto 1):
    al emitir en t, el real conocido llega hasta t - avail_lag.
    Cada feature de una fila usa solo datos con fecha_despacho <= t - avail_lag.
    El backtest debe entrenar con filas cuyo fecha_despacho <= t - avail_lag.

Objetivo: y = log1p(real) - log1p(pronostico). Correccion: expm1(log1p(pron) + delta).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Nombres de columnas del frame (diagnostics.build_eval_frame) ---
CD = "centro_distribucion"
EMI = "fecha_emision"
DESP = "fecha_despacho"
HOR = "horizonte"  # texto "D+1".."D+4"
PRON = "unidades_pronosticadas"
REAL = "real"

# --- Valores por defecto (se pueden sobreescribir en build_features) ---
AVAIL_LAG = 1       # dias de rezago con que se conoce el real
PEAK_Q = 0.80       # cuantil (solo pasado) para el indicador de pico
MIN_HIST_PEAK = 14  # dias minimos de historia para calcular el cuantil

CORE = ["err_7", "err_14", "h_dias", "dia_semana", "finde", "pron_vs_rec7"]  # + dummies cd_*
EXTRA = ["lag7_rel", "pico", "pico_margen"]


def _panel_diario(frame: pd.DataFrame, avail_lag: int, peak_q: float) -> pd.DataFrame:
    """Estadisticas causales por (CD, fecha de EMISION t); ventanas terminan en t-avail_lag."""
    f = frame[[CD, DESP, PRON, REAL]].copy()
    f["_e"] = np.log1p(f[REAL]) - np.log1p(f[PRON])
    d = f.groupby([CD, DESP]).agg(real=(REAL, "first"), err=("_e", "mean")).reset_index()

    piezas = []
    for cd, g in d.groupby(CD):
        g = g.set_index(DESP).sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx)
        p = pd.DataFrame(index=idx)
        p["err_7"] = g["err"].rolling(7, min_periods=3).mean().shift(avail_lag)
        p["err_14"] = g["err"].rolling(14, min_periods=5).mean().shift(avail_lag)
        p["real_m7"] = g["real"].rolling(7, min_periods=3).mean().shift(avail_lag)
        p["peak_thr"] = (
            g["real"].expanding(min_periods=MIN_HIST_PEAK).quantile(peak_q).shift(avail_lag)
        )
        p[CD] = cd
        p[EMI] = p.index
        piezas.append(p.reset_index(drop=True))
    return pd.concat(piezas, ignore_index=True)


def build_features(
    frame: pd.DataFrame, avail_lag: int = AVAIL_LAG, peak_q: float = PEAK_Q
) -> pd.DataFrame:
    """Devuelve el frame con features, dummies cd_* y objetivo `y`. Cada fila es causal."""
    if avail_lag < 1:
        raise ValueError("avail_lag debe ser >= 1: el real del dia de emision no se conoce")
    df = frame.copy()
    df[EMI] = pd.to_datetime(df[EMI])
    df[DESP] = pd.to_datetime(df[DESP])
    df["h_dias"] = df[HOR].astype(str).str.extract(r"(\d)")[0].astype(float)

    df = df.merge(_panel_diario(df, avail_lag, peak_q), on=[CD, EMI], how="left")

    # real del mismo dia de la semana anterior: real[d-7], conocido si d-7 <= t-avail_lag
    real_d = df.groupby([CD, DESP], as_index=False)[REAL].first().rename(columns={REAL: "real_lag7"})
    real_d[DESP] = real_d[DESP] + pd.Timedelta(days=7)
    df = df.merge(real_d, on=[CD, DESP], how="left")
    # si horizonte + avail_lag > 7, el real de d-7 aun no se conoce al emitir -> no disponible (queda en 0)
    df.loc[df["h_dias"] + avail_lag > 7, "real_lag7"] = np.nan

    lp = np.log1p(df[PRON])
    df["dia_semana"] = df[DESP].dt.dayofweek.astype(float)  # ordinal del despacho: 0 = lunes ... 6 = domingo
    df["finde"] = (df[DESP].dt.dayofweek >= 5).astype(float)
    df["pron_vs_rec7"] = lp - np.log1p(df["real_m7"])
    df["lag7_rel"] = np.log1p(df["real_lag7"]) - lp
    df["pico"] = (df[PRON] > df["peak_thr"]).astype(float)  # umbral NaN -> False
    # distancia al umbral antes de binarizar (>0 = sobre el umbral); sin historia -> 0
    df["pico_margen"] = (lp - np.log1p(df["peak_thr"])).fillna(0.0)

    # sin historia suficiente: valor neutro (0 = "sin correccion sugerida")
    for c in ["err_7", "err_14", "pron_vs_rec7", "lag7_rel"]:
        df[c] = df[c].fillna(0.0)

    for cd in sorted(df[CD].unique()):
        df[f"cd_{cd}"] = (df[CD] == cd).astype(float)

    df["y"] = np.log1p(df[REAL]) - lp
    return df.drop(columns=["real_m7", "peak_thr"])  # real_lag7 se conserva (seasonal naive)


def design_matrix(feat: pd.DataFrame, cols: list[str], with_cd: bool = True) -> pd.DataFrame:
    """Matriz X: columnas pedidas + dummies cd_* (creadas una vez en build_features)."""
    extra = [c for c in feat.columns if c.startswith("cd_")] if with_cd else []
    return feat[list(cols) + extra].astype(float)


def apply_delta(pron, delta) -> np.ndarray:
    """Pronostico corregido: expm1(log1p(pron) + delta), nunca negativo."""
    return np.clip(np.expm1(np.log1p(np.asarray(pron, dtype=float)) + delta), 0, None)
