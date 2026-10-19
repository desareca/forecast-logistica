"""Tarea 1: diagnostico de calidad del forecast base y backtesting simple."""
from __future__ import annotations

import numpy as np
import pandas as pd

from logforecast.metrics import error_table


def daily_actuals(orders: pd.DataFrame) -> pd.DataFrame:
    """Demanda real diaria por CD (suma de canales)."""
    return (
        orders.groupby(["centro_distribucion", "fecha_despacho"], as_index=False)["unidades_reales"]
        .sum()
        .rename(columns={"unidades_reales": "real"})
    )


def build_eval_frame(orders: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Une cada pronostico (CD, fecha_despacho, horizonte) con su real diario."""
    frame = forecasts.merge(
        daily_actuals(orders), on=["centro_distribucion", "fecha_despacho"], how="inner"
    )
    return frame.sort_values(["centro_distribucion", "fecha_emision", "fecha_despacho"]).reset_index(
        drop=True
    )


def error_by_cd(frame: pd.DataFrame) -> pd.DataFrame:
    return error_table(frame, "centro_distribucion")


def error_by_horizon(frame: pd.DataFrame) -> pd.DataFrame:
    return error_table(frame, "horizonte")


def error_by_cd_horizon(frame: pd.DataFrame) -> pd.DataFrame:
    return error_table(frame, ["centro_distribucion", "horizonte"])


def common_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Filtra a los (CD, fecha_despacho) que tienen pronostico en todos los horizontes."""
    keys = ["centro_distribucion", "fecha_despacho"]
    n_h = frame["horizonte"].nunique()
    full = frame.groupby(keys)["horizonte"].transform("nunique") == n_h
    return frame[full].reset_index(drop=True)


def horizon_wape_common_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """WAPE por horizonte con todas las fechas vs solo fechas comunes a los horizontes.

    Controla que la degradacion D+1 -> D+4 no se deba a que cada horizonte evalua
    un conjunto distinto de fechas de despacho.
    """
    todas = error_by_horizon(frame).set_index("horizonte")[["wape", "n"]]
    comunes = error_by_horizon(common_dates(frame)).set_index("horizonte")[["wape", "n"]]
    out = todas.join(comunes, lsuffix="_todas", rsuffix="_comunes")
    out["dif_pp"] = (out["wape_comunes"] - out["wape_todas"]) * 100
    return out.reset_index()


def bias_stability(frame: pd.DataFrame, corte) -> pd.DataFrame:
    """Bias y WAPE del base por CD en la primera vs segunda mitad de los despachos <= `corte`.

    Bias = (real - pronostico) / real total; positivo = el base subestima. Si el signo y la
    magnitud se mantienen entre mitades, un factor por CD es defendible.
    """
    ft = frame[frame["fecha_despacho"] <= pd.Timestamp(corte)]
    fechas = np.sort(ft["fecha_despacho"].unique())
    mitad = fechas[len(fechas) // 2]
    g = ft.assign(
        mitad=np.where(ft["fecha_despacho"] <= mitad, "1ra", "2da"),
        err=ft["real"] - ft["unidades_pronosticadas"],
    )
    g["abs"] = g["err"].abs()
    s = g.groupby(["centro_distribucion", "mitad"])[["err", "abs", "real"]].sum()
    return pd.DataFrame({"bias": s["err"] / s["real"], "wape": s["abs"] / s["real"]}).unstack("mitad")


def classify_bias(table: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """Etiqueta cada fila: subestima (bias > tol), sobreestima (bias < -tol) o neutro."""
    out = table.copy()
    out["tendencia"] = "neutro"
    out.loc[out["bias"] > tol, "tendencia"] = "subestima"
    out.loc[out["bias"] < -tol, "tendencia"] = "sobreestima"
    return out
