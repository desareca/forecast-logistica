"""Chequeos de calidad de datos. Devuelve un reporte tabular (dataset, check, valor, ok)."""
from __future__ import annotations

import pandas as pd

HORIZON_DAYS = {"D+1": 1, "D+2": 2, "D+3": 3, "D+4": 4}


def _row(dataset: str, check: str, value, ok: bool) -> dict:
    return {"dataset": dataset, "check": check, "valor": value, "ok": bool(ok)}


def check_quality(
    orders: pd.DataFrame,
    forecasts: pd.DataFrame,
    capacity: pd.DataFrame,
    plan: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    datasets = {"orders": orders, "forecasts": forecasts, "capacity": capacity, "plan": plan}

    for name, df in datasets.items():
        rows.append(_row(name, "nulos_totales", int(df.isna().sum().sum()), df.isna().sum().sum() == 0))
        rows.append(_row(name, "filas_duplicadas", int(df.duplicated().sum()), df.duplicated().sum() == 0))

    rows.append(_row("orders", "unidades_<=0", int((orders["unidades_reales"] <= 0).sum()),
                     (orders["unidades_reales"] > 0).all()))
    rows.append(_row("forecasts", "pronostico_<=0", int((forecasts["unidades_pronosticadas"] <= 0).sum()),
                     (forecasts["unidades_pronosticadas"] > 0).all()))

    # cobertura de fechas por CD (huecos en la serie diaria)
    span = (orders["fecha_despacho"].max() - orders["fecha_despacho"].min()).days + 1
    days_per_cd = orders.groupby("centro_distribucion")["fecha_despacho"].nunique()
    rows.append(_row("orders", "dias_faltantes_max_por_cd", int(span - days_per_cd.min()),
                     (days_per_cd == span).all()))

    # horizonte coherente con fechas
    gap = (forecasts["fecha_despacho"] - forecasts["fecha_emision"]).dt.days
    expected = forecasts["horizonte"].map(HORIZON_DAYS)
    rows.append(_row("forecasts", "horizonte_incoherente", int((gap != expected).sum()), (gap == expected).all()))

    # pronosticos sin real asociado
    daily = orders.groupby(["centro_distribucion", "fecha_despacho"]).size().rename("n").reset_index()
    m = forecasts.merge(daily, on=["centro_distribucion", "fecha_despacho"], how="left")
    rows.append(_row("forecasts", "sin_real_asociado", int(m["n"].isna().sum()), m["n"].notna().all()))

    # flag fuera_de_plan consistente con las fechas
    flag = (plan["fecha_compra_real"] != plan["fecha_plan"]).astype(int)
    rows.append(_row("plan", "flag_inconsistente", int((flag != plan["fuera_de_plan"]).sum()),
                     (flag == plan["fuera_de_plan"]).all()))

    # cobertura del plan sobre el volumen total (informativo)
    cover = plan["unidades_reales"].sum() / orders["unidades_reales"].sum()
    rows.append(_row("plan", "cobertura_volumen_total", round(float(cover), 3), cover > 0.5))

    # cobertura de fines de semana en el plan
    weekend = (plan["fecha_plan"].dt.dayofweek >= 5).mean()
    rows.append(_row("plan", "pct_plan_fin_de_semana", round(float(weekend), 3), weekend > 0))

    # dias sobre capacidad
    d = orders.groupby(["centro_distribucion", "fecha_despacho"], as_index=False)["unidades_reales"].sum()
    d = d.merge(capacity, left_on=["centro_distribucion", "fecha_despacho"],
                right_on=["centro_distribucion", "fecha"], how="left")
    over = int((d["unidades_reales"] > d["capacidad_maxima_unidades"]).sum())
    rows.append(_row("capacity", "dias_cd_sobre_capacidad", over, over == 0))

    return pd.DataFrame(rows)
