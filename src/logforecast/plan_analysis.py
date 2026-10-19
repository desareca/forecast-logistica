"""Tarea 2: estado del plan de entrega (bias, adhesion, concentracion de desvios)."""
from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency

DOW_NAMES = {0: "Lun", 1: "Mar", 2: "Mie", 3: "Jue", 4: "Vie", 5: "Sab", 6: "Dom"}


def enrich_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Agrega variaciones de fecha/volumen y dia de semana planificado."""
    df = plan.copy()
    df["dow_plan"] = df["fecha_plan"].dt.dayofweek.map(DOW_NAMES)
    df["dias_desvio"] = (df["fecha_compra_real"] - df["fecha_plan"]).dt.days
    df["delta_unidades"] = df["unidades_reales"] - df["unidades_plan"]
    df["delta_pct"] = df["delta_unidades"] / df["unidades_plan"]
    return df


def plan_bias(plan: pd.DataFrame) -> dict[str, float]:
    """Bias de unidades (>0: el plan subestima lo real) y sesgo medio de fecha."""
    df = enrich_plan(plan)
    return {
        "bias_unidades": float(df["delta_unidades"].sum() / df["unidades_reales"].abs().sum()),
        "wape_unidades": float(df["delta_unidades"].abs().sum() / df["unidades_reales"].abs().sum()),
        "desvio_fecha_medio_dias": float(df["dias_desvio"].mean()),
        "desvio_fecha_mediano_dias": float(df["dias_desvio"].median()),
    }


def out_of_plan_rate(plan: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    df = enrich_plan(plan)
    if by is None:
        return pd.DataFrame({"fuera_de_plan": [df["fuera_de_plan"].mean()], "n": [len(df)]})
    return (
        df.groupby(by)["fuera_de_plan"]
        .agg(fuera_de_plan="mean", n="size")
        .reset_index()
        .sort_values("fuera_de_plan", ascending=False)
    )


def out_of_plan_chi2(plan: pd.DataFrame, by: str) -> dict[str, float]:
    """Test chi-cuadrado de independencia entre `by` y fuera_de_plan.

    H0: la tasa de compras fuera de plan es la misma en todos los grupos de `by`.
    Acepta columnas de `enrich_plan` (p. ej. 'dow_plan').
    """
    df = enrich_plan(plan)
    tabla = pd.crosstab(df[by], df["fuera_de_plan"])
    chi2, p, gl, _ = chi2_contingency(tabla)
    return {"chi2": float(chi2), "gl": int(gl), "p_valor": float(p), "n": int(tabla.values.sum())}


def client_adherence(plan: pd.DataFrame) -> pd.DataFrame:
    """Adhesion por cliente: % en plan, desvio medio de fecha y de volumen."""
    df = enrich_plan(plan)
    out = df.groupby(["cliente_id", "centro_distribucion"]).agg(
        compras=("fuera_de_plan", "size"),
        pct_fuera_plan=("fuera_de_plan", "mean"),
        dias_desvio_medio=("dias_desvio", "mean"),
        delta_pct_medio=("delta_pct", "mean"),
    )
    out["adhesion"] = 1 - out["pct_fuera_plan"]
    return out.reset_index().sort_values("pct_fuera_plan", ascending=False)


def deviation_concentration(plan: pd.DataFrame, top_share: float = 0.2) -> float:
    """% de las unidades desviadas (|real-plan|) explicado por el top X% de clientes."""
    df = enrich_plan(plan)
    dev = df.assign(abs_dev=df["delta_unidades"].abs()).groupby("cliente_id")["abs_dev"].sum()
    dev = dev.sort_values(ascending=False)
    k = max(1, int(round(len(dev) * top_share)))
    return float(dev.head(k).sum() / dev.sum())


def plan_coverage(plan: pd.DataFrame, orders: pd.DataFrame) -> dict[str, float]:
    """Alcance del plan: clientes, % del volumen total de pedidos y % de fechas en fin de semana."""
    return {
        "clientes": int(plan["cliente_id"].nunique()),
        "cobertura_volumen": float(plan["unidades_reales"].sum() / orders["unidades_reales"].sum()),
        "pct_plan_fin_de_semana": float((plan["fecha_plan"].dt.dayofweek >= 5).mean()),
    }


def plan_bias_detail(plan: pd.DataFrame) -> dict[str, float]:
    """Cuantas compras y cuantos clientes compran mas de lo planificado (en unidades)."""
    df = enrich_plan(plan)
    cl = df.groupby("cliente_id")[["unidades_reales", "unidades_plan"]].sum()
    return {
        "pct_compras_real_sobre_plan": float((df["delta_unidades"] > 0).mean()),
        "clientes_real_sobre_plan": int((cl["unidades_reales"] > cl["unidades_plan"]).sum()),
        "clientes": int(len(cl)),
    }


def plan_bias_by(plan: pd.DataFrame, by: str) -> pd.DataFrame:
    """Bias de unidades del plan por grupo (>0: el plan subestima lo real)."""
    df = enrich_plan(plan).assign(abs_real=lambda d: d["unidades_reales"].abs())
    out = df.groupby(by).agg(
        delta=("delta_unidades", "sum"), den=("abs_real", "sum"), n=("delta_unidades", "size")
    )
    out["bias_unidades"] = out["delta"] / out["den"]
    return out[["bias_unidades", "n"]].reset_index().sort_values("bias_unidades", ascending=False)


def out_of_plan_date_shift(plan: pd.DataFrame) -> dict[str, float]:
    """Direccion y magnitud del desvio de fecha en compras fuera de plan, y dia en que aterrizan."""
    f = enrich_plan(plan).query("fuera_de_plan == 1")
    d = f["dias_desvio"]
    dow_real = f["fecha_compra_real"].dt.dayofweek
    return {
        "n_fuera_de_plan": int(len(f)),
        "pct_despues": float((d > 0).mean()),
        "pct_antes": float((d < 0).mean()),
        "desvio_medio_neto_dias": float(d.mean()),
        "atraso_medio_dias": float(d[d > 0].mean()),
        "atraso_max_dias": int(d.max()),
        "adelanto_min_dias": int(d[d < 0].max()),
        "adelanto_max_dias": int(d.min()),
        "pct_aterriza_martes": float((dow_real == 1).mean()),
        "pct_aterriza_fin_de_semana": float((dow_real >= 5).mean()),
    }


def adherence_summary(plan: pd.DataFrame, umbral: float = 0.3) -> dict[str, float]:
    """Clientes que nunca se salen del plan, los que se salen en >= `umbral` de sus compras y el maximo."""
    ca = client_adherence(plan)
    return {
        "clientes_nunca_fuera": int((ca["pct_fuera_plan"] == 0).sum()),
        f"clientes_fuera_>={umbral:.0%}": int((ca["pct_fuera_plan"] >= umbral).sum()),
        "max_pct_fuera_plan": float(ca["pct_fuera_plan"].max()),
    }


def out_of_plan_concentration(plan: pd.DataFrame, top_share: float = 0.2) -> float:
    """% de las compras fuera de plan hechas por el top X% de clientes (ordenados por n fuera de plan)."""
    cnt = plan.groupby("cliente_id")["fuera_de_plan"].sum().sort_values(ascending=False)
    k = max(1, int(round(len(cnt) * top_share)))
    return float(cnt.head(k).sum() / cnt.sum())


def weekday_share(plan: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """% de unidades por dia de semana: plan (fecha_plan), real del plan (fecha_compra_real)
    y pedidos totales (fecha_despacho). `uniforme` = 1/7 como referencia."""
    def share(fechas: pd.Series, unidades: pd.Series) -> pd.Series:
        return unidades.groupby(fechas.dt.dayofweek).sum() / unidades.sum()

    out = pd.DataFrame({
        "pct_unidades_plan": share(plan["fecha_plan"], plan["unidades_plan"]),
        "pct_unidades_reales_plan": share(plan["fecha_compra_real"], plan["unidades_reales"]),
        "pct_pedidos_totales": share(orders["fecha_despacho"], orders["unidades_reales"]),
    }).reindex(range(7)).fillna(0.0)
    out["uniforme"] = 1 / 7
    out.index = out.index.map(DOW_NAMES)
    out.index.name = "dia"
    return out
