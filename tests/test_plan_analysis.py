"""Tests del test chi-cuadrado de compras fuera de plan."""
import pandas as pd

from logforecast.plan_analysis import out_of_plan_chi2


def _plan(tasas: dict[str, float], n: int = 200) -> pd.DataFrame:
    filas = []
    for cd, tasa in tasas.items():
        k = int(n * tasa)
        for i in range(n):
            f = pd.Timestamp("2026-01-05") + pd.Timedelta(days=i % 5)
            filas.append({"cliente_id": f"{cd}_{i}", "centro_distribucion": cd, "fecha_plan": f,
                          "fecha_compra_real": f, "unidades_plan": 100.0, "unidades_reales": 100.0,
                          "fuera_de_plan": int(i < k)})
    return pd.DataFrame(filas)


def test_chi2_detects_different_rates():
    r = out_of_plan_chi2(_plan({"A": 0.05, "B": 0.40}), "centro_distribucion")
    assert r["p_valor"] < 0.001 and r["gl"] == 1 and r["n"] == 400


def test_chi2_no_difference_when_rates_equal():
    r = out_of_plan_chi2(_plan({"A": 0.20, "B": 0.20}), "centro_distribucion")
    assert r["p_valor"] > 0.5


def test_chi2_accepts_enriched_column():
    r = out_of_plan_chi2(_plan({"A": 0.20}), "dow_plan")
    assert r["gl"] == 4


def _plan_mini() -> pd.DataFrame:
    """4 compras de 2 clientes: 2 fuera de plan (+3 dias a un martes, -1 dia)."""
    t = pd.Timestamp
    return pd.DataFrame({
        "cliente_id": ["C1", "C1", "C2", "C2"],
        "centro_distribucion": ["A", "A", "B", "B"],
        "canal_venta": ["W", "W", "W", "W"],
        "fecha_plan": [t("2026-01-05"), t("2026-01-09"), t("2026-01-07"), t("2026-01-08")],  # lun, vie, mie, jue
        "fecha_compra_real": [t("2026-01-05"), t("2026-01-12") + pd.Timedelta(days=1), t("2026-01-07"), t("2026-01-07")],
        "unidades_plan": [100.0, 100.0, 100.0, 100.0],
        "unidades_reales": [120.0, 150.0, 90.0, 100.0],
        "fuera_de_plan": [0, 1, 0, 1],
    })


def test_date_shift_and_landing():
    from logforecast.plan_analysis import out_of_plan_date_shift
    r = out_of_plan_date_shift(_plan_mini())
    assert r["n_fuera_de_plan"] == 2 and r["pct_despues"] == 0.5 and r["pct_antes"] == 0.5
    assert r["atraso_medio_dias"] == 4 and r["desvio_medio_neto_dias"] == 1.5
    assert r["pct_aterriza_martes"] == 0.5 and r["pct_aterriza_fin_de_semana"] == 0


def test_bias_detail_and_by_group():
    from logforecast.plan_analysis import plan_bias_by, plan_bias_detail
    p = _plan_mini()
    r = plan_bias_detail(p)
    assert r["pct_compras_real_sobre_plan"] == 0.5 and r["clientes_real_sobre_plan"] == 1
    b = plan_bias_by(p, "centro_distribucion").set_index("centro_distribucion")["bias_unidades"]
    assert abs(b["A"] - 70 / 270) < 1e-9 and abs(b["B"] - (-10 / 190)) < 1e-9


def test_adherence_concentration_weekday_coverage():
    from logforecast.plan_analysis import (adherence_summary, out_of_plan_concentration,
                                           plan_coverage, weekday_share)
    p = _plan_mini()
    a = adherence_summary(p)
    assert a["clientes_nunca_fuera"] == 0 and a["max_pct_fuera_plan"] == 0.5
    assert out_of_plan_concentration(p, top_share=0.5) == 0.5
    orders = pd.DataFrame({"fecha_despacho": p["fecha_compra_real"], "unidades_reales": p["unidades_reales"]})
    w = weekday_share(p, orders)
    assert abs(w.drop(columns="uniforme").sum() - 1).max() < 1e-9 and len(w) == 7
    c = plan_coverage(p, orders)
    assert c["cobertura_volumen"] == 1 and c["pct_plan_fin_de_semana"] == 0
