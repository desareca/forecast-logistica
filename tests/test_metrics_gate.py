import pandas as pd

from logforecast.gate import GateStatus, continuous_history_days, evaluate_gate
from logforecast.metrics import bias, wape


def test_wape_bias_signs():
    real, pred = [100, 100], [80, 80]
    assert wape(real, pred) == 0.2
    assert bias(real, pred) == 0.2  # positivo = subestima


def test_continuous_days_resets_on_gap():
    dates = list(pd.date_range("2026-01-01", periods=10)) + list(pd.date_range("2026-01-15", periods=5))
    assert continuous_history_days(dates) == 5


def test_gate_fallback_and_train_without_exceptions():
    long = pd.DataFrame({"centro_distribucion": "A", "fecha_despacho": pd.date_range("2026-01-01", periods=45)})
    short = pd.DataFrame({"centro_distribucion": "B", "fecha_despacho": pd.date_range("2026-01-01", periods=10)})
    d = evaluate_gate(pd.concat([long, short]), 45, centers=["A", "B", "C"])
    assert d["A"].status == GateStatus.TRAIN
    assert d["B"].status == GateStatus.FALLBACK
    assert d["C"].status == GateStatus.FALLBACK and d["C"].continuous_days == 0


def test_gate_empty_input_does_not_raise():
    empty = pd.DataFrame({"centro_distribucion": [], "fecha_despacho": pd.to_datetime([])})
    assert evaluate_gate(empty, 45, centers=["A"])["A"].status == GateStatus.FALLBACK
