import pandas as pd
import pytest

from logforecast import capacity as C
from logforecast import features as F


def _frame():
    return pd.DataFrame({
        F.CD: ["A", "A", "B"],
        F.DESP: pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-05"]),
        F.PRON: [120.0, 80.0, 50.0],
        F.REAL: [90.0, 85.0, 60.0],
    })


def _cap():
    return pd.DataFrame({
        "centro_distribucion": ["A", "A", "B"],
        "fecha": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-05"]),
        "capacidad_maxima_unidades": [100.0, 100.0, 100.0],
    })


def test_attach_and_cap_effect():
    d = C.attach_capacity(_frame(), _cap())
    assert list(d["uso_pron"].round(2)) == [1.2, 0.8, 0.5]
    assert list(C.capped(d)) == [100.0, 80.0, 50.0]
    r = C.cap_effect(d)
    # real total 235: error base 30+5+10=45, con cap 10+5+10=25
    assert r.loc["GLOBAL", "WAPE sin cap (%)"] == pytest.approx(45 / 235 * 100, abs=0.01)
    assert r.loc["GLOBAL", "WAPE con cap (%)"] == pytest.approx(25 / 235 * 100, abs=0.01)
    assert r.loc["GLOBAL", "filas recortadas (%)"] == pytest.approx(100 / 3, abs=0.01)


def test_missing_capacity_raises():
    with pytest.raises(ValueError):
        C.attach_capacity(_frame(), _cap().iloc[:2])


def test_exceed_matrix_and_profile():
    d = C.attach_capacity(_frame(), _cap())
    m = C.exceed_matrix(d)
    assert m.loc["pron > cap", "total"] == 1
    p = C.excess_rows_profile(d)
    assert p["filas"] == 1 and p["% donde el real tambien excede"] == 0
