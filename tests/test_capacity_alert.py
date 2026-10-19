import json

import numpy as np
import pandas as pd

from logforecast import capacity as C
from logforecast import pipeline
from logforecast.config import Settings
from logforecast.handlers import lambda_handler as H


def _cap():
    return pd.DataFrame({
        "centro_distribucion": ["A", "A", "B"],
        "fecha": pd.to_datetime(["2026-03-03", "2026-03-04", "2026-03-03"]),
        "capacidad_maxima_unidades": [100.0, 100.0, 200.0],
    })


def _recs():
    return pd.DataFrame({
        "centro_distribucion": ["A", "A", "A", "B", "C"],
        "fecha_despacho": ["2026-03-03", "2026-03-04", "2026-03-04", "2026-03-03", "2026-03-03"],
        "unidades_corregidas": [50.0, 95.0, 120.0, 190.0, 10.0],
    })


def test_alert_levels():
    r = C.capacity_alert(_recs(), _cap(), warn_ratio=0.9)
    assert list(r["alerta_capacidad"]) == ["OK", "CERCA", "EXCEDE", "CERCA", "SIN_CAPACIDAD"]
    assert r["capacidad_maxima"].iloc[:4].tolist() == [100.0, 100.0, 100.0, 200.0]
    assert np.isnan(r["capacidad_maxima"].iloc[4])


def test_alert_never_raises():
    sin_fecha = _recs().drop(columns="fecha_despacho")
    assert set(C.capacity_alert(sin_fecha, _cap())["alerta_capacidad"]) == {"SIN_CAPACIDAD"}
    assert set(C.capacity_alert(_recs(), None)["alerta_capacidad"]) == {"SIN_CAPACIDAD"}
    fechas_malas = _recs().assign(fecha_despacho="no-es-fecha")
    assert set(C.capacity_alert(fechas_malas, _cap())["alerta_capacidad"]) == {"SIN_CAPACIDAD"}


class _FakeModel:
    trained_centers_ = {"A", "B"}

    def predict(self, df):
        return df["unidades_pronosticadas"].to_numpy() * 1.1


def test_predict_with_capacity_keeps_units():
    recs = pd.DataFrame({"centro_distribucion": ["A", "A"], "fecha_emision": ["2026-03-02", "2026-03-02"],
                         "fecha_despacho": ["2026-03-03", "2026-03-04"], "horizonte": ["D+1", "D+2"],
                         "unidades_pronosticadas": [100.0, 100.0]})
    out = pipeline.predict(Settings(), recs, model=_FakeModel(), capacity=_cap())
    assert out["unidades_corregidas"].tolist() == [110.0, 110.0]  # no se recorta
    assert out["alerta_capacidad"].tolist() == ["EXCEDE", "EXCEDE"]


def test_handler_returns_valid_json_with_missing_capacity(monkeypatch):
    monkeypatch.setattr(H, "_get_model", lambda s: _FakeModel())
    monkeypatch.setattr(H, "_get_capacity", lambda s: _cap())
    ev = {"body": json.dumps({"records": [
        {"centro_distribucion": "A", "fecha_emision": "2026-03-02", "fecha_despacho": "2026-03-03",
         "horizonte": "D+1", "unidades_pronosticadas": 50.0},
        {"centro_distribucion": "C", "fecha_emision": "2026-03-02", "fecha_despacho": "2026-03-03",
         "horizonte": "D+1", "unidades_pronosticadas": 50.0}]})}
    resp = H.handler(ev)
    assert resp["statusCode"] == 200
    recs = json.loads(resp["body"])["records"]  # json.loads falla con NaN? no: debe venir null
    assert recs[0]["alerta_capacidad"] == "OK" and recs[0]["capacidad_maxima"] == 100.0
    assert recs[1]["alerta_capacidad"] == "SIN_CAPACIDAD" and recs[1]["capacidad_maxima"] is None
    assert "NaN" not in resp["body"]


def test_handler_survives_capacity_failure(monkeypatch):
    monkeypatch.setattr(H, "_get_model", lambda s: _FakeModel())
    monkeypatch.setattr(H, "_CAPACITY", None)
    monkeypatch.setattr(H.io, "load_capacity", lambda s: (_ for _ in ()).throw(FileNotFoundError("sin archivo")))
    ev = {"records": [{"centro_distribucion": "A", "fecha_emision": "2026-03-02", "fecha_despacho": "2026-03-03",
                       "horizonte": "D+1", "unidades_pronosticadas": 50.0}]}
    resp = H.handler(ev)
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["records"][0]["alerta_capacidad"] == "SIN_CAPACIDAD"


def test_handler_400_when_required_column_missing(monkeypatch):
    monkeypatch.setattr(H, "_get_model", lambda s: _FakeModel())
    monkeypatch.setattr(H, "_get_capacity", lambda s: _cap())
    ev = {"records": [{"centro_distribucion": "A", "fecha_despacho": "2026-03-03", "unidades_pronosticadas": 50.0}]}
    resp = H.handler(ev)
    assert resp["statusCode"] == 400 and "fecha_emision" in resp["body"]


def _ev_ok():
    return {"records": [{"centro_distribucion": "A", "fecha_emision": "2026-03-02", "fecha_despacho": "2026-03-03",
                         "horizonte": "D+1", "unidades_pronosticadas": 50.0}]}


def test_handler_returns_base_when_model_cannot_load(monkeypatch):
    def _falla(s):
        raise FileNotFoundError("sin model.joblib")
    monkeypatch.setattr(H, "_get_model", _falla)
    monkeypatch.setattr(H, "_get_capacity", lambda s: _cap())
    resp = H.handler(_ev_ok())
    assert resp["statusCode"] == 200
    r = json.loads(resp["body"])["records"][0]
    assert r["unidades_corregidas"] == 50.0 and r["corregido"] is False


def test_predict_returns_base_when_model_raises():
    class _Roto:
        trained_centers_ = {"A"}

        def predict(self, df):
            raise RuntimeError("features invalidas")

    recs = pd.DataFrame(_ev_ok()["records"])
    out = pipeline.predict(Settings(), recs, model=_Roto())
    assert out["unidades_corregidas"].tolist() == [50.0] and not out["corregido"].any()
