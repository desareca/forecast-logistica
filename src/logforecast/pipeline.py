"""Orquestacion de alto nivel. Mismas funciones desde notebook, CLI (ECS) y Lambda."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from logforecast import io
from logforecast.capacity import capacity_alert
from logforecast.config import Settings
from logforecast.diagnostics import build_eval_frame
from logforecast.model import ElasticNetCorrector, GatedCorrector

log = logging.getLogger(__name__)

# Columnas obligatorias de cada registro de inferencia (las features del modelo las necesitan)
REQUIRED = ["centro_distribucion", "fecha_emision", "fecha_despacho", "horizonte", "unidades_pronosticadas"]


def load_eval_frame(settings: Settings) -> pd.DataFrame:
    return build_eval_frame(io.load_orders(settings), io.load_forecasts(settings))


def train(settings: Settings) -> dict:
    """Entrena con toda la historia disponible y persiste modelo + estado del gate.

    El artefacto incluye la historia que necesitan las features de inferencia: correr a diario.
    """
    frame = load_eval_frame(settings)
    model = GatedCorrector(ElasticNetCorrector, settings.min_history_days).fit(frame)

    io.save_model(model, io.join_uri(settings.artifacts_uri, "model.joblib"))
    gate = {cd: d.to_dict() for cd, d in model.decisions_.items()}
    io.write_json(gate, io.join_uri(settings.artifacts_uri, "gate_status.json"))
    return gate


def predict(
    settings: Settings,
    records: pd.DataFrame,
    model: GatedCorrector | None = None,
    capacity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Corrige pronosticos. records: las columnas de REQUIRED (una fila por CD, emision, despacho y horizonte).

    Un CD desconocido o en fallback (cold start) recibe el pronostico base, sin excepcion.
    Si el modelo falla al predecir, devuelve el base con corregido = False y registra el error.
    Si falta una columna obligatoria lanza ValueError (el handler responde 400).
    Si se entrega `capacity` (tabla de capacidad por CD y dia), agrega `capacidad_maxima`,
    `uso_capacidad` y `alerta_capacidad` sobre la prediccion corregida. No modifica las unidades.
    """
    if model is None:
        model = io.load_model(io.join_uri(settings.artifacts_uri, "model.joblib"))
    faltan = [c for c in REQUIRED if c not in records.columns]
    if faltan:
        raise ValueError(f"faltan columnas: {faltan}")
    out = records.copy()
    try:
        pred = np.asarray(model.predict(out), dtype=float)
        corregido = out["centro_distribucion"].isin(getattr(model, "trained_centers_", set()))
    except Exception:  # noqa: BLE001 - una falla del modelo no debe dejar a la app sin numero
        log.exception("fallo la prediccion del modelo; se devuelve el pronostico base")
        pred = out["unidades_pronosticadas"].to_numpy(float)
        corregido = pd.Series(False, index=out.index)
    out["unidades_corregidas"] = np.round(pred, 2)
    out["corregido"] = corregido
    if capacity is not None:
        out = capacity_alert(out, capacity, "unidades_corregidas", settings.capacity_warn_ratio)
    return out
