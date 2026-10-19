"""Handler de inferencia para AWS Lambda (detras de API Gateway).

Request  (body JSON): {"records": [{"centro_distribucion": "CD_SUR", "fecha_emision": "2026-03-02",
                                    "fecha_despacho": "2026-03-03", "horizonte": "D+1",
                                    "unidades_pronosticadas": 1500.0}]}
Response (body JSON): {"records": [{..., "unidades_corregidas": ..., "corregido": true,
                                    "capacidad_maxima": 2640.0, "uso_capacidad": 0.57,
                                    "alerta_capacidad": "OK"}]}
alerta_capacidad: OK | CERCA (uso >= LF_CAPACITY_WARN_RATIO) | EXCEDE (uso > 1) | SIN_CAPACIDAD.
Las unidades no se recortan: el consumidor decide que hacer con la alerta.
El modelo (ElasticNet con gate, e historia para las features) se carga una vez por contenedor
(warm start) desde LF_ARTIFACTS_URI. Si no se puede cargar o falla, responde con el pronostico
base y corregido = false (se reintenta la carga en el siguiente request). Falta una columna -> 400.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from logforecast import io, pipeline
from logforecast.config import Settings
from logforecast.model import BaseOnlyCorrector

log = logging.getLogger(__name__)
_MODEL = None
_CAPACITY = None


def _get_model(settings: Settings):
    global _MODEL
    if _MODEL is None:
        _MODEL = io.load_model(io.join_uri(settings.artifacts_uri, "model.joblib"))
    return _MODEL


def _get_model_or_base(settings: Settings):
    """Modelo vigente; si no se puede cargar, un corrector que devuelve el base (sin cachearlo)."""
    try:
        return _get_model(settings)
    except Exception:  # noqa: BLE001
        log.exception("no se pudo cargar el modelo; se responde con el pronostico base")
        return BaseOnlyCorrector()


def _get_capacity(settings: Settings):
    """Tabla de capacidad (cache por contenedor). Si no se puede leer, el endpoint responde igual sin alerta."""
    global _CAPACITY
    if _CAPACITY is None:
        try:
            _CAPACITY = io.load_capacity(settings)
        except Exception:  # noqa: BLE001 - la capacidad es informativa, no debe tumbar la inferencia
            log.exception("no se pudo leer la capacidad; alerta_capacidad = SIN_CAPACIDAD")
            return pd.DataFrame()  # vacia: la respuesta mantiene los campos, con SIN_CAPACIDAD (se reintenta luego)
    return _CAPACITY


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    try:
        body = event.get("body", event)
        payload = json.loads(body) if isinstance(body, str) else body
        records = pd.DataFrame(payload["records"])
        settings = Settings.from_env()
        out = pipeline.predict(settings, records, model=_get_model_or_base(settings), capacity=_get_capacity(settings))
        out = out.astype(object).where(out.notna(), None)  # NaN -> null (JSON valido)
        return {"statusCode": 200, "body": json.dumps({"records": out.to_dict("records")}, default=str)}
    except (KeyError, ValueError, TypeError) as exc:
        return {"statusCode": 400, "body": json.dumps({"error": f"payload invalido: {exc}"})}
    except Exception as exc:  # noqa: BLE001
        log.exception("error en inferencia")
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
