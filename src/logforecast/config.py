"""Configuracion central. Todo se parametriza por variables de entorno.

Local:  LF_DATA_URI=/app/data/raw        LF_ARTIFACTS_URI=/app/artifacts
AWS:    LF_DATA_URI=s3://bucket/raw      LF_ARTIFACTS_URI=s3://bucket/artifacts
El codigo no cambia entre ambos entornos, solo las variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    data_uri: str = "data/raw"
    artifacts_uri: str = "artifacts"
    min_history_days: int = 45
    bias_tolerance: float = 0.05
    capacity_warn_ratio: float = 0.9  # uso de la capacidad desde el cual el endpoint avisa "CERCA"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_uri=os.getenv("LF_DATA_URI", cls.data_uri),
            artifacts_uri=os.getenv("LF_ARTIFACTS_URI", cls.artifacts_uri),
            min_history_days=int(os.getenv("LF_MIN_HISTORY_DAYS", cls.min_history_days)),
            bias_tolerance=float(os.getenv("LF_BIAS_TOLERANCE", cls.bias_tolerance)),
            capacity_warn_ratio=float(os.getenv("LF_CAPACITY_WARN_RATIO", cls.capacity_warn_ratio)),
        )
