"""Capa de IO agnostica al almacenamiento: rutas locales o s3:// (via fsspec/s3fs)."""
from __future__ import annotations

import json
from typing import Any

import fsspec
import joblib
import pandas as pd

from logforecast.config import Settings

FILES = {
    "orders": "pedidos_historicos_genericos.csv",
    "forecasts": "pronostico_demanda_generico.csv",
    "capacity": "capacidad_logistica_generica.csv",
    "plan": "plan_entrega_clientes_generico.csv",
}


def join_uri(base: str, name: str) -> str:
    return base.rstrip("/\\") + "/" + name


def _read(settings: Settings, key: str, parse_dates: list[str]) -> pd.DataFrame:
    return pd.read_csv(join_uri(settings.data_uri, FILES[key]), parse_dates=parse_dates)


def load_orders(settings: Settings) -> pd.DataFrame:
    return _read(settings, "orders", ["fecha_despacho"])


def load_forecasts(settings: Settings) -> pd.DataFrame:
    return _read(settings, "forecasts", ["fecha_emision", "fecha_despacho"])


def load_capacity(settings: Settings) -> pd.DataFrame:
    return _read(settings, "capacity", ["fecha"])


def load_plan(settings: Settings) -> pd.DataFrame:
    return _read(settings, "plan", ["fecha_plan", "fecha_compra_real"])


def write_json(obj: Any, uri: str) -> None:
    with fsspec.open(uri, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(uri: str) -> Any:
    with fsspec.open(uri, "r") as f:
        return json.load(f)


def write_parquet(df: pd.DataFrame, uri: str) -> None:
    df.to_parquet(uri, index=False)


def save_model(obj: Any, uri: str) -> None:
    with fsspec.open(uri, "wb") as f:
        joblib.dump(obj, f)


def load_model(uri: str) -> Any:
    with fsspec.open(uri, "rb") as f:
        return joblib.load(f)
