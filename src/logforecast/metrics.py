"""Metricas del desafio.

WAPE = sum|real - pred| / sum|real|
Bias = sum(real - pred) / sum|real|   (>0 subestima, <0 sobreestima)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REAL = "real"
PRED = "unidades_pronosticadas"


def wape(real, pred) -> float:
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(real).sum()
    return float("nan") if denom == 0 else float(np.abs(real - pred).sum() / denom)


def bias(real, pred) -> float:
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(real).sum()
    return float("nan") if denom == 0 else float((real - pred).sum() / denom)


def error_table(
    df: pd.DataFrame,
    by: str | list[str] | None = None,
    real_col: str = REAL,
    pred_col: str = PRED,
) -> pd.DataFrame:
    """WAPE, Bias y n por grupo. Con by=None devuelve una sola fila (global)."""

    def _calc(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "wape": wape(g[real_col], g[pred_col]),
                "bias": bias(g[real_col], g[pred_col]),
                "n": len(g),
            }
        )

    if by is None:
        return _calc(df).to_frame().T
    by = [by] if isinstance(by, str) else list(by)
    return df.groupby(by, observed=True).apply(_calc, include_groups=False).reset_index()
