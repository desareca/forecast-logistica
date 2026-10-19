"""Tarea 4: gate de historia minima (cold start).

Regla: un CD con < min_days de historia CONTINUA (sin huecos) usa FALLBACK
(pronostico base sin correccion). Con >= min_days habilita el modelo.
Nunca lanza excepciones: ante cualquier problema devuelve FALLBACK con la razon.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


class GateStatus(str, Enum):
    TRAIN = "TRAIN"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class GateDecision:
    centro_distribucion: str
    status: GateStatus
    continuous_days: int
    min_days: int
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def continuous_history_days(dates: Iterable) -> int:
    """Dias consecutivos de historia que terminan en la ultima fecha disponible."""
    s = pd.Series(pd.to_datetime(list(dates))).dt.normalize().drop_duplicates().sort_values()
    s = s.reset_index(drop=True)
    if s.empty:
        return 0
    gaps = s.diff().dt.days.fillna(1)
    breaks = gaps[gaps > 1]
    start = s.iloc[breaks.index[-1]] if len(breaks) else s.iloc[0]
    return int((s.iloc[-1] - start).days) + 1


def evaluate_gate(
    daily: pd.DataFrame,
    min_days: int = 45,
    centers: Iterable[str] | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict[str, GateDecision]:
    """Evalua el gate por CD.

    daily: columnas centro_distribucion, fecha_despacho (una fila por dia y CD basta).
    centers: CDs a evaluar (los que no tengan historia quedan en FALLBACK con 0 dias).
    as_of: solo se usa historia con fecha <= as_of (evita fuga en backtest).
    """
    decisions: dict[str, GateDecision] = {}
    try:
        df = daily if as_of is None else daily[daily["fecha_despacho"] <= pd.Timestamp(as_of)]
        names = list(centers) if centers is not None else sorted(df["centro_distribucion"].unique())
        for cd in names:
            try:
                days = continuous_history_days(df.loc[df["centro_distribucion"] == cd, "fecha_despacho"])
                ok = days >= min_days
                decisions[cd] = GateDecision(
                    cd,
                    GateStatus.TRAIN if ok else GateStatus.FALLBACK,
                    days,
                    min_days,
                    "historia suficiente" if ok else f"historia insuficiente ({days} < {min_days} dias)",
                )
            except Exception as exc:  # noqa: BLE001 - el gate no debe romper el flujo
                log.warning("gate fallo para %s: %s", cd, exc)
                decisions[cd] = GateDecision(cd, GateStatus.FALLBACK, 0, min_days, f"error: {exc}")
    except Exception as exc:  # noqa: BLE001
        log.warning("gate fallo globalmente: %s", exc)
    return decisions
