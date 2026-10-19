"""Modelos de correccion. Interfaz comun: fit(train_df) / predict(df) -> Series.

train_df: eval frame (diagnostics.build_eval_frame), con centro_distribucion, fecha_emision,
fecha_despacho, horizonte, unidades_pronosticadas y real.
  - ElasticNetCorrector: modelo complementario de la Tarea 3 (el que usa el pipeline).
  - RatioCorrector: factor por CD; se mantiene como baseline simple.
  - GatedCorrector: aplica el gate de historia minima (Tarea 4) sobre cualquiera de los dos.
"""
from __future__ import annotations

import logging
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from logforecast import features as F
from logforecast.cv import chosen_spec
from logforecast.diagnostics import daily_actuals  # noqa: F401  (re-export util)
from logforecast.evaluation import make_estimator
from logforecast.gate import GateStatus, evaluate_gate

log = logging.getLogger(__name__)


class Corrector(Protocol):
    def fit(self, train: pd.DataFrame) -> "Corrector": ...
    def predict(self, df: pd.DataFrame) -> pd.Series: ...


class RatioCorrector:
    """Baseline: factor multiplicativo por CD = sum(real) / sum(pronostico).

    CDs no vistos en fit reciben factor 1 (sin correccion).
    """

    def __init__(self, clip: tuple[float, float] = (0.5, 2.0)):
        self.clip = clip
        self.factors_: dict[str, float] = {}

    def fit(self, train: pd.DataFrame) -> "RatioCorrector":
        g = train.groupby("centro_distribucion")
        ratio = g["real"].sum() / g["unidades_pronosticadas"].sum()
        self.factors_ = ratio.clip(*self.clip).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        f = df["centro_distribucion"].map(self.factors_).fillna(1.0)
        return df["unidades_pronosticadas"] * f


class ElasticNetCorrector:
    """Modelo complementario de la Tarea 3: el ElasticNet elegido por CV (`cv.ELEGIDO`).

    Predice y = log1p(real) - log1p(pronostico) con las features de `features.build_features` y
    corrige con expm1(log1p(pron) + delta), acotado a >= 0.

    Las features de inferencia (errores moviles, demanda reciente, umbral de pico, real de la semana
    anterior) necesitan los reales recientes de cada CD. Por eso `fit` guarda, junto al estimador,
    la historia con la que entreno (`history_`): `predict` la concatena con las filas pedidas y
    calcula las features con la misma funcion que el backtest. Como la historia viaja dentro del
    artefacto, el batch debe reentrenar a diario para que la inferencia vea los reales nuevos.
    """

    KEYS = [F.CD, F.EMI, F.DESP, F.HOR, F.PRON]

    def __init__(self, spec: dict | None = None, avail_lag: int = F.AVAIL_LAG):
        self.spec = spec
        self.avail_lag = avail_lag

    @staticmethod
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[F.EMI] = pd.to_datetime(out[F.EMI])
        out[F.DESP] = pd.to_datetime(out[F.DESP])
        out[F.HOR] = out[F.HOR].astype(str)
        out[F.PRON] = out[F.PRON].astype(float)
        return out

    def fit(self, train: pd.DataFrame) -> "ElasticNetCorrector":
        self.spec_ = dict(self.spec or chosen_spec())
        hist = self._prep(train[self.KEYS + [F.REAL]]).dropna(subset=[F.REAL]).reset_index(drop=True)
        feat = F.build_features(hist, avail_lag=self.avail_lag)
        X = F.design_matrix(feat, self.spec_["cols"])
        self.columns_ = list(X.columns)
        self.estimator_ = make_estimator(self.spec_).fit(X, feat["y"])
        self.history_ = hist
        self.centers_ = set(hist[F.CD])
        self.history_until_ = hist[F.DESP].max()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        out = df[F.PRON].astype(float).copy()  # CD no visto en fit -> pronostico base
        pos = np.flatnonzero(df[F.CD].isin(self.centers_).to_numpy())
        if len(pos) == 0:
            return out
        q = self._prep(df.iloc[pos][self.KEYS]).reset_index(drop=True).assign(**{F.REAL: np.nan, "_pos": pos})
        atraso = (q[F.EMI] - pd.Timedelta(days=self.avail_lag) - self.history_until_).dt.days.max()
        if atraso > 0:
            log.warning("la historia del modelo llega al %s: faltan %d dias de reales para estas emisiones "
                        "(reentrenar)", self.history_until_.date(), atraso)
        comb = pd.concat([self.history_.assign(_pos=-1), q], ignore_index=True)
        feat = F.build_features(comb, avail_lag=self.avail_lag)
        fq = feat[feat["_pos"] >= 0]
        X = F.design_matrix(fq, self.spec_["cols"]).reindex(columns=self.columns_, fill_value=0.0)
        pred = F.apply_delta(fq[F.PRON], self.estimator_.predict(X))
        out.iloc[fq["_pos"].to_numpy()] = pred
        return out


class BaseOnlyCorrector:
    """Devuelve el pronostico base sin corregir. Lo usa la inferencia si el modelo no se puede cargar."""

    trained_centers_: set = set()

    def fit(self, train: pd.DataFrame) -> "BaseOnlyCorrector":
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df[F.PRON].astype(float)


class GatedCorrector:
    """Aplica el gate de historia minima: solo CDs con >= min_days usan el modelo."""

    def __init__(self, model_factory: Callable[[], Corrector] = ElasticNetCorrector, min_days: int = 45):
        self.model_factory = model_factory
        self.min_days = min_days
        self.model_: Corrector | None = None
        self.decisions_: dict = {}

    def fit(self, train: pd.DataFrame, centers: list[str] | None = None) -> "GatedCorrector":
        days = train[["centro_distribucion", "fecha_despacho"]].drop_duplicates()
        self.decisions_ = evaluate_gate(days, self.min_days, centers)
        ok = [cd for cd, d in self.decisions_.items() if d.status == GateStatus.TRAIN]
        self.model_ = self.model_factory().fit(train[train["centro_distribucion"].isin(ok)]) if ok else None
        self.trained_centers_ = set(ok)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pred = df["unidades_pronosticadas"].astype(float).copy()  # fallback = pronostico base
        if self.model_ is None:
            return pred
        mask = df["centro_distribucion"].isin(self.trained_centers_)
        if mask.any():
            pred.loc[mask] = self.model_.predict(df.loc[mask])
        return pred
