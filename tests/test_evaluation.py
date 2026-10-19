"""Tests del motor de backtest, la CV y el estudio del gate con datos sinteticos."""
import numpy as np
import pandas as pd
import pytest

from logforecast import cv, evaluation as ev, features as F, gate_study as gs
from logforecast.diagnostics import bias_stability

CDS = ["CD_A", "CD_B", "CD_C"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Un pronostico por (CD, emision, horizonte) y su real diario. Cubre 2026-01-01 a 2026-03-01."""
    rng = np.random.default_rng(0)
    dias = pd.date_range("2026-01-01", "2026-03-01")
    nivel = {"CD_A": 1000.0, "CD_B": 500.0, "CD_C": 200.0}
    sesgo = {"CD_A": 0.85, "CD_B": 1.0, "CD_C": 1.15}
    dow = np.array([1.0, 1.3, 1.0, 1.0, 1.0, 0.7, 0.6])
    real = {(cd, d): nivel[cd] * dow[d.dayofweek] * rng.lognormal(0, 0.05) for cd in CDS for d in dias}
    filas = []
    for cd in CDS:
        for e in pd.date_range("2026-01-01", "2026-02-28"):
            for h in range(1, 5):
                d = e + pd.Timedelta(days=h)
                if d > dias[-1]:
                    continue
                r = real[(cd, d)]
                filas.append({F.CD: cd, F.EMI: e, F.DESP: d, F.HOR: f"D+{h}", F.PRON: r * sesgo[cd] * rng.lognormal(0, 0.03),
                              F.REAL: r})
    return pd.DataFrame(filas)


@pytest.fixture(scope="module")
def feat(frame) -> pd.DataFrame:
    return F.build_features(frame, avail_lag=1)


def test_backtest_trains_only_on_known_rows(feat):
    """Cada origen entrena con despacho <= t - rezago y evalua solo lo emitido en t."""
    t = pd.Timestamp("2026-02-01")
    d = ev.backtest(feat, ev.SPEC_RATIO_CD, 1, [t], "cv")
    assert set(d[F.EMI]) == {t} and set(d["origen"]) == {t}
    # el factor del ratio de ese origen solo puede depender de reales <= t-1
    tr = feat[feat[F.DESP] <= t - pd.Timedelta(days=1)]
    s = tr.groupby(F.CD)[[F.REAL, F.PRON]].sum()
    esperado = (s[F.REAL] / s[F.PRON])
    g = d.groupby(F.CD).first()
    assert np.allclose(g["pred"] / g[F.PRON], esperado.loc[g.index])


def test_desp_max_blocks_future_targets(feat):
    d = ev.backtest(feat, ev.SPEC_BASE, 1, ev.ORIG_TRAIN, "cv", desp_max=ev.ORIG_TEST[0])
    assert d[F.DESP].max() <= ev.ORIG_TEST[0]


def test_train_hasta_freezes_the_model(feat):
    """Con train_hasta fijo, el factor es el mismo en todos los origenes de test."""
    corte = ev.ORIG_TEST[0] - pd.Timedelta(days=1)
    d = ev.backtest(feat, ev.SPEC_RATIO_CD, 1, ev.ORIG_TEST, "test", train_hasta=corte)
    f = (d["pred"] / d[F.PRON]).groupby(d[F.CD]).agg(["min", "max"])
    assert np.allclose(f["min"], f["max"])


def test_ratio_corrects_a_biased_forecast(feat):
    base = ev.backtest(feat, ev.SPEC_BASE, 1, ev.ORIG_TEST, "test")
    ratio = ev.backtest(feat, ev.SPEC_RATIO_CD, 1, ev.ORIG_TEST, "test", train_hasta=ev.ORIG_TEST[0])
    assert ev.wape(ratio) < ev.wape(base)


def test_ventana_limits_training_rows(feat):
    t = pd.Timestamp("2026-02-20")
    spec = dict(kind="ratio_mov", window=14)
    a = ev.backtest(feat, spec, 1, [t], "cv")
    b = ev.backtest(feat, dict(kind="ratio"), 1, [t], "cv", ventana=14)
    assert np.allclose(a["pred"].to_numpy(), b["pred"].to_numpy())


def test_min_rows_falls_back_to_base(feat):
    spec = dict(kind="ridge", cols=F.CORE + F.EXTRA, alpha=10.0, min_rows=10**9)
    d = ev.backtest(feat, spec, 1, [pd.Timestamp("2026-02-20")], "cv")
    assert np.allclose(d["pred"], d[F.PRON])


def test_run_models_and_summaries(frame):
    modelos = {"base": ev.SPEC_BASE, "ratio_cd": ev.SPEC_RATIO_CD, ev.REF_NAME: ev.SPEC_REF,
               "seasonal_naive": dict(kind="snaive")}
    corridas = ev.run_lags(frame, modelos, lags=(1, 2))
    glob, por_cd, dif = ev.summarize_runs(corridas)
    assert set(glob.index.get_level_values("zona")) == {"train", "test"}
    assert (dif[ev.REF_NAME].abs() < 1e-9).all()
    info = ev.describe_runs(corridas)
    assert list(info.index) == [1, 2]


def test_cv_study_ridge(feat):
    study = cv.run_cv_study(feat, cv.ridge_specs(alphas=(1, 100)))
    assert set(study.best) == {"ridge"}
    t = study.table()
    assert list(t.columns) == ["modelo", "config", "WAPE CV (%)"] and len(t) == 2
    assert study.block_table().shape[0] == 3  # referencia, base, ridge
    d = study.decision()
    assert d.loc["ridge", f"reemplaza a {ev.REF_NAME}"] in (True, False)
    ho, tab, dif = study.holdout()
    assert {"base", "ratio_cd", "ridge_cv"} <= set(tab.columns) and f"ridge_cv - {ev.REF_NAME}" in dif.columns and ev.REF_NAME in tab.columns
    assert study.passing() == [f for f in d.index if d.loc[f, f"reemplaza a {ev.REF_NAME}"]]
    assert "GLOBAL" in tab.index
    coef = study.coefficients()
    assert list(coef.columns) == ["ridge"] and len(coef) == len(F.CORE + F.EXTRA) + len(CDS)


def test_cv_never_sees_test_targets(feat):
    study = cv.run_cv_study(feat, cv.ridge_specs(alphas=(10,)))
    for d in study.res.values():
        assert d[F.DESP].max() <= ev.ORIG_TEST[0]
    assert study.refs["base"][F.DESP].max() <= ev.ORIG_TEST[0]


def test_cv_enet_and_lgbm_specs_and_labels():
    e = cv.enet_specs(alphas=(1e-3,), l1_ratios=(0.5, 1.0))
    assert len(e) == 2 and cv.label("enet", (0.5, 1e-3)) == "l1_ratio=0.5 alpha=0.001"
    g = cv.lgbm_specs(grid={"num_leaves": (4,), "n_estimators": (50,), "min_child_samples": (20,)})
    (k, spec), = g.items()
    assert k == ("lgbm", (4, 50, 20)) and spec["params"]["num_leaves"] == 4
    assert cv.label(*k) == "leaves=4 n=50 min_child=20"


def test_cv_study_enet(feat):
    study = cv.run_cv_study(feat, {**cv.ridge_specs(alphas=(10,)), **cv.enet_specs(alphas=(1e-2,), l1_ratios=(0.5,))})
    d = study.decision()
    assert {"ridge", "enet"} == set(d.index) and "bloques vs ridge" in d.columns
    assert study.winner() in (None, "ridge", "enet")
    assert set(study.coefficients().columns) == {"ridge", "enet"}


def test_cv_study_lgbm(feat):
    pytest.importorskip("lightgbm")
    specs = cv.lgbm_specs(grid={"num_leaves": (4,), "n_estimators": (30,), "min_child_samples": (20,)})
    study = cv.run_cv_study(feat, specs)
    imp = study.importance()
    assert abs(imp.sum() - 100) < 1e-6


def test_blocks_cover_all_origins():
    b = cv.make_blocks(ev.ORIG_TRAIN, 4)
    assert sum(len(x) for x in b) == len(ev.ORIG_TRAIN) and len(b) == 4


def test_bias_stability(frame):
    t = bias_stability(frame, "2026-02-14")
    assert t.loc["CD_A", ("bias", "1ra")] > 0 > t.loc["CD_C", ("bias", "1ra")]  # A subestima, C sobreestima


def test_gate_study_pipeline(frame):
    orders = frame.drop_duplicates([F.CD, F.DESP]).rename(columns={F.REAL: "unidades_reales"})[[F.CD, F.DESP, "unidades_reales"]]
    hist = gs.history_frame(orders)
    assert gs.gate_state(hist, "2026-01-21", CDS)["estado"].eq("FALLBACK").all()
    # los despachos sinteticos parten el 2-ene: 44 dias el 14-feb, 45 el 15-feb
    assert gs.gate_state(hist, "2026-02-14", CDS)["estado"].eq("FALLBACK").all()
    assert gs.gate_state(hist, "2026-02-15", CDS)["estado"].eq("TRAIN").all()
    assert gs.first_open_date(hist, CDS, 45, 1, "2026-01-02", "2026-03-01") == pd.Timestamp("2026-02-16")
    ec = gs.edge_cases(hist, CDS, "2026-02-28")
    assert (ec["excepción"] == "no").all()
    dec, chk = gs.cold_start_demo(frame, CDS, "CD_C", "2026-02-01", "2026-02-28")
    assert dec.loc["CD_C", "status"] == "FALLBACK" and bool(chk.loc["CD_C", "pred == base"])
    assert dec.loc["CD_A", "status"] == "TRAIN" and not bool(chk.loc["CD_A", "pred == base"])


def test_gate_study_threshold_simulation(frame):
    feat = F.build_features(frame, avail_lag=1)
    orders = frame.drop_duplicates([F.CD, F.DESP]).rename(columns={F.REAL: "unidades_reales"})[[F.CD, F.DESP, "unidades_reales"]]
    hist = gs.history_frame(orders)
    origs = pd.date_range("2026-01-15", "2026-02-28")
    D, tabla = gs.backtest_by_history(feat, hist, CDS, origs)
    assert {"pred_base", f"pred_{ev.REF_NAME}", f"pred_{cv.ELEGIDO[0]}", "tramo"} <= set(D.columns)
    assert "45+" in set(tabla.index)
    ab = gs.open_status(hist, CDS, origs, [21, 45])
    sim = gs.simulate_thresholds(D, ab, ev.REF_NAME, n_boot=20)
    assert sim.loc[21, "cobertura (%)"] > sim.loc[45, "cobertura (%)"]
    curva, n = gs.learning_curve(feat, pd.date_range("2026-02-15", "2026-02-28"), [14, 28, None])
    assert list(curva.columns) == ["base", ev.REF_NAME, cv.ELEGIDO[0]] and n > 0


def test_common_dates_keeps_only_full_horizon_dates(frame):
    """Solo quedan (CD, despacho) con los 4 horizontes y el n queda igual entre horizontes."""
    from logforecast.diagnostics import common_dates, horizon_wape_common_dates

    c = common_dates(frame)
    assert (c.groupby([F.CD, F.DESP])[F.HOR].nunique() == 4).all()
    t = horizon_wape_common_dates(frame)
    assert t["n_comunes"].nunique() == 1
    assert (t["n_comunes"] <= t["n_todas"]).all()


def test_dia_semana_is_ordinal_of_dispatch(feat):
    """dia_semana = dia de la semana del despacho (0 = lunes ... 6 = domingo) y esta en CORE."""
    assert "dia_semana" in F.CORE
    assert (feat["dia_semana"] == feat[F.DESP].dt.dayofweek).all()
    assert set(feat["dia_semana"].unique()) == set(range(7))


def test_chosen_spec_matches_elegido():
    spec = cv.chosen_spec()
    assert spec["kind"] == cv.ELEGIDO[0] == "enet"
    assert (spec["l1_ratio"], spec["alpha"]) == cv.ELEGIDO[1]
    assert spec["cols"] == cv.default_cols()


def test_elasticnet_corrector_matches_backtest(frame):
    """El corrector de produccion predice lo mismo que el motor de backtest de la CV (mismas features)."""
    from logforecast.model import ElasticNetCorrector

    t = ev.ORIG_TEST[0]
    corte = t - pd.Timedelta(days=1)
    m = ElasticNetCorrector().fit(frame[frame[F.DESP] <= corte])
    req = frame[frame[F.EMI] == t]
    p_prod = m.predict(req.drop(columns=F.REAL)).to_numpy()
    feat = F.build_features(frame, avail_lag=1)
    p_cv = ev.fit_predict(feat[feat[F.DESP] <= corte], feat[feat[F.EMI] == t], cv.chosen_spec())
    assert np.allclose(p_prod, p_cv)


def test_elasticnet_corrector_unknown_cd_and_serialization(frame, tmp_path):
    from logforecast import io
    from logforecast.model import ElasticNetCorrector, GatedCorrector

    corte = pd.Timestamp("2026-02-20")
    g = GatedCorrector(ElasticNetCorrector, min_days=45).fit(frame[frame[F.DESP] <= corte])
    req = frame[frame[F.EMI] == corte + pd.Timedelta(days=1)].drop(columns=F.REAL).copy()
    nuevo = req.iloc[:2].assign(**{F.CD: "CD_NUEVO"})
    req = pd.concat([req, nuevo], ignore_index=True)
    p = g.predict(req)
    assert np.allclose(p[req[F.CD] == "CD_NUEVO"], req.loc[req[F.CD] == "CD_NUEVO", F.PRON])  # fallback
    assert not np.allclose(p[req[F.CD] == "CD_A"], req.loc[req[F.CD] == "CD_A", F.PRON])      # corregido
    uri = str(tmp_path / "model.joblib")
    io.save_model(g, uri)
    assert np.allclose(io.load_model(uri).predict(req), p)
