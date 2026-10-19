"""Graficos reutilizables desde el notebook. Cada funcion dibuja y muestra la figura."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HORIZONTES = ["D+1", "D+2", "D+3", "D+4"]
COL_HORIZONTE = dict(zip(HORIZONTES, ["#E45756", "#F58518", "#4C78A8", "#54A24B"]))
CANALES = ["PORTAL_WEB", "AGENTE_COMERCIAL", "ATENCION_TELEFONICA"]
COL_CANAL = dict(zip(CANALES, ["#4C78A8", "#F58518", "#54A24B"]))


def _weekly_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))


def plot_forecast_vs_real(frame: pd.DataFrame) -> None:
    """Real vs pronostico (D+1..D+4) por CD."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for ax, cd in zip(axes.ravel(), sorted(frame["centro_distribucion"].unique())):
        g = frame[frame["centro_distribucion"] == cd]
        real = g.drop_duplicates("fecha_despacho").sort_values("fecha_despacho")
        ax.plot(real["fecha_despacho"], real["real"], color="black", lw=2, label="Real")
        for h in HORIZONTES:
            gh = g[g["horizonte"] == h].sort_values("fecha_despacho")
            ax.plot(gh["fecha_despacho"], gh["unidades_pronosticadas"],
                    color=COL_HORIZONTE[h], lw=1, alpha=0.8, label=h)
        ax.set_title(cd)
        _weekly_axis(ax)
    axes[0, 0].legend(ncol=5, fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_bias_rolling(frame: pd.DataFrame, window: int = 7) -> None:
    """Bias movil por CD, un panel por horizonte (misma escala vertical)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, h in zip(axes.ravel(), HORIZONTES):
        d = frame[frame["horizonte"] == h].sort_values("fecha_despacho").copy()
        d["err"] = d["real"] - d["unidades_pronosticadas"]
        for cd, g in d.groupby("centro_distribucion"):
            bias = g["err"].rolling(window).sum() / g["real"].rolling(window).sum()
            ax.plot(g["fecha_despacho"], bias * 100, label=cd)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Horizonte {h}")
        _weekly_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Bias móvil {window} días (%)")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_real_by_channel(orders: pd.DataFrame, fc: pd.DataFrame, horizonte: str = "D+1") -> None:
    """Demanda real por canal y total por CD, con el pronostico (nivel CD) punteado."""
    por_canal = (orders.groupby(["centro_distribucion", "fecha_despacho", "canal_venta"])
                 ["unidades_reales"].sum().reset_index())
    total = (orders.groupby(["centro_distribucion", "fecha_despacho"])
             ["unidades_reales"].sum().reset_index())
    fch = fc[fc["horizonte"] == horizonte]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for ax, cd in zip(axes.ravel(), sorted(orders["centro_distribucion"].unique())):
        for c in CANALES:
            g = por_canal[(por_canal["centro_distribucion"] == cd) & (por_canal["canal_venta"] == c)]
            ax.plot(g["fecha_despacho"], g["unidades_reales"], color=COL_CANAL[c], lw=1.2, label=c)
        t = total[total["centro_distribucion"] == cd]
        ax.plot(t["fecha_despacho"], t["unidades_reales"], color="black", lw=2, label="Total real")
        f = fch[fch["centro_distribucion"] == cd].sort_values("fecha_despacho")
        ax.plot(f["fecha_despacho"], f["unidades_pronosticadas"], color="#E45756",
                lw=1, ls="--", label=f"Pronóstico {horizonte} (total CD)")
        ax.set_title(cd)
        _weekly_axis(ax)
    axes[0, 0].legend(fontsize=7)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def _weekly(df: pd.DataFrame, date_col: str, value_col: str, keys: list[str]) -> pd.DataFrame:
    d = df.assign(semana=df[date_col].dt.to_period("W").dt.start_time)
    return d.groupby([*keys, "semana"])[value_col].sum().reset_index()


def plot_plan_by_channel(plan: pd.DataFrame) -> None:
    """Plan vs real (unidades por semana) por CD, una linea por canal.

    Linea punteada = plan (semana de fecha_plan); linea continua = real (semana de fecha_compra_real).
    """
    p = _weekly(plan, "fecha_plan", "unidades_plan", ["centro_distribucion", "canal_venta"])
    r = _weekly(plan, "fecha_compra_real", "unidades_reales", ["centro_distribucion", "canal_venta"])
    # se excluye la primera semana si es parcial (empieza a mitad de semana) y las semanas
    # posteriores al ultimo plan, para no mostrar caidas artificiales
    first = plan["fecha_plan"].min()
    first_full = first.to_period("W").start_time + (pd.Timedelta(days=7) if first.dayofweek else pd.Timedelta(0))
    last = plan["fecha_plan"].max().to_period("W").start_time
    p = p[(p["semana"] >= first_full) & (p["semana"] <= last)]
    r = r[(r["semana"] >= first_full) & (r["semana"] <= last)]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for ax, cd in zip(axes.ravel(), sorted(plan["centro_distribucion"].unique())):
        for c in CANALES:
            m = (p["centro_distribucion"] == cd) & (p["canal_venta"] == c)
            n = (r["centro_distribucion"] == cd) & (r["canal_venta"] == c)
            ax.plot(p.loc[m, "semana"], p.loc[m, "unidades_plan"], color=COL_CANAL[c], ls="--", lw=1.2)
            ax.plot(r.loc[n, "semana"], r.loc[n, "unidades_reales"], color=COL_CANAL[c], lw=1.8, label=c)
        ax.set_title(cd)
        _weekly_axis(ax)
    axes[0, 0].legend(fontsize=7, title="continuo = real, punteado = plan", title_fontsize=7)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_plan_by_client(plan: pd.DataFrame, cd: str) -> None:
    """Plan vs real por cliente de un CD. Puntos rojos = compras fuera de plan."""
    d = plan[plan["centro_distribucion"] == cd]
    clientes = sorted(d["cliente_id"].unique())
    ncols = 3
    nrows = -(-len(clientes) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows), sharex=True)
    axes = axes.ravel()
    for ax, cli in zip(axes, clientes):
        g = d[d["cliente_id"] == cli]
        canal = g["canal_venta"].iloc[0]
        ax.plot(g["fecha_plan"], g["unidades_plan"], color="gray", ls="--", marker="o", ms=3, label="Plan")
        gr = g.sort_values("fecha_compra_real")
        ax.plot(gr["fecha_compra_real"], gr["unidades_reales"], color=COL_CANAL[canal], marker="o", ms=3, label="Real")
        fuera = gr[gr["fuera_de_plan"] == 1]
        ax.scatter(fuera["fecha_compra_real"], fuera["unidades_reales"], color="#E45756", s=28, zorder=3,
                   label="Fuera de plan")
        ax.set_title(f"{cli} · {canal}", fontsize=9)
        _weekly_axis(ax)
    for ax in axes[len(clientes):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle(cd, y=1.0)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()



# ----------------------------------------------------------------------------- Tarea 3: modelos y CV
def _cds_and_global(tabla: pd.DataFrame) -> list[str]:
    return [i for i in tabla.index if i != "GLOBAL"] + ["GLOBAL"]


def plot_wape_by_lag(corridas: dict, modelos: list[str], orig_test_start, principales=("base", "ratio_cd"),
                     zonas=("train", "test")) -> None:
    """Por rezago: WAPE por CD en train y en test, y WAPE global por origen (linea punteada = inicio de test)."""
    from logforecast.metrics import wape

    lags = list(corridas)
    fig, axs = plt.subplots(len(lags), len(zonas) + 1, figsize=(7.3 * (len(zonas) + 1), 4.5 * len(lags)),
                            squeeze=False)
    for i, L in enumerate(lags):
        _, res_l, tablas_l = corridas[L]
        for j, z in enumerate(zonas):
            t = tablas_l[z]
            (t.loc[_cds_and_global(t), modelos] * 100).plot.bar(ax=axs[i, j], width=0.9, legend=(i == 0 and j == 0))
            axs[i, j].set_title(f"AVAIL_LAG={L} · {z}: WAPE por CD")
            axs[i, j].set_ylabel("WAPE (%)")
            axs[i, j].tick_params(axis="x", rotation=0)
        ax = axs[i, len(zonas)]
        for n in principales:
            w = pd.Series({o: wape(g["real"], g["pred"]) for o, g in res_l[n].groupby("origen")}) * 100
            ax.plot(w.index, w.values, marker="o", ms=3, label=n)
        ax.axvline(orig_test_start, color="k", ls=":")
        ax.set_title(f"AVAIL_LAG={L}: WAPE global por origen")
        ax.tick_params(axis="x", rotation=45)
        if i == 0:
            axs[0, 0].legend(fontsize=7, ncol=2)
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_real_vs_pred(res: dict, cds: list[str], curvas: list[str], h_dias: int = 1, zona: str = "test") -> None:
    """Real vs base vs modelos (horizonte D+h_dias), un panel por CD. `res`: {modelo: backtest}."""
    def _sel(d, cd):
        if "zona" in d:
            d = d[d["zona"] == zona]
        return d[(d["centro_distribucion"] == cd) & (d["h_dias"] == h_dias)].sort_values("fecha_despacho")

    fig, axs = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
    for a, cd in zip(axs.ravel(), cds):
        g0 = _sel(res["base"], cd)
        a.plot(g0["fecha_despacho"], g0["real"], "k-o", ms=3, label="real")
        a.plot(g0["fecha_despacho"], g0["unidades_pronosticadas"], "r--", label="base")
        for n in curvas:
            g = _sel(res[n], cd)
            a.plot(g["fecha_despacho"], g["pred"], label=n)
        a.set_title(cd)
        a.tick_params(axis="x", rotation=45)
    axs[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cv_overview(study, tab_ho: pd.DataFrame) -> None:
    """WAPE de CV por configuracion (un panel por familia, con la referencia en linea punteada) y test final por CD."""
    from logforecast.cv import label
    from logforecast.metrics import wape

    fams = list(study.best)
    fig, axs = plt.subplots(1, len(fams) + 1, figsize=(6.5 * (len(fams) + 1), 4.8))
    ref = wape(study.refs["ref"]["real"], study.refs["ref"]["pred"]) * 100
    for ax, f in zip(axs, fams):
        ks = [k for k in study.scores if k[0] == f]
        if f == "ridge":
            ax.plot([k[1][0] for k in ks], [study.scores[k] * 100 for k in ks], marker="o")
            ax.set_xscale("log")
            ax.set_xlabel("alpha")
            ax.axhline(ref, color="k", ls="--", label=study.ref_name)
            ax.set_ylabel("WAPE CV (%)")
            ax.legend(fontsize=8)
        elif f == "enet":
            for l1 in sorted({k[1][0] for k in ks}):
                kk = sorted((k for k in ks if k[1][0] == l1), key=lambda k: k[1][1])
                ax.plot([k[1][1] for k in kk], [study.scores[k] * 100 for k in kk], marker="o", label=f"l1_ratio={l1:g}")
            ax.set_xscale("log")
            ax.set_xlabel("alpha")
            ax.axhline(ref, color="k", ls="--", label=study.ref_name)
            ax.set_ylabel("WAPE CV (%)")
            ax.legend(fontsize=8)
        else:  # lgbm y otras: barras horizontales por configuracion
            ks = sorted(ks, key=study.scores.get, reverse=True)
            ax.barh([label(*k) for k in ks], [study.scores[k] * 100 for k in ks], color="tab:blue")
            ax.axvline(ref, color="k", ls="--")
            ax.set_xlim(left=min(study.scores[k] for k in ks) * 100 - 1)
            ax.set_xlabel("WAPE CV (%)")
            ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"{f} (CV)")
    (tab_ho.loc[_cds_and_global(tab_ho)] * 100).plot.bar(ax=axs[-1], width=0.85)
    axs[-1].set_ylabel("WAPE (%)")
    axs[-1].set_title("Test final por CD")
    axs[-1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()


def plot_coefficients(coef: pd.DataFrame, cols: list[str]) -> None:
    """Coeficientes estandarizados: features (izq.) y dummies de CD (der.), una barra por modelo."""
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"][: coef.shape[1]]
    feats, dummies = coef.loc[cols], coef.drop(index=cols)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    feats.iloc[::-1].plot.barh(ax=axs[0], width=0.8, color=colores)
    axs[0].set_title("Features (coeficiente estandarizado)")
    dummies.iloc[::-1].plot.barh(ax=axs[1], width=0.8, color=colores)
    axs[1].set_title("Dummies de CD (efecto fijo por centro)")
    for ax in axs:
        ax.axvline(0, color="k", lw=0.8)
    plt.tight_layout()
    plt.show()


def plot_importance(imp: pd.Series, titulo: str = "Importancia por ganancia (%)") -> None:
    imp.plot.barh(figsize=(7, 5), color="tab:blue", width=0.7, title=titulo)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------- Tarea 4: gate
def plot_learning_curve(curva: pd.DataFrame) -> None:
    """WAPE segun dias de historia de entrenamiento; la linea punteada es el base (fallback)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = [str(i) for i in curva.index]
    for c, st in zip([c for c in curva.columns if c != "base"], ["o-", "s-", "^-", "d-"]):
        ax.plot(xs, curva[c] * 100, st, label=c)
    ax.axhline(curva["base"].iloc[0] * 100, color="k", ls="--", label="base (fallback)")
    ax.set_xlabel("días de historia usados para entrenar")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Curva de aprendizaje")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_tramos(tabla: pd.DataFrame) -> None:
    """WAPE (%) por tramo de dias de historia continua al emitir."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = [c for c in tabla.columns if c not in ("orígenes", "filas") and not c.endswith("(pp)")]
    tabla[cols].plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("WAPE (%)")
    ax.set_xlabel("días de historia continua al emitir")
    ax.set_title("WAPE por tramo de historia (ventana expansiva)")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()


def plot_threshold_sim(sims: dict[str, pd.DataFrame], base_wape: float) -> None:
    """Cobertura (barras) y WAPE con gate (linea) por umbral, un panel por modelo. `sims`: {nombre: tabla}."""
    fig, axs = plt.subplots(1, len(sims), figsize=(7 * len(sims), 4.5), squeeze=False)
    for ax, (nom, t) in zip(axs[0], sims.items()):
        xs = t.index.astype(str)
        ax.bar(xs, t["cobertura (%)"], color="tab:gray", alpha=0.35, label="cobertura (%)")
        ax.set_ylabel("cobertura del modelo (%)")
        ax.set_xlabel("umbral T (días)")
        ax2 = ax.twinx()
        ax2.plot(xs, t["WAPE gate (%)"], "o-", color="tab:blue", label="WAPE con gate")
        ax2.axhline(base_wape * 100, color="k", ls="--", label="base")
        ax2.set_ylabel("WAPE (%)")
        ax.set_title(f"Gate con {nom}: cobertura y WAPE por umbral")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_capacity(d: pd.DataFrame) -> None:
    """Uso de capacidad: % de pronosticos sobre el tope por CD (izq.) y pronostico/cap vs real/cap por fila (der.)."""
    from logforecast import features as F
    from logforecast.capacity import CAP

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    ex = d.groupby(F.CD).apply(lambda g: (g[F.PRON] > g[CAP]).mean() * 100, include_groups=False)
    ex.plot.bar(ax=axs[0], color="tab:red", alpha=0.75)
    axs[0].set_ylabel("% de pronósticos sobre la capacidad")
    axs[0].set_title("Pronósticos que exceden la capacidad (todos los horizontes)")
    axs[0].tick_params(axis="x", rotation=30)
    for cd, g in d.groupby(F.CD):
        axs[1].scatter(g["uso_pron"], g["uso_real"], s=10, alpha=0.5, label=cd)
    lim = max(d["uso_pron"].max(), d["uso_real"].max()) * 1.03
    axs[1].plot([0, lim], [0, lim], "k:", lw=1)
    axs[1].axvline(1, color="tab:red", ls="--"); axs[1].axhline(1, color="tab:red", ls="--")
    axs[1].set_xlabel("pronóstico / capacidad"); axs[1].set_ylabel("real / capacidad")
    axs[1].set_title("Cada punto: un pronóstico (la línea roja es el tope)")
    axs[1].legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()


def plot_plan_deviation_by_client(plan: pd.DataFrame) -> None:
    """Desvio real vs plan (%) por cliente, un panel por CD. Barras = media +- desviacion; puntos rojos = fuera de plan."""
    import numpy as np

    rng = np.random.default_rng(0)
    d = plan.assign(dev_pct=(plan["unidades_reales"] - plan["unidades_plan"]) / plan["unidades_plan"] * 100)
    cli = (d.groupby(["centro_distribucion", "canal_venta", "cliente_id"])["dev_pct"]
             .agg(media="mean", std="std").reset_index())

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharey=True)
    for ax, cd in zip(axes.ravel(), sorted(cli["centro_distribucion"].unique())):
        g = cli[cli["centro_distribucion"] == cd].copy()
        g["canal_venta"] = pd.Categorical(g["canal_venta"], CANALES)
        g = g.sort_values(["canal_venta", "media"]).reset_index(drop=True)
        ax.bar(g.index, g["media"], yerr=g["std"], capsize=2,
               color=g["canal_venta"].map(COL_CANAL), error_kw={"lw": 0.8, "ecolor": "#444"})
        pos = dict(zip(g["cliente_id"], g.index))
        fuera = d[(d["centro_distribucion"] == cd) & (d["fuera_de_plan"] == 1)]
        ax.scatter(fuera["cliente_id"].map(pos) + rng.uniform(-0.25, 0.25, len(fuera)),
                   fuera["dev_pct"], color="#E45756", s=12, zorder=3, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(g.index)
        ax.set_xticklabels(g["cliente_id"].str[-3:], fontsize=7, rotation=90)
        ax.set_title(cd)
    for ax in axes[:, 0]:
        ax.set_ylabel("Real vs plan (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL_CANAL[c]) for c in CANALES]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#E45756", ms=4))
    axes[0, 0].legend(handles, CANALES + ["Compra fuera de plan"], fontsize=7)
    plt.tight_layout()
    plt.show()


def plot_plan_deviation_by_cd_channel(plan: pd.DataFrame) -> None:
    """Desvio de volumen (% real vs plan) y de fecha (dias) por CD y canal; puntos rojos = fuera de plan."""
    import numpy as np

    rng = np.random.default_rng(0)
    d = plan.assign(
        dev_pct=(plan["unidades_reales"] - plan["unidades_plan"]) / plan["unidades_plan"] * 100,
        dias=(plan["fecha_compra_real"] - plan["fecha_plan"]).dt.days,
    )
    res = d.groupby(["centro_distribucion", "canal_venta"]).agg(
        pct=("dev_pct", "mean"), pct_std=("dev_pct", "std"),
        dias=("dias", "mean"), dias_std=("dias", "std"),
    ).reset_index()
    cds = sorted(res["centro_distribucion"].unique())
    x = np.arange(len(cds))
    w = 0.26
    fuera_all = d[d["fuera_de_plan"] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    for ax, (m, s, col, titulo) in zip(axes, [("pct", "pct_std", "dev_pct", "Volumen: real vs plan (%)"),
                                               ("dias", "dias_std", "dias", "Fecha: días de desvío")]):
        # rango acotado: barras +- error y percentil 5-95 de los puntos fuera de plan
        lo = min((res[m] - res[s]).min(), fuera_all[col].quantile(0.05))
        hi = max((res[m] + res[s]).max(), fuera_all[col].quantile(0.95))
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        for i, c in enumerate(CANALES):
            g = res[res["canal_venta"] == c].set_index("centro_distribucion").reindex(cds)
            ax.bar(x + (i - 1) * w, g[m], w, yerr=g[s], capsize=3, color=COL_CANAL[c], label=c,
                   error_kw={"lw": 0.8, "ecolor": "#444"})
            f = fuera_all[fuera_all["canal_venta"] == c]
            xf = f["centro_distribucion"].map(dict(zip(cds, x))) + (i - 1) * w + rng.uniform(-w / 3, w / 3, len(f))
            ax.scatter(xf, f[col].clip(lo, hi), color="#E45756", s=8, zorder=3, edgecolor="white", linewidth=0.2)
        ax.set_ylim(lo, hi)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cds, fontsize=8)
        ax.set_title(titulo)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#E45756", ms=4))
    axes[0].legend(handles, labels + ["Compra fuera de plan"], fontsize=7)
    plt.tight_layout()
    plt.show()
