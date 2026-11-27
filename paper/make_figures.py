#!/usr/bin/env python3
"""Figures of the two-layer paper (numpy + matplotlib only).
    python3 make_figures.py          -> figures/*.pdf (and .png previews)
Data: data/*.json, data/*.csv (see compute_paper_data.py and the README of two_layer)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stability"))
import two_layer_modes as tl          # noqa: E402
from pencil import matrix_pencil, dominant  # noqa: E402

D = HERE / "data"
F = HERE / "figures"
F.mkdir(exist_ok=True)

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LS = ["-", "--", "-.", (0, (5, 1.5, 1, 1.5, 1, 1.5)), ":"]
MK = ["o", "s", "^", "D", "v"]
GREY, DARK = "#8a8a8a", "#333333"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 3.5, "savefig.dpi": 200,
    "mathtext.fontset": "cm", "font.family": "serif"})
W1, W2 = 3.37, 7.0          # REVTeX column and page widths (in)


def label(ax, s):
    ax.set_title(s, loc="left", fontsize=9, pad=4)


def save(fig, name):
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
def fig_sketch():
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    L = 2.0
    ax.add_patch(Rectangle((0, 0), L, 0.5, fc="#dbe8f7", ec="none"))
    ax.add_patch(Rectangle((0, 0.5), L, 0.5, fc="#fbe3d6", ec="none"))
    x = np.linspace(0, L, 200)
    ax.plot(x, 0.5 + 0.05 * np.sin(2 * np.pi * x / L * 1.5), color=DARK, lw=1.0)
    ax.plot([0, L], [0.5, 0.5], color=DARK, lw=0.6, ls="--")
    for yy in (0, 1):
        ax.plot([0, L], [yy, yy], color="k", lw=2.5)
    ax.text(L / 2, 0.25, "liquid 2 (denser)\n" r"$\varepsilon_r,\ K_r,\ \mu_r$", ha="center", va="center", fontsize=8)
    ax.text(L / 2, 0.78, "liquid 1 (reference)\n" r"$\varepsilon=K=\mu=1$", ha="center", va="center", fontsize=8)
    ax.text(L + 0.05, 0.0, r"injector: $V=1,\ q=1$", va="center", fontsize=8)
    ax.text(L + 0.05, 1.0, r"collector: $V=0$", va="center", fontsize=8)
    ax.text(L + 0.05, 0.5, r"$y=\frac{1}{2}+\eta(x,t)$", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((-0.18, 0.75), (-0.18, 0.35), arrowstyle="-|>", mutation_scale=9, color=DARK))
    ax.text(-0.24, 0.55, r"$\mathbf{g}$", ha="right", va="center", fontsize=9)
    ax.set_xlim(-0.35, L + 1.25); ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    save(fig, "fig_sketch")


# -----------------------------------------------------------------------------
def fig_base():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), constrained_layout=True)
    cases = [(1., 1.), (2., 0.5), (0.5, 10.)]
    for i, (er, Kr) in enumerate(cases):
        yl = np.linspace(0, .5, 400); yu = yl + .5
        b = tl.base(10., er, Kr, yl, yu)
        lab = rf"$\varepsilon_r={er:g},\ K_r={Kr:g}$"
        for j, key in enumerate(("E", "q")):
            ax[j].plot(np.r_[yl, np.nan, yu], np.r_[b["L"][key], np.nan, b["U"][key]],
                       color=COL[i], ls=LS[i], label=lab)
    for a_ in ax:
        a_.axvline(0.5, color=GREY, lw=0.6, ls=":")
        a_.set_xlabel(r"$y$")
    ax[0].set_ylabel(r"$\bar E$"); ax[1].set_ylabel(r"$\bar q$")
    ax[1].set_yscale("log")
    ax[0].legend(frameon=False, loc="upper left")
    label(ax[0], "(a)"); label(ax[1], "(b)")
    save(fig, "fig_base")


# -----------------------------------------------------------------------------
def fig_neutral():
    d = json.load(open(D / "paper_data.json"))["neutral"]
    fig, ax = plt.subplots(figsize=(W1, 2.7), constrained_layout=True)
    tt = np.array(d["twotier"])
    ax.plot(tt[:, 0], tt[:, 1], color=DARK, ls="--", lw=1.1, label="stationary two-tier mode")
    for i, Gr in enumerate(("0.0", "10.0", "100.0", "1000.0", "3000.0")):
        a = np.array(d[Gr])
        osc = a[:, 2] > 1e-6 if Gr != "0.0" else np.ones(len(a), bool)
        a = a[osc]
        lab = r"$Gr=0$ (homogeneous)" if Gr == "0.0" else rf"$Gr={float(Gr):g}$"
        ax.plot(a[:, 0], a[:, 1], color=COL[i], ls=LS[i], marker=MK[i], markevery=3, label=lab)
    ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$T$")
    ax.set_xlim(1.4, 16.2); ax.set_ylim(150, 1000)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    save(fig, "fig_neutral")


# -----------------------------------------------------------------------------
def fig_Tc_Gr():
    d = json.load(open(D / "Tc_Gr.json"))
    M = 10.
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), constrained_layout=True)
    names = (("full", "full model"), ("stokes", "no inertia (Stokes)"), ("no_lag", "no charge lag"))
    for i, (key, lab) in enumerate(names):
        a = np.array(d["curves"][key]); a = a[a[:, 0] <= 3000]
        Gr, Tc, kc, om = a.T
        kw = dict(color=COL[i], ls=LS[i], marker=MK[i], label=lab)
        ax[0].plot(Gr, Tc, **kw)
        ax[1].plot(Gr, om * Tc / M**2, **kw)
    ax[0].axhline(d["Ts"], color=DARK, lw=1.0, ls="--", label="two-tier mode")
    ax[0].axhline(164.0614, color=GREY, lw=0.8, ls=":", label="homogeneous liquid")
    ax[0].axvline(d["Gstar"], color=GREY, lw=0.6)
    ax[0].annotate(r"$Gr^*$", (d["Gstar"], 600), xytext=(-4, 0), textcoords="offset points",
                   ha="right", va="center", fontsize=8, color=DARK)
    ax[0].set_xscale("log"); ax[0].set_ylim(150, 620)
    ax[0].set_xlabel(r"$Gr$"); ax[0].set_ylabel(r"$T_c$")
    ax[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.45))
    g = np.array([0.1, 30.])
    ax[1].plot(g, 1.45 * 0.3172 * 1.6436 * np.sqrt(g), color=GREY, lw=0.9, ls=":", label=r"$\propto Gr^{1/2}$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$Gr$"); ax[1].set_ylabel(r"$\omega_c\,T_c/M^2$")
    ax[1].legend(frameon=False, loc="upper left")
    label(ax[0], "(a)"); label(ax[1], "(b)")
    save(fig, "fig_Tc_Gr")


# -----------------------------------------------------------------------------
def fig_eigen():
    d = json.load(open(D / "paper_data.json"))["eig"]
    fig = plt.figure(figsize=(W2, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.25])
    ax0 = fig.add_subplot(gs[0])
    cases = (("homog", r"$Gr=0$"), ("hopf100", r"$Gr=100$"), ("hopf1000", r"$Gr=1000$"),
             ("twotier", "two-tier"))
    for i, (key, lab) in enumerate(cases):
        e = d[key]; y = np.array(e["y"]); v = np.array(e["v_re"]) + 1j * np.array(e["v_im"])
        ax0.plot(np.abs(v), y, color=COL[i] if key != "twotier" else DARK,
                 ls=LS[i] if key != "twotier" else "--", label=lab)
    ax0.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax0.set_xlabel(r"$|\hat v|/\max|\hat v|$"); ax0.set_ylabel(r"$y$")
    ax0.set_ylim(0, 1)
    ax0.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, fontsize=7,
               columnspacing=0.8, handlelength=1.8)
    label(ax0, "(a)")
    for j, (key, ttl) in enumerate((("hopf100", "(b)"), ("twotier", "(c)"))):
        ax = fig.add_subplot(gs[j + 1])
        e = d[key]; k = e["k"]
        y = np.array(e["y"]); v = np.array(e["v_re"]) + 1j * np.array(e["v_im"])
        o = np.argsort(y); y, v = y[o], v[o]
        yy = np.linspace(0, 1, 200)
        vi = np.interp(yy, y, v.real) + 1j * np.interp(yy, y, v.imag)
        x = np.linspace(0, 2 * np.pi / k, 160)
        psi = np.real(1j * vi[:, None] / k * np.exp(1j * k * x[None, :]))
        lev = np.linspace(-np.abs(psi).max(), np.abs(psi).max(), 14)
        ax.contour(x, yy, psi, levels=lev[lev > 0], colors=COL[0], linewidths=0.8)
        ax.contour(x, yy, psi, levels=lev[lev < 0], colors=COL[1], linewidths=0.8, linestyles="--")
        ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
        ax.set_aspect("equal"); ax.set_xlabel(r"$x$")
        ax.set_yticks([0, 0.5, 1]); ax.set_ylabel(r"$y$")
        label(ax, ttl)
    save(fig, "fig_eigen")


# -----------------------------------------------------------------------------
def fig_delta():
    d = json.load(open(D / "delta_scan.json"))
    fig, ax = plt.subplots(figsize=(W1, 2.5), constrained_layout=True)
    for i, Gr in enumerate(("10", "100", "1000")):
        Ts = d[Gr]["sharp"][0]; a = np.array(d[Gr]["rows"])
        ax.loglog(a[:, 0], np.abs(a[:, 1] / Ts - 1), color=COL[i], ls=LS[i], marker=MK[i],
                  label=rf"$Gr={Gr}$")
    dd = np.array([0.005, 0.08])
    ax.loglog(dd, 2.0 * dd**2, color=GREY, ls=":", lw=0.9, label=r"$\propto\delta^2$")
    ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$|T_n(\delta)/T_n(0)-1|$")
    ax.set_xticks([0.005, 0.01, 0.02, 0.04, 0.08])
    ax.set_xticklabels(["0.005", "0.01", "0.02", "0.04", "0.08"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_delta")


# -----------------------------------------------------------------------------
def fig_firedrake():
    def load(name):
        r = list(csv.DictReader(open(D / name)))
        t = np.array([float(x["t"]) for x in r]); a = np.array([float(x["vproj_c"]) for x in r])
        return t, a
    t1, a1 = load("fwd_32x40_T192_E1e-12.csv")
    t2, a2 = load("fwd_32x40_T192_E1e-8.csv")
    sel = t1 >= 15
    lam, amp, _ = matrix_pencil(a1[sel], t1[1] - t1[0], order=6)
    lam, amp = dominant(lam, amp, t1[sel][-1] - t1[sel][0])
    sig = lam[0].real
    fig, ax = plt.subplots(figsize=(W1, 2.4), constrained_layout=True)
    ax.plot(t1, a1 / 1e-6, color=COL[0], lw=1.0, label=r"$E_0=10^{-12}$")
    ax.plot(t2, a2 / 1e-4, color=COL[1], lw=1.0, ls="--", label=r"$E_0=10^{-8}$ (scaled by $10^{-2}$)")
    env = 2 * np.abs(amp[0]) * np.exp(sig * (t1 - t1[sel][0])) / 1e-6
    ax.plot(t1[sel], env[sel], color=GREY, lw=0.8, ls=":", label=r"$\pm e^{\sigma t}$, fitted")
    ax.plot(t1[sel], -env[sel], color=GREY, lw=0.8, ls=":")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$a_c\ (\times 10^{6})$")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    save(fig, "fig_firedrake")


if __name__ == "__main__":
    fig_sketch(); fig_base(); fig_neutral(); fig_Tc_Gr(); fig_eigen(); fig_delta(); fig_firedrake()
    print("figures written to", F)
