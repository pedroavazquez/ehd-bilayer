#!/usr/bin/env python3
"""Comparison of the Basilisk run (T=175, Ras=2000, Sc=1000) with Traore et al. (2024).
Inputs: digitized Fig. 6a and Fig. 10 (Sc=1000) of the paper, the Basilisk logs, and the
diffuse-interface linear scan (ripple_scan.txt: delta Gr sigma omega, viscous units)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
n = np.arange(1, 4001); a = -2*np.sin(n*np.pi/2)/(n*np.pi)
def lacey(t, Sc): return np.sqrt(0.5*np.sum(a[:, None]**2*np.exp(-2*(n[:, None]*np.pi)**2*t/Sc), 0))

def load(f):
    d = np.loadtxt(f); return d[np.argsort(d[:, 0])]
d1 = load(here + "/../examples/demo_T175/log_T175"); d2 = load(here + "/log_cont")
d = np.vstack([d1[d1[:, 0] < d2[0, 0], :10], d2[:, :10]])

f6 = np.loadtxt(here + "/fig6a_digit.txt")
f6 = f6[(f6[:, 0] < 88) | ((f6[:, 0] > 116) & (f6[:, 1] < 0.05))]   # drop points on the legend and fitted lines
f10 = np.loadtxt(here + "/fig10_sc1000_digit.txt")
f10 = f10[(f10[:, 0] > 50) & (f10[:, 1] > 0)]
rs = np.loadtxt(here + "/ripple_scan.txt")

c = dict(paper="#444444", bas="#1f6fb4", d1="#c0392b", d2="#e08e0b")
fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
t = np.linspace(0.01, 140, 300)

A = ax[0, 0]
A.plot(f6[:, 0], f6[:, 1], "o", ms=3, color=c["paper"], label="paper, Fig. 6a")
A.plot(t, lacey(t, 1000.), "-", color=c["d1"], lw=1.5, label="diffusion only, Sc = 1000")
A.plot(t, lacey(t, 2000.), "--", color=c["d2"], lw=1.5, label="diffusion only, Sc = 2000")
A.plot(d[:, 0], d[:, 2], "-", color=c["bas"], lw=2, label="Basilisk, seed 1e-4")
A.set_xlabel("t"); A.set_ylabel("Lacey index"); A.set_xlim(0, 140); A.set_ylim(0, 0.62)
A.legend(frameon=False, fontsize=8, loc="upper right")

B = ax[0, 1]
tb = d[:, 0]; vb = d[:, 3]
t3 = tb[np.argmax(vb > 3)]; tp3 = f10[np.argmax(f10[:, 1] > 3), 0]; sh = tp3 - t3
B.semilogy(f10[:, 0], f10[:, 1], "-", color=c["paper"], lw=1.2, label="paper, Fig. 10, Sc = 1000")
B.semilogy(tb + sh, vb, "-", color=c["bas"], lw=1.5, label="Basilisk, shifted by %.1f" % sh)
B.set_xlabel("t"); B.set_ylabel("Vmax"); B.set_xlim(40, 140); B.set_ylim(5e-3, 20)
B.legend(frameon=False, fontsize=8, loc="lower right")

C = ax[1, 0]
for dl, ls in zip([0.3, 0.4, 0.5], ["-", "--", ":"]):
    m = rs[:, 0] == dl
    C.plot(rs[m, 1], rs[m, 3]/rs[m, 2], ls, color=c["bas"], label=r"$\delta$ = %.1f" % dl)
C.axhline(10.1, color=c["paper"], lw=1); C.text(6, 10.4, "paper, Fig. 10", fontsize=8)
C.set_xlabel("Gr"); C.set_ylabel(r"$\omega/\sigma$"); C.set_ylim(0, 14)
C.legend(frameon=False, fontsize=8, loc="lower right")

D = ax[1, 1]
for dl, ls in zip([0.3, 0.4, 0.5], ["-", "--", ":"]):
    m = rs[:, 0] == dl
    D.plot(rs[m, 1], np.pi/rs[m, 3], ls, color=c["bas"], label=r"$\delta$ = %.1f" % dl)
D.axhline(3.59, color=c["paper"], lw=1); D.text(26, 3.75, "paper, Fig. 10", fontsize=8)
D.set_xlabel("Gr"); D.set_ylabel(r"ripple spacing $\pi/\omega$"); D.set_ylim(0, 8)
D.legend(frameon=False, fontsize=8, loc="upper right")

for A_, l in zip(ax.flat, "abcd"):
    A_.set_title("(%s)" % l, loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(here + "/compare_paper.png", dpi=150)
print("shift", sh, "Basilisk final Vmax", vb[-1], "Lacey", d[-1, 2])
