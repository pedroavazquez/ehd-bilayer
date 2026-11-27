#!/usr/bin/env python3
"""Lacey index and maximum velocity versus time from one or more bilayer logs.
    python3 plot_log.py run1/log [run2/log ...] -o history.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("-o", "--output", default="history.png")
a = ap.parse_args()

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
styles = ["-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5))]
for i, f in enumerate(a.logs):
    d = np.loadtxt(f)
    kw = dict(color=colors[i % 5], ls=styles[i % 5], lw=1.4, label=f)
    ax[0].plot(d[:, 0], d[:, 2], **kw)
    ax[1].semilogy(d[:, 0], d[:, 3], **kw)
ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"Lacey index $\sigma$")
ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$V_{max}$")
ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
if len(a.logs) > 1:
    ax[0].legend(frameon=False, fontsize=7)
fig.savefig(a.output, dpi=150)
print("wrote", a.output)
