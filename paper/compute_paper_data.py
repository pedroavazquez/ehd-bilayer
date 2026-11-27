#!/usr/bin/env python3
"""Data for the figures of the paper that are not already in the repository (Chebyshev,
sharp interface, C = M = 10, equal properties): neutral curves T(k) with the type of the
leading mode, and the eigenfunctions of the Hopf and two-tier modes."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stability"))
import two_layer_modes as tl

C, M = 10., 10.
out = {"neutral": {}, "eig": {}}
ks = np.r_[np.linspace(1.5, 9., 16), np.linspace(9.5, 16., 8)]
for Gr in (0., 10., 100., 1000., 3000.):
    rows = []
    for k in ks:
        try:
            T = tl.neutral_T(k, Tlo=100., Thi=2500., C=C, M=M, Gr=Gr)
            lam = tl.leading(T, k=k, C=C, M=M, Gr=Gr)
            rows.append([float(k), float(T), float(abs(lam.imag))])
        except ValueError:
            pass
    out["neutral"][str(Gr)] = rows
    print("Gr", Gr, "done", len(rows), flush=True)
# stationary two-tier branch alone (Gr-independent): Gr = 1e6
rows = []
for k in np.linspace(6., 16., 21):
    T = tl.neutral_T(k, Tlo=300., Thi=2500., C=C, M=M, Gr=1e6)
    rows.append([float(k), float(T), 0.0])
out["neutral"]["twotier"] = rows

def eig(T, k, Gr):
    w, V, info = tl.modes(T, k=k, C=C, M=M, Gr=Gr, N=32, return_vectors=True)
    if Gr == 0:
        keep = np.abs(w) > 1e-8; w, V = w[keep], V[:, keep]
    n = info["n"]; x = V[:, 0]
    b = info["base"]
    yL, yU = b["L"]["y"], b["U"]["y"]
    v = np.r_[x[:n], x[4 * n + 1:5 * n]]
    q = np.r_[x[2 * n:3 * n], x[6 * n + 1:7 * n]]
    y = np.r_[yL, yU[1:]]
    s = v[np.argmax(np.abs(v))]
    v, q = v / s, q / s
    return dict(lam=[float(w[0].real), float(w[0].imag)], y=y.tolist(),
                v_re=v.real.tolist(), v_im=v.imag.tolist(), q_re=q.real.tolist(), q_im=q.imag.tolist(),
                eta=[float((x[-1] / s).real), float((x[-1] / s).imag)], k=k, T=T, Gr=Gr)
out["eig"]["homog"] = eig(164.0614, 5.1125, 0.)
out["eig"]["hopf100"] = eig(192.0616, 4.8744, 100.)
out["eig"]["hopf1000"] = eig(352.5673, 4.0438, 1000.)
out["eig"]["twotier"] = eig(477.6823, 10.4165, 5000.)
json.dump(out, open(HERE / "paper_data.json", "w"))
print("saved")
