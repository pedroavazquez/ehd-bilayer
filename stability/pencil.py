#!/usr/bin/env python3
"""Matrix-pencil fit of a sampled signal to a sum of damped complex exponentials,
    x(t_n) = sum_j a_j exp(lambda_j t_n),     t_n = t_0 + n dt,
(Hua & Sarkar 1990). Pure numpy; used to extract sigma +- i omega from the time series of
two_layer_forward.py.

    python3 pencil.py series.csv --column vproj_c --t-min 20 [--order 4]
"""
import argparse
import csv

import numpy as np


def matrix_pencil(x, dt, order=None, pencil=None, svd_tol=1e-6):
    """Exponents lambda_j and complex amplitudes a_j (referred to the first sample) of the
    model x_n = sum_j a_j exp(lambda_j n dt). The model order is the number of singular values
    above svd_tol * s_max unless given. Returns (lam, amp, singular values)."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    Lp = pencil or N // 3
    Y = np.array([x[i:i + Lp + 1] for i in range(N - Lp)])      # (N-Lp) x (Lp+1) Hankel
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    M = order or int(np.sum(s > svd_tol * s[0]))
    V = Vh.conj().T[:, :M]
    V1, V2 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    lam = np.log(z) / dt
    Z = np.vander(z, N, increasing=True).T                     # N x M
    amp = np.linalg.lstsq(Z, x, rcond=None)[0]
    return lam, amp, s


def dominant(lam, amp, t_span):
    """Order the exponents by their amplitude at the end of the window."""
    w = np.abs(amp) * np.exp(lam.real * t_span)
    o = np.argsort(-w)
    return lam[o], amp[o]


def _selftest():
    rng = np.random.default_rng(1)
    dt = 0.02
    t = np.arange(0, 30, dt)
    lam0 = 0.031 + 2.687j
    x = (np.exp(lam0 * t) * (0.7 + 0.2j)).real * 2 + 0.3 * np.exp(-0.8 * t) \
        + 1e-6 * rng.standard_normal(t.size)
    lam, amp, s = matrix_pencil(x, dt, order=3)
    lam, amp = dominant(lam, amp, t[-1])
    print("self-test: recovered", np.round(lam[:2], 5), " exact", lam0, np.conj(lam0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--column", default="vproj_c")
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, default=np.inf)
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or a.csv is None:
        _selftest()
        return
    rows = list(csv.DictReader(open(a.csv)))
    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r[a.column]) for r in rows])
    sel = (t >= a.t_min) & (t <= a.t_max)
    t, x = t[sel], x[sel]
    dt = np.median(np.diff(t))
    lam, amp, s = matrix_pencil(x, dt, order=a.order)
    lam, amp = dominant(lam, amp, t[-1] - t[0])
    print(f"window [{t[0]:.3f}, {t[-1]:.3f}], {len(t)} samples, order {len(lam)}")
    for l, c in zip(lam, amp):
        print(f"  lambda = {l.real:+.6f} {l.imag:+.6f}i   |a| = {abs(c):.3e}")


if __name__ == "__main__":
    main()
