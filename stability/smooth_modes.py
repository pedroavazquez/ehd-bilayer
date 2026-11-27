#!/usr/bin/env python3
"""Temporal eigenvalues of a stratified layer with unipolar injection, Sc = infinity, smooth
concentration profile c_b(y) = (1 + tanh((y - 1/2)/delta))/2, equal electrical properties.

Without diffusion any c_b(y) with u = 0 is a steady state. Drift scaling as in
two_layer_modes.py (G = Gr M^4/T^2). The concentration perturbation is written with the
vertical displacement xi, c' = -c_b' xi, lambda xi = v:
  lambda zeta = nu (D^2-k^2) zeta - k^2 C M^2 (E_b q + q_b' phi) + k^2 G c_b' xi
  0 = zeta - (D^2-k^2) v
  lambda q = -q_b' v - E_b q_y - 2 C q_b q + q_b' phi_y
  0 = (D^2-k^2) phi + C q
  lambda xi = v
As delta -> 0, c_b' -> delta(y - 1/2) and xi(1/2) -> eta: the sharp-interface problem of
two_layer_modes.py. Two Chebyshev domains split at y = 1/2 (where the profile is steepest),
each mapped with the field coordinate s = E_b(y) and, optionally, clustered towards y = 1/2.

Mode selection: without diffusion the stratified fluid carries slow buoyancy modes that
accumulate at lambda = 0 (exact zeros where c_b' underflows, overdamped modes where it is
small), so "largest real part" is not a valid selector. leading() therefore follows the
oscillatory (Hopf) branch, |Im lambda| > imin, which is the branch whose threshold depends
on Gr. The stationary branches (Gr = 0, two-tier) are computed with the sharp solver.
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq, minimize_scalar
import two_layer_modes as tl


def grid(N, a, b, y0, h=0.5, cluster=0.0, toward="top"):
    """Field-coordinate grid of one layer; cluster > 0 adds a sinh stretching of the
    Chebyshev variable towards one end (the interface)."""
    Dx, x = tl.cheb(N)
    if cluster > 0:
        # xi in [-1, 1] -> x in [-1, 1], points concentrated at x = -1 (top of the layer in
        # the index-0-at-bottom convention, y = y0 + (1 - x) h/2) or x = +1 (bottom)
        sgn = 1. if toward == "top" else -1.
        A = cluster
        xm = sgn * (1 - 2 * np.sinh(A * (1 - sgn * x) / 2) / np.sinh(A))
        dxm = A * np.cosh(A * (1 - sgn * x) / 2) / np.sinh(A)      # d xm / d x
        D0 = np.diag(1 / dxm) @ Dx
        x = xm
    else:
        D0 = Dx
    sa, sb = np.sqrt(a), np.sqrt(a + b * h)
    s = sa + (sb - sa) * (1 - x) / 2
    jac = 2 * s / b
    D = np.diag(1 / jac) @ (-(2 / (sb - sa)) * D0)
    return y0 + (s**2 - a) / b, D


def operators(T, C=10., M=10., k=5., Gr=0., delta=0.02, N=48, cluster=0.0, inertia=True,
              charge_lag=True):
    n = N + 1
    par = tl.base(C)["par"]
    gL = grid(N, *par["L"], cluster=cluster, toward="top")
    gU = grid(N, *par["U"], cluster=cluster, toward="bottom")
    b = tl.base(C, 1., 1., gL[0], gU[0])
    G = Gr * M**4 / T**2
    nu = M**2 / T
    I = np.eye(n)
    m = 10 * n
    A = np.zeros((m, m)); B = np.zeros((m, m))
    lay = {}
    for nm, g, o in (("L", gL, 0), ("U", gU, 5 * n)):
        s = b[nm]; s["y"], s["D"] = g
        D = s["D"]; s["D2"] = D @ D
        s["c"] = (o, o + n, o + 2 * n, o + 3 * n, o + 4 * n)
        cv, cz, cq, cp, cx = s["c"]
        dc = 0.5 / delta / np.cosh((s["y"] - 0.5) / delta)**2      # c_b'
        s["dc"] = dc
        Lap = s["D2"] - k**2 * I
        E, q, dq = np.diag(s["E"]), np.diag(s["q"]), np.diag(s["dq"])
        A[cv:cv + n, cz:cz + n] = I; A[cv:cv + n, cv:cv + n] = -Lap
        A[cz:cz + n, cz:cz + n] = nu * Lap
        A[cz:cz + n, cq:cq + n] = -k**2 * C * M**2 * E
        A[cz:cz + n, cp:cp + n] = -k**2 * C * M**2 * dq
        A[cz:cz + n, cx:cx + n] = k**2 * G * np.diag(dc)
        B[cz:cz + n, cz:cz + n] = I if inertia else 0.
        A[cq:cq + n, cv:cv + n] = -dq
        A[cq:cq + n, cq:cq + n] = -E @ D - 2 * C * q
        A[cq:cq + n, cp:cp + n] = dq @ D
        B[cq:cq + n, cq:cq + n] = I if charge_lag else 0.
        A[cp:cp + n, cp:cp + n] = Lap
        A[cp:cp + n, cq:cq + n] = C * I
        A[cx:cx + n, cv:cv + n] = I
        B[cx:cx + n, cx:cx + n] = I
        lay[nm] = s
    L, U = lay["L"], lay["U"]
    bot, top = 0, N
    def setrow(r, entries):
        A[r] = 0; B[r] = 0
        for col0, vec in entries:
            A[r, col0:col0 + n] += vec
    e = lambda j: I[j]
    vL, zL, qL, pL, xL = L["c"]; vU, zU, qU, pU, xU = U["c"]
    DL, DU = L["D"], U["D"]
    # walls
    setrow(vL + bot, [(vL, e(bot))]); setrow(zL + bot, [(vL, DL[bot])])
    setrow(vU + top, [(vU, e(top))]); setrow(zU + top, [(vU, DU[top])])
    setrow(pL + bot, [(pL, e(bot))]); setrow(pU + top, [(pU, e(top))])
    setrow(qL + bot, [(qL, e(bot))])
    # interface y = 1/2: plain continuity (smooth profile, equal properties)
    setrow(vL + top, [(vU, e(bot)), (vL, -e(top))])
    setrow(zL + top, [(vU, DU[bot]), (vL, -DL[top])])
    setrow(vU + bot, [(zU, e(bot)), (zL, -e(top))])
    setrow(zU + bot, [(zU, DU[bot]), (zL, -DL[top])])
    setrow(pL + top, [(pU, e(bot)), (pL, -e(top))])
    setrow(pU + bot, [(pU, DU[bot]), (pL, -DL[top])])
    setrow(qU + bot, [(qU, e(bot)), (qL, -e(top))])
    return A, B, dict(n=n, L=L, U=U)


def spectrum(T, imin=1e-3, **kw):
    """Oscillatory eigenvalues (|Im| > imin), sorted by decreasing real part."""
    A, B, info = operators(T, **kw)
    w = eig(A, B, right=False)
    w = w[np.isfinite(w) & (np.abs(w) < 1e8) & (np.abs(w.imag) > imin)]
    return w[np.argsort(-w.real)]


def leading(T, N=48, rtol=1e-4, dN=16, **kw):
    """Leading electroconvective eigenvalue, kept if it persists from N to N + dN."""
    w1 = spectrum(T, N=N, **kw)
    w2 = spectrum(T, N=N + dN, **kw)
    for w in w1:
        if np.min(np.abs(w2 - w)) < rtol * max(1, abs(w)):
            return w
    raise RuntimeError("no converged mode")


def neutral_T(k, Tlo=100., Thi=1000., **kw):
    return brentq(lambda T: leading(T, k=k, **kw).real, Tlo, Thi, xtol=1e-6)
