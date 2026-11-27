#!/usr/bin/env python3
"""Temporal eigenvalues of two stacked dielectric layers with unipolar injection, Sc = infinity.

Sharp interface at y = 1/2 (no diffusion of the concentration). Lower layer [0, 1/2] is
liquid 2 (injector side, denser), upper layer [1/2, 1] is liquid 1 (reference liquid).
Same drift scaling as temporal_modes.py: H, H^2/(K1 V0), K1 V0/H, V0, q_i, with T, C, M
built with the properties of liquid 1 (as in Traore et al., Fluids 2024).

Bulk equations in each layer (eps, K, mu constant; nu = mu M^2/T):
  u_t + u.grad u = -grad p + nu Lap u + C M^2 q E + G c y_hat
  q_t + div[q (u + K E)] = 0,   eps Lap phi = -C q,   E = -grad phi
G = Gr M^4/T^2 is the buoyancy parameter in drift units, Gr = Ra_S/Sc the viscous Grashof
number of the paper. c = 1 in the upper (lighter) liquid, 0 in the lower one.

Linearized about the two-layer hydrostatic state, normal modes exp(ikx + lambda t), with
zeta = (D^2 - k^2) v (second-order splitting of the momentum equation):
  lambda zeta = nu (D^2-k^2) zeta - k^2 C M^2 (E_b q + q_b' phi),   0 = zeta - (D^2-k^2) v
  lambda q    = -q_b' v - K E_b q_y - 2 (K C/eps) q_b q + K q_b' phi_y
  0           = eps (D^2-k^2) phi + C q
Walls: v = v_y = 0, phi = 0 at y = 0, 1; q = 0 at the injector y = 0.
Interface y = 1/2, [f] = f(1/2+) - f(1/2-):
  [v] = 0,  [v_y] = 0,  [nu (zeta + 2 k^2 v)] = 0                 (velocity, shear stress)
  [-nu zeta_y + 2 nu k^2 v_y] + k^2 C M^2 [q_b phi] - k^2 M^2 [eps E_b phi_y] = k^2 G eta
                                                                 (normal stress)
  lambda eta = v(1/2)                                            (kinematic)
  [phi] = eta [E_b],  [eps phi_y] = eta C [q_b]                  (potential, normal D)
  [K (E_b q - q_b phi_y)] = 0                                    (normal current)

Discretization: Chebyshev collocation in each layer on the local field coordinate
s = E_b(y) (y = y0 + (s^2 - a)/b), which removes the branch point of E_b just below the
injector and makes the charge advection K E_b d/dy = (K b/2) d/ds uniform. The momentum
equation is split into two second-order equations (conditioning ~N^4 instead of ~N^8).
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq, minimize_scalar


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2, np.ones(N - 1), 2] * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(1))
    return D, x


def base(C, er=1., Kr=1., yl=None, yu=None):
    """Two-layer hydrostatic state. Lower layer (eps=er, K=Kr), upper layer (eps=1, K=1).
    Lower: E^2 = E0^2 + 2 C E0 y/er.  Upper: E^2 = Ei^2 + 2 C Kr E0 (y - 1/2), Ei = er E_l(1/2).
    Current J = Kr E0 (q = 1 at y = 0). E0 fixed by int_0^1 E dy = 1."""
    def parts(E0):
        aL, bL = E0**2, 2 * C * E0 / er                  # E^2 = a + b (y - y0)
        El = np.sqrt(aL + bL * 0.5)
        aU, bU = (er * El)**2, 2 * C * Kr * E0
        I = lambda a, b, h: 2 / (3 * b) * ((a + b * h)**1.5 - a**1.5)
        return aL, bL, aU, bU, I(aL, bL, 0.5) + I(aU, bU, 0.5)
    E0 = brentq(lambda E0: parts(E0)[-1] - 1, 1e-12, 10)
    aL, bL, aU, bU, _ = parts(E0)
    J = Kr * E0
    out = {"E0": E0, "J": J, "par": {"L": (aL, bL, 0.), "U": (aU, bU, .5)}}
    for name, y, a, b, y0, eps, K in (("L", yl, aL, bL, 0., er, Kr), ("U", yu, aU, bU, .5, 1., 1.)):
        if y is None:
            y = np.array([y0, y0 + .5])
        E = np.sqrt(a + b * (y - y0))
        q = J / (K * E)
        dE = C * q / eps
        dq = -J * dE / (K * E**2)
        out[name] = dict(y=y, E=E, q=q, dE=dE, dq=dq, eps=eps, K=K)
    return out


def layer_grid(N, a, b, y0, h=0.5, mapped=True):
    """Nodes y (index 0 at y0) and d/dy on one layer."""
    Dx, x = cheb(N)
    if not mapped:
        return y0 + (1 - x) * h / 2, -(2 / h) * Dx
    sa, sb = np.sqrt(a), np.sqrt(a + b * h)
    s = sa + (sb - sa) * (1 - x) / 2
    jac = 2 * s / b                              # dy/ds
    D = np.diag(1 / jac) @ (-(2 / (sb - sa)) * Dx)
    return y0 + (s**2 - a) / b, D


def operators(T, C=10., M=10., k=2 * np.pi / 1.228, Gr=0., er=1., Kr=1., mr=1., N=24,
              mapped=True, inertia=True, charge_lag=True):
    """Matrices A, B of A x = lambda B x.
    Unknowns (per layer v, zeta, q, phi): [vL, zL, qL, pL, vU, zU, qU, pU, eta].
    Mechanism tests: inertia=False drops lambda zeta (Stokes flow, velocity slaved to the
    forces); charge_lag=False drops lambda q (charge slaved to the instantaneous velocity,
    no transit-time delay). The interface keeps its own dynamics, lambda eta = v(1/2)."""
    n = N + 1
    par = base(C, er, Kr)["par"]
    grids = {nm: layer_grid(N, *par[nm], mapped=mapped) for nm in "LU"}
    b = base(C, er, Kr, grids["L"][0], grids["U"][0])
    L, U = b["L"], b["U"]
    L["mu"], U["mu"] = mr, 1.
    G = Gr * M**4 / T**2
    I = np.eye(n)
    m = 8 * n + 1
    A = np.zeros((m, m)); B = np.zeros((m, m))
    off = {"L": 0, "U": 4 * n}
    ie = 8 * n
    for nm, s in (("L", L), ("U", U)):
        s["y"], s["D"] = grids[nm]
        D = s["D"]; s["D2"] = D @ D
        o = off[nm]; cv, cz, cq, cp = o, o + n, o + 2 * n, o + 3 * n
        s["c"] = (cv, cz, cq, cp)
        Lap = s["D2"] - k**2 * I
        nu = s["mu"] * M**2 / T
        s["nu"] = nu
        K, eps = s["K"], s["eps"]
        E, q, dq = np.diag(s["E"]), np.diag(s["q"]), np.diag(s["dq"])
        # zeta-definition rows (cv): 0 = zeta - Lap v
        A[cv:cv + n, cz:cz + n] = I
        A[cv:cv + n, cv:cv + n] = -Lap
        # momentum rows (cz): lambda zeta = nu Lap zeta - k^2 C M^2 (E q + q' phi)
        A[cz:cz + n, cz:cz + n] = nu * Lap
        A[cz:cz + n, cq:cq + n] = -k**2 * C * M**2 * E
        A[cz:cz + n, cp:cp + n] = -k**2 * C * M**2 * dq
        B[cz:cz + n, cz:cz + n] = I if inertia else 0.
        # charge rows (cq)
        A[cq:cq + n, cv:cv + n] = -dq
        A[cq:cq + n, cq:cq + n] = -K * E @ D - 2 * K * C / eps * q
        A[cq:cq + n, cp:cp + n] = K * dq @ D
        B[cq:cq + n, cq:cq + n] = I if charge_lag else 0.
        # Poisson rows (cp)
        A[cp:cp + n, cp:cp + n] = eps * Lap
        A[cp:cp + n, cq:cq + n] = C * I

    bot, top = 0, N
    def setrow(r, entries):
        A[r] = 0; B[r] = 0
        for col0, vec in entries:
            if np.isscalar(vec):
                A[r, col0] += vec
            else:
                A[r, col0:col0 + n] += vec
    e = lambda j: I[j]
    vL, zL, qL, pL = L["c"]; vU, zU, qU, pU = U["c"]
    DL, DU = L["D"], U["D"]
    jL, jU = top, bot                            # interface node of each layer
    # ---- walls: zeta-def row -> v = 0, momentum row -> v_y = 0; phi = 0; q = 0 at injector
    setrow(vL + bot, [(vL, e(bot))]); setrow(zL + bot, [(vL, DL[bot])])
    setrow(vU + top, [(vU, e(top))]); setrow(zU + top, [(vU, DU[top])])
    setrow(pL + bot, [(pL, e(bot))]); setrow(pU + top, [(pU, e(top))])
    setrow(qL + bot, [(qL, e(bot))])
    # ---- interface
    setrow(vL + jL, [(vU, e(jU)), (vL, -e(jL))])                                   # [v]
    setrow(zL + jL, [(vU, DU[jU]), (vL, -DL[jL])])                                 # [v_y]
    setrow(vU + jU, [(zU, U["nu"] * e(jU)), (vU, 2 * k**2 * U["nu"] * e(jU)),
                     (zL, -L["nu"] * e(jL)), (vL, -2 * k**2 * L["nu"] * e(jL))])   # shear
    setrow(zU + jU, [(zU, -U["nu"] * DU[jU]), (vU, 2 * k**2 * U["nu"] * DU[jU]),
                     (zL, L["nu"] * DL[jL]), (vL, -2 * k**2 * L["nu"] * DL[jL]),
                     (pU, k**2 * C * M**2 * U["q"][jU] * e(jU)
                      - k**2 * M**2 * U["eps"] * U["E"][jU] * DU[jU]),
                     (pL, -k**2 * C * M**2 * L["q"][jL] * e(jL)
                      + k**2 * M**2 * L["eps"] * L["E"][jL] * DL[jL]),
                     (ie, -k**2 * G)])                                             # normal stress
    setrow(pL + jL, [(pU, e(jU)), (pL, -e(jL)), (ie, -(U["E"][jU] - L["E"][jL]))])  # [phi]
    setrow(pU + jU, [(pU, U["eps"] * DU[jU]), (pL, -L["eps"] * DL[jL]),
                     (ie, -C * (U["q"][jU] - L["q"][jL]))])                        # [eps phi_y]
    setrow(qU + jU, [(qU, U["K"] * U["E"][jU] * e(jU)), (pU, -U["K"] * U["q"][jU] * DU[jU]),
                     (qL, -L["K"] * L["E"][jL] * e(jL)),
                     (pL, L["K"] * L["q"][jL] * DL[jL])])                          # current
    A[ie, vL + jL] = 1.; B[ie, ie] = 1.                                            # kinematic
    return A, B, dict(base=b, n=n, eta=ie)


def modes(T, return_vectors=False, **kw):
    """Finite eigenvalues (drift time units) sorted by decreasing real part."""
    A, B, info = operators(T, **kw)
    if return_vectors:
        w, V = eig(A, B)
    else:
        w, V = eig(A, B, right=False), None
    ok = np.isfinite(w) & (np.abs(w) < 1e8)
    w = w[ok]
    o = np.argsort(-w.real)
    if return_vectors:
        return w[o], V[:, ok][:, o], info
    return w[o]


def physical(w, Gr, tol=1e-10):
    """Drop the passive interface mode lambda = 0 (eta decoupled when Gr = 0, equal properties)."""
    return w[np.abs(w) > tol] if Gr == 0 else w


def converged(T, n=8, N=24, dN=8, rtol=1e-6, **kw):
    """Eigenvalues present (within rtol relative) at N and N + dN."""
    Gr = kw.get("Gr", 0.)
    w1 = physical(modes(T, N=N, **kw), Gr)
    w2 = physical(modes(T, N=N + dN, **kw), Gr)
    out = [w for w in w1 if np.min(np.abs(w2 - w)) < rtol * max(1, abs(w))]
    return np.array(out[:n])


def leading(T, N=24, rtol=1e-4, **kw):
    """Leading physical eigenvalue (persists from N to N + 8 within rtol)."""
    return converged(T, n=1, N=N, dN=8, rtol=rtol, **kw)[0]


def neutral_T(k, Tlo=60., Thi=400., **kw):
    """Marginal T at wavenumber k for the leading mode."""
    return brentq(lambda T: leading(T, k=k, **kw).real, Tlo, Thi, xtol=1e-7)


def critical(kbounds=(2.5, 8.), Tlo=60., Thi=400., **kw):
    """(T_c, k_c, leading eigenvalue at criticality), minimum of the neutral curve."""
    r = minimize_scalar(lambda k: neutral_T(k, Tlo=Tlo, Thi=Thi, **kw), bounds=kbounds,
                        method="bounded", options=dict(xatol=1e-6))
    return r.fun, r.x, leading(r.fun, k=r.x, **kw)
