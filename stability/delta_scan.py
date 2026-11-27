#!/usr/bin/env python3
"""Hopf threshold at k = 5 (k = 4 for Gr = 1000 too) versus the interface thickness delta of
c_b = (1 + tanh((y - 1/2)/delta))/2, Sc = infinity, C = M = 10; delta -> 0 must give the
sharp-interface values of two_layer_modes.py."""
import numpy as np
import smooth_modes as sm
import two_layer_modes as tl
M = 10.
for Gr, k in ((10., 5.), (100., 5.), (1000., 5.)):
    Ts = tl.neutral_T(k, Tlo=100., Thi=1000., C=10., M=M, Gr=Gr)
    ws = tl.leading(Ts, k=k, C=10., M=M, Gr=Gr)
    print(f"Gr={Gr:6g} k={k}: sharp      T_n={Ts:9.4f}  omega={abs(ws.imag):.4f}", flush=True)
    for d in (0.08, 0.04, 0.02, 0.01, 0.005):
        N = 48 if d >= 0.01 else 80
        T = sm.neutral_T(k, Tlo=0.7 * Ts, Thi=1.3 * Ts, C=10., M=M, Gr=Gr, delta=d, N=N)
        w = sm.leading(T, k=k, C=10., M=M, Gr=Gr, delta=d, N=N)
        print(f"            delta={d:5.3f}  T_n={T:9.4f}  omega={abs(w.imag):.4f}  (T_n - sharp)/sharp = {T/Ts-1:+.2e}", flush=True)
