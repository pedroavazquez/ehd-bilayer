# Linear stability of two stacked dielectric layers

Chebyshev collocation solvers used in the paper draft (`../paper`). They need numpy and scipy only.

* `two_layer_modes.py`: sharp interface (Sc → ∞). Base state, generalized eigenproblem with the
  interface conditions, neutral curves T_c(k, Gr), eigenfunctions. Field coordinate s = Ē(y)
  and second-order splitting ζ = (D² − k²)v.
* `smooth_modes.py`: diffuse interface, c̄ = ½(1 + tanh((y − ½)/δ)). Converges to the sharp
  problem as δ². `leading(T, k, Gr, delta)` returns the leading eigenvalue in drift units
  (multiply by T/M² for viscous units).
* `delta_scan.py`: threshold and frequency against δ (data for the paper).
* `pencil.py`: matrix-pencil fit of growth rates and frequencies from time series.

The Firedrake verification (time stepping and eigenvalues of the full equations) lives in
`firedrake/two_layer/` of `pedroavazquez/nonLinearStabilityMonolithic`, branch `two-layer`.
