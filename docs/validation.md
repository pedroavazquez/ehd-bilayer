# Validation

## Reproducing the paper

The paper starts from rest with all fields zero except $C$, and lets numerical noise
trigger the instability. Our default adds a small seed (`-A 1e-4`) so that the onset
is reproducible. `-A 0` gives the paper's initial condition, but then the onset time
depends on round-off and may be very long, because the problem is symmetric.

```bash
# Fig. 5 to 7 (T = 175, Sc = 1000), aspect ratio 1.25 with two rolls
mpirun -np 8 ./bilayer_mpi -T 175 -Ras 2000 -Sc 1000 -level 6 -qerr 1e-3 -uerr 0.05 -tend 300
# Fig. 9 (epsilon ratio), T = 190, Sc = 1000
mpirun -np 8 ./bilayer_mpi -T 190 -Sc 1000 -Ras 2000 -er 1.5 -level 6 -qerr 1e-3 -uerr 0.05
# Fig. 10 (Vmax for several Sc), T = 175
for Sc in 100 300 1000; do mkdir Sc$Sc; (cd Sc$Sc; ../bilayer -T 175 -Sc $Sc -Ras 2000 ...); done
```

The paper does not give the aspect ratio, the mesh or the value of $Ra_S$ for each
figure. Its forward threshold, $T=167.8$, corresponds to $Ra_S/Sc\approx13$ in linear
theory.

## Validation against linear theory

All tests start from the analytical hydrostatic state (`-hydro 1`) with a seed of two
rolls, $k=2\pi/1.25$, $C=M=10$. Linear values come from the Chebyshev solver of
`firedrake/two_layer`, converted to viscous time with $\lambda_v=\lambda_d\,T/M^2$.

| case | mesh | simulation | linear theory |
|---|---|---|---|
| hydrostatic state, $T=150$, $Gr=100$ | 64 cells | rest kept, $V_{max}\sim10^{-4}$; charge 0.1724, current 0.206 | 0.1727, 0.2076 |
| homogeneous, $T=190$ | adaptive, levels 4–7 | $\lambda=1.234$ | 1.271 |
| $Gr=100$, $Sc=\infty$, $T=210$ | 64 cells | $\lambda=0.409\pm5.243i$ | $0.494\pm5.236i$ |
| same | 128 cells | $\lambda=0.457\pm5.236i$ | same |

The stratified case is a Hopf bifurcation. Its frequency is reproduced to four digits,
and its growth rate converges with resolution.

## Comparison with the paper (Figs. 6a and 10, T = 175, Sc = 1000)

`compare_paper.py` builds `compare_paper.png` from the Basilisk logs, the
digitized curves of Figs. 6a and 10 of the paper and a diffuse-interface linear scan
(`ripple_scan.txt`, from `smooth_modes.py`).

* **Saturated convection.** Vmax tends to 5.85, against 5.8 in Fig. 10. The fully
  mixed state does not depend on Gr or Sc, so this compares the EHD part directly, in
  the same units.
* **Diffusive phase.** Before the onset, Basilisk follows the exact 1D solution for
  Sc = 1000. The paper's curve follows the exact solution for Sc ≈ 2000 (best fit
  2160) to three decimals. The paper's mixing by diffusion is therefore twice as slow as
  its stated Sc. The cause is not known (a factor in the diffusion coefficient, or the
  Lacey index averaged over cells of a non-uniform mesh).
* **Onset time.** The paper starts from rest without a seed, so its onset (t = 107.65)
  is set by numerical noise. Here it is set by `-A`. Fig. 7 of the paper puts the onset
  for the same T = 175 near t = 230, which confirms that onset times are not a
  reproducible quantity. Compare curves shifted in time.
* **Ripples.** Fig. 10 (Sc = 1000) has ripples spaced by 3.6 that grow at 0.087, so
  ω/σ ≈ 10. With Ras = 2000 (Gr = 2) the ripples here are spaced by 5.3–5.7.
  * With the interface thickness of their run (δ ≈ 0.3–0.4 for Sc_eff ≈ 2000), the
    linear theory gives the spacing 3.6 for **Gr ≈ 6–7**.
  * Their growth rate is 2.5–3 times smaller than the linear one at T = 175. This is
    consistent with their numerical threshold being higher than the linear one (Fig. 6c
    gives 167.8–169.9).
  * The paper does not state Ras for these figures. Our guess for reproducing them is
    `-Sc 2000 -Ras 13000` (Gr = 6.5).

