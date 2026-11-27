# ehd-bilayer: electroconvection in two stacked layers of miscible dielectric liquids

Basilisk script for the configuration of Traoré, Pérez, Mondal, Bhattacharya,
Vázquez & Yan, *Fluids* **9**, 219 (2024). It solves the same equations as the paper,
Eqs. (11)–(20): Boussinesq Navier–Stokes with $\mu(C)$, transport of the concentration
$C$ with diffusivity $1/Sc$, and conservation of the injected charge with mobility
$K(C)$. It also solves Gauss's law with $\varepsilon(C)$ and adds the Coulomb plus
polarization force as the divergence of the Maxwell stress. The scaling is the viscous
one of the paper, with the properties of liquid 1. The boundary conditions are those of
Fig. 2 of the paper. The domain is the rectangle $[0,n_x/n_y]\times[0,1]$, built from
$n_x\times n_y$ root cells of the patched multi-root tree grid, with adaptive mesh
refinement.

## Build

The patched tree must be compiled for its own location (its `qcc` stores the source
path).

```bash
export BASILISK=/path/to/basilisk-multiroot/src
export PATH=$BASILISK:$PATH
make              # serial solver ./bilayer
make bilayer_mpi  # MPI solver
# or by hand:
qcc -O2 -Wall -disable-dimensions src/bilayer.c -o bilayer -lm
# MPI (with this version, _GNU_SOURCE is needed for madvise in memindex/virtual.h)
CC99='mpicc -std=c99' qcc -D_MPI=1 -D_GNU_SOURCE -O2 -Wall -disable-dimensions \
    src/bilayer.c -o bilayer_mpi -lm
mpirun -np 8 ./bilayer_mpi [options]
```

## Options

| option | meaning | default |
|---|---|---|
| `-T -C -M` | $T$, $C_{inj}$, $M$ (liquid 1) | 175, 10, 10 |
| `-Ras -Sc` | solutal Rayleigh and Schmidt numbers | 2000, 1000 |
| `-Gr` | buoyancy coefficient $Ra_S/Sc$; give it with `-Sc 0` | $Ra_S/Sc$ |
| `-Sc 0` | $Sc=\infty$, no diffusion of $C$ | |
| `-er -Kr -mur` | $\varepsilon_2/\varepsilon_1$, $K_2/K_1$, $\mu_2/\mu_1$ | 1, 1, 1 |
| `-nx -ny` | root cells, aspect ratio $L=n_x/n_y$ | 5, 4 |
| `-level -minlevel` | refinement levels inside each root cell | 6, 3 |
| `-adapt` | 1: wavelet adaptation on $C$, $q$, $\mathbf u$ | 1 |
| `-Cerr -qerr -uerr` | adaptation tolerances | 1e-2, 1e-2, 1e-2 |
| `-tend` | final time (viscous units) | 200 |
| `-cfl -dtmax` | CFL number of the charge drift, maximum time step | 0.5, 1e-2 |
| `-A -m` | amplitude and number of rolls of the initial velocity seed | 1e-4, 2 |
| `-delta` | initial interface thickness (0: sharp) | 0 |
| `-hydro` | 1: start from the analytical hydrostatic state | 0 |
| `-dtout -dtdump -logevery -png` | output intervals, log stride, picture size | 1, 10, 10, 512 |
| `-phitol` | tolerance of the Poisson solver for the potential | 1e-5 |
| `-restart file` | restart from a dump (give the same parameters) | |

Cells across the gap at level $\ell$: $n_y2^\ell$ (`-ny 4 -level 6` gives 256).

## Output

* `log`: t, dt, Lacey index $\sigma=\langle(C-\tfrac12)^2\rangle^{1/2}$, $V_{max}$, kinetic
  energy, mean $C$, total charge, current through the collector, number of cells, the
  projection of $v$ on the seeded mode, and the multigrid iterations of the projection,
  the viscous solve and the potential.
* `C-*.png`, `q-*.png`, `omega-*.png` every `-dtout`.
* `dump-*` every `-dtdump`, `restart` (latest), `dump-final`. These can be opened with
  bview.

## Resolution: the injector layer

With $C_{inj}=10$ the base charge drops from 1 to 0.3 within $y\approx0.05$ of the
injector. Linear thresholds and growth rates are only reproduced when this layer is
well resolved. On uniform meshes of 64 or 128 cells across the gap, the homogeneous
case at $T=190$ decays (discrete threshold above 210), although linear theory gives
$\lambda=1.27$. Adaptive refinement on $q$ with `-qerr 1e-3` solves this at a small cost:
level 7 at the injector and level 4 elsewhere (24k cells instead of 327k) gives
$\lambda=1.234$.

## Demonstration run

```bash
mpirun -np 2 ./bilayer_mpi -T 175 -C 10 -M 10 -Ras 2000 -Sc 1000 -nx 5 -ny 4 -level 6 \
    -minlevel 3 -qerr 1e-3 -Cerr 1e-2 -uerr 0.05 -A 1e-4 -m 2 -tend 40 -dtout 2 -dtdump 10
```

Starting from rest, the charge front crosses the gap in about $M^2/T\approx0.6$ time units.
The two liquids then mix by diffusion only until $t\approx25$, when convection sets in and
the Lacey index drops from 0.35 to 0.06 by $t=31.5$ (`examples/demo_T175/history_T175.png`,
`examples/demo_T175/snapshots_T175.png`). In the growth phase $V_{max}$ shows the ripples of
Fig. 10 of the paper. Consecutive minima are 5.3 and 5.7 time units apart. That is the
half period of the Hopf mode of linear theory for $Gr=2$ and an interface already spread
by diffusion to $\delta\approx0.2$–$0.3$, which gives 5.3 to 6.0. The envelope grows at
about 0.22, against 0.25 in linear theory. The run took about 25 minutes up to $t=31.5$
on 2 cores, with about 20k cells. It then slows down to about 0.2 s per step, mostly
spent in the Poisson solver for the potential (see `-phitol`).

## Numerical notes

* The charge is transported with the BCG scheme of `bcg.h`, in conservative form, with
  the drift face velocity $\mathbf w=\mathbf u_f+(T/M^2)K_f\mathbf E_f$. The predictor
  receives the source $-q\nabla\cdot\mathbf w$, and the flux through the injector is
  imposed exactly. There is no charge diffusion, as in the paper.
* The time step obeys the CFL condition of the drift (`-cfl`, default 0.5) in addition
  to that of `centered.h`.
* The electric force is `ehd/stress.h` with permittivity $(T^2/M^2)\varepsilon(C)$. With
  $\nabla\cdot(\varepsilon\nabla\phi)=-C_{inj}q$ its divergence is the Coulomb plus
  polarization force of Eq. (12).
* The properties on faces use the face average of $C$, clipped to $[0,1]$.
* Multi-root grid: `init_grid (N)` builds level $\lceil\log_2(N/n_x)\rceil$ in each root, so
  the script calls `init_grid (nx << level)`.

## Layout

* `src/bilayer.c`: the solver.
* `tools/plot_log.py`: Lacey index and Vmax against time from one or more logs.
* `validation/`: linear tests (`linear_tests.sh`) and their logs.
* `examples/demo_T175/`: log and figures of the demonstration run.
* `docs/`: validation notes and the comparison with the paper.

## Validation

Tests against linear theory and a comparison with Traoré et al. (2024) are in
[`docs/validation.md`](docs/validation.md).
