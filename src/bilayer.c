/**
# Coulomb-driven electroconvection in two stacked layers of miscible dielectric liquids

Simulations of the configuration of Traoré, Pérez, Mondal, Bhattacharya,
Vázquez & Yan, *Fluids* **9**, 219 (2024). Two miscible dielectric
liquids fill the gap between plane electrodes. Liquid 1 (reference,
lighter) occupies the upper half and liquid 2 the lower half. Charge is
injected from the lower electrode (homogeneous, autonomous injection).

## Equations (Eqs. 11-20 of the paper)

Viscous scaling with the properties of liquid 1: length $H$, velocity
$\mu_1/(\rho_1 H)$, time $\rho_1H^2/\mu_1$, potential $\Delta V$,
charge density $q_0$.
$$
\partial_t\mathbf u+\nabla\cdot(\mathbf u\mathbf u)=-\nabla p+
\nabla\cdot\left[\mu\left(\nabla\mathbf u+\nabla\mathbf u^T\right)\right]
+\frac{C_{inj}T^2}{M^2}\,q\mathbf E-\frac{T^2}{M^2}\frac{E^2}{2}\nabla\varepsilon
+\frac{Ra_S}{Sc}\,C\,\hat{\mathbf y}
$$
$$
\partial_tC+\mathbf u\cdot\nabla C=\frac{1}{Sc}\nabla^2C,\qquad
\partial_tq+\nabla\cdot\left[\left(\mathbf u+\frac{T}{M^2}K\mathbf E\right)q\right]=0,
$$
$$
\nabla\cdot(\varepsilon\nabla\phi)=-C_{inj}\,q,\qquad \mathbf E=-\nabla\phi,
$$
$$
\mu=C+(1-C)\mu_r,\quad \varepsilon=C+(1-C)\varepsilon_r,\quad K=C+(1-C)K_r .
$$
$C$ is the concentration of liquid 1, so $C=1$ in the upper layer. The
two electric terms of the momentum equation are written together as the
divergence of the Maxwell stress
$(T^2/M^2)\,\varepsilon(\mathbf E\mathbf E-\tfrac12E^2\mathsf I)$, which is
what [ehd/stress.h](/src/ehd/stress.h) computes. With the Poisson
equation above, its divergence is exactly the Coulomb plus polarization
force.

The buoyancy coefficient is $Gr=Ra_S/Sc$, the Grashof number. It stays
finite in the limit $Sc\to\infty$, which is selected with `-Sc 0`
(no diffusion of $C$) together with `-Gr`.

## Boundary conditions (Fig. 2 of the paper)

* Injector, $y=0$: $u=v=0$, $\phi=1$, $q=1$, $\partial_yC=0$.
* Collector, $y=1$: $u=v=0$, $\phi=0$, $\partial_yq=0$, $\partial_yC=0$.
* Side walls: symmetry, $u=0$, $\partial_xv=\partial_xq=\partial_xC=\partial_x\phi=0$
  (the Basilisk default).

## Numerical method

* Navier--Stokes: [centered.h](/src/navier-stokes/centered.h) (BCG
  advection, implicit viscosity with $\mu(C)$ on faces, approximate
  projection).
* Concentration: [tracer.h](/src/tracer.h) (BCG) and
  [diffusion.h](/src/diffusion.h).
* Charge: conservative BCG transport with the drift face velocity
  $\mathbf w=\mathbf u_f+(T/M^2)K_f\mathbf E_f$. The velocity $\mathbf w$
  is not solenoidal, so the predictor receives the source
  $-q\nabla\cdot\mathbf w$. The flux through the injector is imposed
  exactly, $q_{inj}\,w_y$. There is no charge diffusion, as in the paper.
* Potential: [poisson.h](/src/poisson.h) with $\varepsilon(C)$ on faces.
* The time step satisfies the CFL condition of $\mathbf u$ (centered.h)
  and that of the drift velocity $\mathbf w$ (`-cfl`).
* Adaptive mesh refinement on $C$, $q$ and $\mathbf u$, on a rectangular
  domain $[0,n_x/n_y]\times[0,1]$ made of $n_x\times n_y$ root cells
  (patched `dimensions()`).

## Usage

~~~bash
qcc -O2 -Wall -disable-dimensions bilayer.c -o bilayer -lm
./bilayer -T 175 -C 10 -M 10 -Ras 2000 -Sc 1000 -nx 5 -ny 4 -level 6 -tend 200
~~~

All options are listed in `parse_options()` below and with `-h`.
Output in the current directory:

* `log`: t, dt, Lacey index, Vmax, kinetic energy, mean C, total charge,
  current through the collector, number of cells, and the projection
  $\int v\cos(m\pi x/L)\sin(\pi y)\,dv$ of the vertical velocity on the
  seeded mode (for growth rates and frequencies).
* `C-<t>.png`, `q-<t>.png`, `omega-<t>.png` every `-dtout`.
* `dump-<t>` every `-dtdump`, and `restart` (the latest state). Restart
  with `-restart restart`.
*/

#include "navier-stokes/centered.h"
#include "tracer.h"
#include "diffusion.h"
#include "poisson.h"

scalar phi[];
face vector epsilon[];   // (T^2/M^2) eps(C) on faces, used by ehd/stress.h
#include "ehd/stress.h"

scalar C[], q[];
scalar * tracers = {C};
face vector muv[], epsf[], Kf[];

/**
## Parameters */

double Te = 175., Cinj = 10., Mn = 10.;     // T, C_inj, M
double Ras = 2000., Sc = 1000., Gr = -1.;   // Gr = Ras/Sc unless given
double er = 1., Kr = 1., mur = 1.;          // ratios liquid 2 / liquid 1
int nx = 5, ny = 4;                         // root cells: L = nx/ny
int maxlevel = 6, minlevel = 3;
double tend = 200.;
double C_err = 1e-2, q_err = 1e-2, u_err = 1e-2;
double cfl_q = 0.5, dt_max = 1e-2;
double A = 1e-4;          // amplitude of the initial velocity seed
int m_rolls = 2;          // number of rolls of the seed
double delta0 = 0.;       // initial interface thickness (0: sharp)
int hydro = 0;            // 1: start from the analytical hydrostatic state
double dtout = 1., dtdump = 10.;
int logevery = 10, png_n = 512, adapt_on = 1;
double phitol = 1e-5;       // tolerance of the Poisson solver for the potential
char restart_file[256] = "";

#define drift (Te/sq(Mn))
#define kappa_C ((Sc > 0. && Sc < HUGE) ? 1./Sc : 0.)

static void usage (void)
{
  fprintf (stderr,
	   "options: -T -C -M -Ras -Sc -Gr -er -Kr -mur -nx -ny -level -minlevel\n"
	   "         -tend -Cerr -qerr -uerr -cfl -dtmax -A -m -delta -hydro\n"
	   "         -dtout -dtdump -logevery -png -adapt -phitol -restart\n"
	   "  -Sc 0 means Sc = infinity (no diffusion of C); then give -Gr.\n");
}

static void parse_options (int argc, char * argv[])
{
  for (int i = 1; i < argc; i++) {
    char * o = argv[i];
    if (!strcmp (o, "-h")) { usage(); exit (0); }
    if (i + 1 >= argc) { usage(); exit (1); }
    char * v = argv[++i];
    if      (!strcmp (o, "-T"))        Te = atof (v);
    else if (!strcmp (o, "-C"))        Cinj = atof (v);
    else if (!strcmp (o, "-M"))        Mn = atof (v);
    else if (!strcmp (o, "-Ras"))      Ras = atof (v);
    else if (!strcmp (o, "-Sc"))       Sc = atof (v);
    else if (!strcmp (o, "-Gr"))       Gr = atof (v);
    else if (!strcmp (o, "-er"))       er = atof (v);
    else if (!strcmp (o, "-Kr"))       Kr = atof (v);
    else if (!strcmp (o, "-mur"))      mur = atof (v);
    else if (!strcmp (o, "-nx"))       nx = atoi (v);
    else if (!strcmp (o, "-ny"))       ny = atoi (v);
    else if (!strcmp (o, "-level"))    maxlevel = atoi (v);
    else if (!strcmp (o, "-minlevel")) minlevel = atoi (v);
    else if (!strcmp (o, "-tend"))     tend = atof (v);
    else if (!strcmp (o, "-Cerr"))     C_err = atof (v);
    else if (!strcmp (o, "-qerr"))     q_err = atof (v);
    else if (!strcmp (o, "-uerr"))     u_err = atof (v);
    else if (!strcmp (o, "-cfl"))      cfl_q = atof (v);
    else if (!strcmp (o, "-dtmax"))    dt_max = atof (v);
    else if (!strcmp (o, "-A"))        A = atof (v);
    else if (!strcmp (o, "-m"))        m_rolls = atoi (v);
    else if (!strcmp (o, "-delta"))    delta0 = atof (v);
    else if (!strcmp (o, "-hydro"))    hydro = atoi (v);
    else if (!strcmp (o, "-dtout"))    dtout = atof (v);
    else if (!strcmp (o, "-dtdump"))   dtdump = atof (v);
    else if (!strcmp (o, "-logevery")) logevery = atoi (v);
    else if (!strcmp (o, "-png"))      png_n = atoi (v);
    else if (!strcmp (o, "-adapt"))    adapt_on = atoi (v);
    else if (!strcmp (o, "-phitol"))   phitol = atof (v);
    else if (!strcmp (o, "-restart"))  strncpy (restart_file, v, 255);
    else { fprintf (stderr, "unknown option %s\n", o); usage(); exit (1); }
  }
  if (Gr < 0.)
    Gr = (Sc > 0. && Sc < HUGE) ? Ras/Sc : 0.;
}

/**
## Boundary conditions */

phi[bottom] = dirichlet (1.);
phi[top]    = dirichlet (0.);
q[bottom]   = dirichlet (1.);
q[top]      = neumann (0.);
u.t[bottom] = dirichlet (0.);
u.t[top]    = dirichlet (0.);

int main (int argc, char * argv[])
{
  parse_options (argc, argv);
  dimensions (nx = nx, ny = ny);
  size ((double) nx/ny);
  origin (0., 0.);
  mu = muv;
  /* With nx roots along x, init_grid (N) builds level ceil(log2(N/nx)) in each root:
     nx << maxlevel gives exactly maxlevel, i.e. ny*2^maxlevel cells across the gap. */
  init_grid (nx << maxlevel);
  fprintf (stderr,
	   "# bilayer: T = %g, Cinj = %g, M = %g, Ras = %g, Sc = %g, Gr = %g\n"
	   "#          er = %g, Kr = %g, mur = %g, L = %g, level = %d, cells per unit"
	   " length at max level = %d\n",
	   Te, Cinj, Mn, Ras, Sc, Gr, er, Kr, mur, (double) nx/ny, maxlevel,
	   ny*(1 << maxlevel));
  run();
}

/**
## Properties and potential

The face properties are the mixing laws evaluated with the face
average of $C$, clipped to $[0,1]$. */

static void update_properties (void)
{
  foreach_face() {
    double cf = clamp ((C[] + C[-1])/2., 0., 1.);
    muv.x[] = fm.x[]*(cf + (1. - cf)*mur);
    epsf.x[] = fm.x[]*(cf + (1. - cf)*er);
    epsilon.x[] = sq(Te/Mn)*epsf.x[];
    Kf.x[] = cf + (1. - cf)*Kr;
  }
}

mgstats mgphi;

static mgstats solve_potential (void)
{
  scalar rhs[];
  foreach()
    rhs[] = - Cinj*q[]*cm[];
  return poisson (phi, rhs, epsf, tolerance = phitol);
}

/**
## Analytical hydrostatic state (sharp interface at $y=1/2$)

In each layer $(\varepsilon E)'=C_{inj}q$ and $KqE=J$ is uniform, so
$E^2$ is linear in $y$. Lower layer (liquid 2):
$E^2=E_0^2+2C_{inj}E_0y/\varepsilon_r$, $J=K_rE_0$. Upper layer:
$E^2=E_i^2+2C_{inj}K_rE_0(y-\tfrac12)$ with $E_i=\varepsilon_rE(\tfrac12^-)$.
$E_0$ is fixed by $\int_0^1E\,dy=1$. */

static double Ibar (double a, double b, double h)
{
  return 2./(3.*b)*(pow (a + b*h, 1.5) - pow (a, 1.5));
}

static double base_drop (double E0, double * aL, double * bL, double * aU, double * bU)
{
  *aL = sq(E0); *bL = 2.*Cinj*E0/er;
  *aU = sq(er)*(*aL + *bL*0.5); *bU = 2.*Cinj*Kr*E0;
  return Ibar (*aL, *bL, 0.5) + Ibar (*aU, *bU, 0.5);
}

static void hydrostatic_state (void)
{
  double lo = 1e-12, hi = 10., aL, bL, aU, bU;
  for (int k = 0; k < 200; k++) {
    double mid = (lo + hi)/2.;
    if (base_drop (mid, &aL, &bL, &aU, &bU) > 1.) hi = mid; else lo = mid;
  }
  double E0 = (lo + hi)/2.;
  base_drop (E0, &aL, &bL, &aU, &bU);
  double J = Kr*E0, dropL = Ibar (aL, bL, 0.5);
  foreach() {
    if (y < 0.5) {
      double E = sqrt (aL + bL*y);
      q[] = J/(Kr*E);
      phi[] = 1. - Ibar (aL, bL, y);
    }
    else {
      double E = sqrt (aU + bU*(y - 0.5));
      q[] = J/E;
      phi[] = 1. - dropL - Ibar (aU, bU, y - 0.5);
    }
  }
  fprintf (stderr, "# hydrostatic state: E0 = %.10g\n", E0);
}

/**
## Initial conditions

At rest, all fields zero except the concentration (as in the paper),
or the analytical hydrostatic state with `-hydro 1`. A small seed of
`-m` rolls of amplitude `-A` triggers the instability in a controlled
way, with stream function $\psi=A\sin^2(\pi y)\sin(m\pi x/L)$, which
satisfies the no-slip and symmetry conditions. */

event init (i = 0)
{
  if (!restart_file[0] || !restore (file = restart_file)) {
    double L = (double) nx/ny;
    foreach() {
      C[] = delta0 > 0. ? 0.5*(1. + tanh ((y - 0.5)/delta0)) : (y > 0.5 ? 1. : 0.);
      q[] = 0.; phi[] = 0.;
      u.x[] = A*pi*sin (2.*pi*y)*sin (m_rolls*pi*x/L);
      u.y[] = - A*sq(sin (pi*y))*(m_rolls*pi/L)*cos (m_rolls*pi*x/L);
    }
    if (hydro)
      hydrostatic_state();
  }
  update_properties();
  solve_potential();
  DT = dt_max;
}

/**
## Charge transport

Conservative BCG transport with the drift velocity $\mathbf w$. The
predictor of [bcg.h](/src/bcg.h) solves
$\partial_tq+\mathbf w\cdot\nabla q=-q\nabla\cdot\mathbf w$, so the
source is $-q\nabla\cdot\mathbf w$. The update uses the face fluxes, so
the total charge is conserved up to the fluxes through the electrodes.
The flux through the injector is set to $q_{inj}w_y=w_y$. */

static void drift_velocity (face vector w)
{
  foreach_face()
    w.x[] = uf.x[] - fm.x[]*drift*Kf.x[]*(phi[] - phi[-1])/Delta;
}

event tracer_advection (i++)
{
  face vector w[];
  drift_velocity (w);
  scalar src[];
  foreach() {
    double div = 0.;
    foreach_dimension()
      div += w.x[1] - w.x[];
    src[] = - q[]*div/(Delta*cm[]);
  }
  face vector flux[];
  tracer_fluxes (q, w, flux, dt, src);
  foreach_face (y)
    if (y < Y0 + 1e-6*Delta)
      flux.y[] = w.y[] > 0. ? w.y[] : 0.;
  foreach()
    foreach_dimension()
      q[] += dt*(flux.x[] - flux.x[1])/(Delta*cm[]);
}

/**
## Diffusion of the concentration */

event tracer_diffusion (i++)
{
  if (kappa_C > 0.) {
    face vector D[];
    foreach_face()
      D.x[] = fm.x[]*kappa_C;
    diffusion (C, dt, D);
  }
}

/**
## Properties and potential at the new time level */

event properties (i++)
{
  update_properties();
  mgphi = solve_potential();
}

/**
## Buoyancy

The electric force is added by [ehd/stress.h](/src/ehd/stress.h). */

event acceleration (i++)
{
  face vector av = a;
  foreach_face (y)
    av.y[] += fm.y[]*Gr*(C[] + C[0,-1])/2.;
}

/**
## Diagnostics

Lacey index $\sigma=\left[\langle(C-\tfrac12)^2\rangle\right]^{1/2}$
(volume weighted), maximum velocity, kinetic energy, mean
concentration, total charge and current through the collector. */

event logfile (i++)
{
  if (i % logevery)
    return 0;
  double s2 = 0., vol = 0., vmax = 0., ke = 0., cmean = 0., qtot = 0., Jtop = 0., vp = 0.;
  double L = (double) nx/ny;
  foreach (reduction(+:s2) reduction(+:vol) reduction(max:vmax) reduction(+:ke)
	   reduction(+:cmean) reduction(+:qtot) reduction(+:Jtop) reduction(+:vp)) {
    double dv = sq(Delta)*cm[];
    s2 += sq(C[] - 0.5)*dv;
    vol += dv;
    double u2 = sq(u.x[]) + sq(u.y[]);
    if (u2 > vmax) vmax = u2;
    ke += 0.5*u2*dv;
    cmean += C[]*dv;
    qtot += q[]*dv;
    vp += u.y[]*cos (m_rolls*pi*x/L)*sin (pi*y)*dv;
    if (y + Delta/2. > 1. - 1e-9) {
      double Kc = clamp (C[], 0., 1.) + (1. - clamp (C[], 0., 1.))*Kr;
      Jtop += q[]*(u.y[] + drift*Kc*(phi[0,-1] - phi[0,1])/(2.*Delta))*Delta;
    }
  }
  static FILE * fp = NULL;
  if (!fp) {
    fp = fopen ("log", restart_file[0] ? "a" : "w");
    fprintf (fp, "# t dt lacey Vmax KE Cmean Qtot Jtop ncells vproj mg_p mg_u mg_phi\n");
  }
  fprintf (fp, "%.6f %.4e %.8f %.6e %.6e %.8f %.8f %.6f %ld %.6e %d %d %d\n",
	   t, dt, sqrt (s2/vol), sqrt (vmax), ke, cmean/vol, qtot, Jtop, grid->tn, vp,
	   mgp.i, mgu.i, mgphi.i);
  fflush (fp);
}

/**
## Pictures and dumps */

event pictures (t = 0.; t += dtout; t <= tend)
{
  char name[80];
  sprintf (name, "C-%08.2f.png", t);
  output_ppm (C, file = name, n = png_n, min = 0., max = 1., linear = true);
  sprintf (name, "q-%08.2f.png", t);
  output_ppm (q, file = name, n = png_n, min = 0., max = 1., linear = true);
  scalar omega[];
  vorticity (u, omega);
  sprintf (name, "omega-%08.2f.png", t);
  output_ppm (omega, file = name, n = png_n, linear = true);
}

event dumps (t = dtdump; t += dtdump; t <= tend)
{
  char name[80];
  sprintf (name, "dump-%08.2f", t);
  dump (file = name);
  dump (file = "restart");
}

event finish (t = tend)
{
  dump (file = "restart");
  dump (file = "dump-final");
}

/**
## Mesh adaptation */

#if TREE
event adapt (i++)
{
  if (adapt_on)
    adapt_wavelet ({C, q, u}, (double[]){C_err, q_err, u_err, u_err},
		   maxlevel, minlevel);
}
#endif

/**
## Time step of the drift

The drift CFL condition sets `DT`, which
[centered.h](/src/navier-stokes/centered.h) uses as the maximum time
step at the next iteration. This event runs after all the others of the
iteration. */

event drift_timestep (i++)
{
  face vector w[];
  drift_velocity (w);
  double r = 0.;
  foreach_face (reduction(max:r))
    if (fabs (w.x[])/(fm.x[]*Delta) > r)
      r = fabs (w.x[])/(fm.x[]*Delta);
  DT = min (dt_max, cfl_q/(r + 1e-30));
}
