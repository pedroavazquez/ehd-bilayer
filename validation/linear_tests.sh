#!/bin/bash
# Linear tests: growth rate (Gr = 0) and Hopf mode (Gr = 100, Sc = infinity) from the
# hydrostatic state, k = 2 pi/1.25 (2 rolls in L = 1.25), fixed mesh.
# Usage: validation/linear_tests.sh <level>
# BILAYER: path to the compiled solver (default ./bilayer). RUNS: output directory (default runs).
B=$(realpath "${BILAYER:-./bilayer}")
lev=${1:-5}
for case in "G0 -T 190 -Gr 0 -Sc 0 -tend 16" "G100 -T 210 -Gr 100 -Sc 0 -tend 40"; do
  set -- $case; name=$1; shift
  d=${RUNS:-runs}/${name}_L$lev; mkdir -p "$d"
  (cd "$d" && "$B" "$@" -hydro 1 -A 1e-4 -m 2 -level $lev -adapt 0 -dtout 4 -dtdump 1000 \
      -logevery 5 -png 256 2> err.log)
done
