# Requires the patched multi-root Basilisk; BASILISK must point to its src directory.
QCC ?= $(BASILISK)/qcc
CFLAGS ?= -O2 -Wall -disable-dimensions

all: bilayer
bilayer: src/bilayer.c
	$(QCC) $(CFLAGS) $< -o $@ -lm
bilayer_mpi: src/bilayer.c
	CC99='mpicc -std=c99' $(QCC) $(CFLAGS) -D_MPI=1 -D_GNU_SOURCE $< -o $@ -lm
clean:
	rm -f bilayer bilayer_mpi
.PHONY: all clean
