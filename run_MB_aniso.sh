#!/usr/bin/env bash
# Versioned launcher for the moving-boundary aniso run.
#
# WHY THIS WRAPPER EXISTS: MB_example.xml solves the mesh-motion equation with FSILS,
# for which PETSC_OPTIONS is irrelevant. The alternative is <Linear_algebra type="petsc">
# with the GAMG (algebraic multigrid) preconditioner, used for one of the production
# cases. svMP's XML only exposes `petsc-jacobi` / `petsc-rcs` as <Preconditioner>
# values -- there is NO XML enum for GAMG -- so GAMG is reachable ONLY through the
# PETSc options database, i.e. the PETSC_OPTIONS environment variable below. With
# PETSc and without it, the solver falls back to jacobi and the ill-conditioned fine
# near-wall mesh blows the linear solver up (lsIt ~90k) and inverts around step ~11.
# Keep the export versioned here, never rely on a shell that happens to have it set.
set -euo pipefail

XML="${1:-MB_example.xml}"
RUNDIR="${RUNDIR:-runs/MB}"
NP="${SVMP_NPROCS:-4}"
SVMP="${SVMP_BIN:-svmultiphysics}"

export PETSC_OPTIONS="-pc_type gamg -pc_gamg_type agg"

mkdir -p "$RUNDIR"
cd "$RUNDIR"
echo "[run_MB_aniso] XML=$XML"
echo "[run_MB_aniso] PETSC_OPTIONS=$PETSC_OPTIONS  NP=$NP"
echo "[run_MB_aniso] $SVMP"
exec mpirun -np "$NP" "$SVMP" "$XML"
