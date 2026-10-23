#!/usr/bin/env bash
# svmp_run.sh -- svMultiPhysics launcher with safeguards.
#
# Encodes rules learned the hard way:
#   * never two solver runs at once on the machine  -> abort if one is running;
#   * a stale STOP_SIM file stops a relaunched run at its second step -> purged here;
#   * libElas.so not found in a non-interactive shell -> LD_LIBRARY_PATH set;
#   * mesh equation on PETSc -> PETSC_OPTIONS set to GAMG;
#   * long runs -> DETACHED tmux session (survives an ssh disconnection);
#   * -np must match the stamp of the restart .bin (otherwise the solver refuses).
#
# Usage:
#   svmp_run.sh <config.xml> <rundir> [session] [nprocs]
# Example:
#   svmp_run.sh cases/MB_case.xml cases/MB_case/run1 mbrun1 4
# Binary: $SVMP_BIN (default $HOME/svmp_bin/svmultiphysics, as installed by install.sh).
set -euo pipefail

XML="${1:?usage: svmp_run.sh <config.xml> <rundir> [session] [nprocs]}"
RUNDIR="${2:?missing rundir}"
SESSION="${3:-svmp_$(basename "$RUNDIR")}"
NP="${4:-4}"
SVMP="${SVMP_BIN:-$HOME/svmp_bin/svmultiphysics}"

[[ -f "$XML"  ]] || { echo "!! XML not found: $XML" >&2; exit 2; }
[[ -x "$SVMP" ]] || { echo "!! solver binary not found: $SVMP" >&2; exit 2; }

# --- one solver run at a time --------------------------------------------------
if pgrep -x svmultiphysics >/dev/null 2>&1; then
  echo "!! ABORT: a solver run is already active:" >&2
  pgrep -a svmultiphysics >&2
  exit 3
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!! ABORT: tmux session '$SESSION' already exists (tmux kill-session -t $SESSION)" >&2
  exit 3
fi

mkdir -p "$RUNDIR"

# --- stale STOP_SIM -------------------------------------------------------------
if [[ -f "$RUNDIR/STOP_SIM" ]]; then
  echo "[safeguard] stale STOP_SIM found -> removed (it would stop the run immediately)"
  rm -f "$RUNDIR/STOP_SIM"
fi

# --- restart: -np must match the stamp of the .bin -----------------------------
INIT=$(grep -oE '<Simulation_initialization_file_path>[^<]+' "$XML" 2>/dev/null | sed 's/.*>//' | tr -d ' ' || true)
if [[ -n "${INIT:-}" ]]; then
  if [[ ! -f "$INIT" ]]; then
    echo "!! ABORT: declared restart/initialization file not found: $INIT" >&2; exit 4
  fi
  STAMP=$(basename "$(dirname "$INIT")")   # e.g. "4-procs"
  echo "[restart] init = $INIT  (stamp: $STAMP; launched with -np $NP)"
  [[ "$STAMP" == "${NP}-procs" ]] || echo "[WARNING] stamp '$STAMP' != '${NP}-procs': the solver may refuse it."
fi

echo "[launch] session=$SESSION  np=$NP  rundir=$RUNDIR"
cd "$RUNDIR"
tmux new -d -s "$SESSION" \
  "export LD_LIBRARY_PATH=\"$HOME/lib:\${LD_LIBRARY_PATH:-}\"; \
   export PETSC_OPTIONS=\"-pc_type gamg -pc_gamg_type agg\"; \
   mpirun -np $NP '$SVMP' '$XML' > run.log 2>&1; \
   echo DONE_EXIT_\$? >> run.log"

sleep 8
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[OK] '$SESSION' active - $(pgrep -c -x svmultiphysics 2>/dev/null || echo 0) processes"
  echo "     follow : tail -f $RUNDIR/run.log"
  echo "     last   : grep -E ' NS [0-9]+-' $RUNDIR/run.log | tail -1"
  echo "     stop   : touch $RUNDIR/STOP_SIM   (clean stop)"
else
  echo "!! the session ended immediately - see $RUNDIR/run.log" >&2
  tail -20 "$RUNDIR/run.log" 2>/dev/null || true
  exit 5
fi
