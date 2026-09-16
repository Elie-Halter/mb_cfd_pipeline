#!/usr/bin/env bash
# svmp_run.sh — lanceur svMultiPhysics AVEC garde-fous (pc1).
#
# Encode les règles du projet et les bugs déjà payés :
#   * Règle N°3 : JAMAIS 2 svMP simultanés  -> abort si un tourne déjà.
#   * STOP_SIM résiduel   -> un run relancé s'arrêtait au 2e pas (bug vécu). Purgé ici.
#   * libElas.so introuvable en shell non-interactif -> LD_LIBRARY_PATH forcé.
#   * mesh eq PETSc       -> PETSC_OPTIONS positionné.
#   * Tâche longue        -> tmux DÉTACHÉ (survit à la déconnexion ssh/sshfs).
#   * -np doit matcher le stamp du .bin de restart (sinon svMP refuse).
#
# Usage :
#   svmp_run.sh <config.xml> <rundir> [session] [nprocs]
# Exemple :
#   svmp_run.sh ~/4826_sim/cap_pinning/MB_DTHALF_PROD.xml \
#               ~/4826_sim/cap_pinning/MB_DTHALF_PROD  mbdthalf  4
set -euo pipefail

XML="${1:?usage: svmp_run.sh <config.xml> <rundir> [session] [nprocs]}"
RUNDIR="${2:?rundir manquant}"
SESSION="${3:-svmp_$(basename "$RUNDIR")}"
NP="${4:-4}"
SVMP="${SVMP_BIN:-/home/halter/svmp_bin/svmultiphysics}"

[[ -f "$XML"  ]] || { echo "!! XML introuvable : $XML" >&2; exit 2; }
[[ -x "$SVMP" ]] || { echo "!! binaire introuvable : $SVMP" >&2; exit 2; }

# --- Règle N°3 : un seul svMP à la fois ---------------------------------------
if pgrep -x svmultiphysics >/dev/null 2>&1; then
  echo "!! ABORT (Règle N°3) : un svMP tourne déjà :" >&2
  pgrep -a svmultiphysics >&2
  exit 3
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!! ABORT : la session tmux '$SESSION' existe déjà (tmux kill-session -t $SESSION)" >&2
  exit 3
fi

mkdir -p "$RUNDIR"

# --- Purge du STOP_SIM résiduel (bug vécu : run tué au 2e pas) ----------------
if [[ -f "$RUNDIR/STOP_SIM" ]]; then
  echo "[garde-fou] STOP_SIM résiduel trouvé -> supprimé (sinon arrêt immédiat)"
  rm -f "$RUNDIR/STOP_SIM"
fi

# --- Avertissement restart : -np doit matcher le stamp du .bin ----------------
INIT=$(grep -oE '<Simulation_initialization_file_path>[^<]+' "$XML" 2>/dev/null | sed 's/.*>//' | tr -d ' ' || true)
if [[ -n "${INIT:-}" ]]; then
  if [[ ! -f "$INIT" ]]; then
    echo "!! ABORT : fichier de restart déclaré mais introuvable : $INIT" >&2; exit 4
  fi
  STAMP=$(basename "$(dirname "$INIT")")   # ex "4-procs"
  echo "[restart] init = $INIT  (stamp: $STAMP ; lancé avec -np $NP)"
  [[ "$STAMP" == "${NP}-procs" ]] || echo "[ATTENTION] stamp '$STAMP' != '${NP}-procs' : svMP risque de refuser."
fi

echo "[lancement] session=$SESSION  np=$NP  rundir=$RUNDIR"
cd "$RUNDIR"
tmux new -d -s "$SESSION" \
  "export LD_LIBRARY_PATH=/home/halter/lib; \
   export PETSC_OPTIONS=\"-pc_type gamg -pc_gamg_type agg\"; \
   mpirun -np $NP '$SVMP' '$XML' > run.log 2>&1; \
   echo DONE_EXIT_\$? >> run.log"

sleep 8
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[OK] '$SESSION' active — $(pgrep -c -x svmultiphysics 2>/dev/null || echo 0) procs"
  echo "     suivi   : ssh pc1 \"tail -f $RUNDIR/run.log\""
  echo "     dernier : ssh pc1 \"grep -E ' NS [0-9]+-' $RUNDIR/run.log | tail -1\""
  echo "     stop    : ssh pc1 \"touch $RUNDIR/STOP_SIM\"   (arrêt propre)"
else
  echo "!! la session s'est terminée immédiatement — voir $RUNDIR/run.log" >&2
  tail -20 "$RUNDIR/run.log" 2>/dev/null || true
  exit 5
fi
