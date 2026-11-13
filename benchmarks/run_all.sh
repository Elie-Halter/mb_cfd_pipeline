#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- the synthetic analytic-benchmark suite (no patient data).
#
#   B1  free-stream preservation on a moving mesh
#   B2  analytically expanding tube: moving no-slip, induced flow, discrete mass balance
#   B3  Womersley flow in a rigid tube (flow-rate amplitude and phase, profile error)
#
# Usage:
#   bash benchmarks/run_all.sh [options]
# Options:
#   --out DIR       where cases and runs are written (default: benchmarks/_out)
#   --np N          MPI ranks (default 4; reference values were obtained with 4)
#   --cases "..."   subset of B1 B2 B3 (default: all three)
#   --full          also run B2 at dt/2 and dt/4 (temporal order) and B3 on grids g2, g3
#                   (refinement); g3 is ~431k tets and takes several hours
#   --no-solver     generate the cases, render the XML files and run every pre-flight check,
#                   but do not call the solver (cheap; useful to validate an installation)
#   --regen         regenerate the cases even if they already exist
# Environment:
#   SVMP_BIN        official svMultiPhysics (default: $HOME/svmp_bin/svmultiphysics, as built by
#                   install.sh at commit 86729244, unmodified)
# The moving-mesh cases (B1, B2) are written in the pipeline's authoring format and converted by
# tools/to_official_solver.py (FSI equation, fluid domain + mesh equation) before each run.
#   PYTHON          Python interpreter (default python3)
#
# Runs are strictly sequential, and the script refuses to start a run while another
# svmultiphysics process is alive on the machine.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
BIN="${SVMP_BIN:-$HOME/svmp_bin/svmultiphysics}"
OUT="$HERE/_out"
NP=4
CASES="B1 B2 B3"
FULL=0
NOSOLVER=0
REGEN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --np) NP="$2"; shift 2 ;;
    --cases) CASES="$2"; shift 2 ;;
    --full) FULL=1; shift ;;
    --no-solver) NOSOLVER=1; shift ;;
    --regen) REGEN=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

if [ "$NOSOLVER" -eq 0 ]; then
  [ -x "$BIN" ] || { echo "solver not found: $BIN (set SVMP_BIN)" >&2; exit 1; }
  command -v mpirun >/dev/null || { echo "mpirun not found" >&2; exit 1; }
fi

SUMMARY=()
FAIL=0
note() {  # note <name> <exit code> [informative]
  local s="PASS"; [ "$2" -ne 0 ] && s="FAIL"
  if [ "${3:-}" = "info" ]; then s="done"; [ "$2" -ne 0 ] && s="error"; fi
  SUMMARY+=("$(printf '%-34s %s' "$1" "$s")")
  [ "$s" = "FAIL" ] || [ "$s" = "error" ] && FAIL=1
  return 0
}

gen() {  # gen <case_subdir> <out_dir> [extra args]
  local dir="$1" dst="$2"; shift 2
  if [ "$REGEN" -eq 1 ] || [ ! -f "$dst/.generated" ]; then
    rm -rf "$dst"; mkdir -p "$dst"
    "$PY" "$HERE/$dir/gen_case.py" --out "$dst" "$@" || return 1
    touch "$dst/.generated"
  fi
}

solve() {  # solve <bin> <xml> <run_dir>
  local bin="$1" xml="$2" run="$3"
  if pgrep -x svmultiphysics >/dev/null 2>&1; then
    echo "refusing to start: another svmultiphysics process is running" >&2; return 1
  fi
  rm -rf "$run/${NP}-procs" "$run/run.log"
  ( cd "$run" && mpirun -np "$NP" "$bin" "$xml" > run.log 2>&1 )
  local rc=$?
  tail -3 "$run/run.log"
  return $rc
}

# render <template> <xml_out> KEY=VAL...
render() { "$PY" "$HERE/common/render_xml.py" "$@" >/dev/null; }

# official <authoring xml> <run_dir>: write <run_dir>/case_official.xml + per-face displacement files
official() { "$PY" "$HERE/../tools/to_official_solver.py" "$1" "$2/case_official.xml" "$2/disp_official" > "$2/to_official.log" 2>&1; }

# ---------------------------------------------------------------------------- B1
run_B1() {
  local C="$OUT/B1_freestream/case"
  gen B1_freestream "$C" || { note "B1 generate" 1; return; }
  for lab in official; do
    local R="$OUT/B1_freestream/run_$lab"; mkdir -p "$R"
    render "$HERE/B1_freestream/case.xml.in" "$R/case.xml" CASE="$C"
    "$PY" "$HERE/B1_freestream/preflight.py" --case "$C" --xml "$R/case.xml" > "$R/preflight.log" 2>&1
    note "B1 preflight ($lab)" $? ; tail -1 "$R/preflight.log"
    official "$R/case.xml" "$R"; local rc=$?; note "B1 convert to official input" $rc
    [ $rc -eq 0 ] || continue
    [ "$NOSOLVER" -eq 1 ] && continue
    echo "[run_all] B1 $lab: 200 steps (~10-20 min on 4 ranks)"
    solve "$BIN" "$R/case_official.xml" "$R"; note "B1 solver ($lab)" $? info
    "$PY" "$HERE/B1_freestream/analyze.py" "$R" --label "$lab" | tail -2
    note "B1 free stream ($lab)" "${PIPESTATUS[0]}"
  done
}

# ---------------------------------------------------------------------------- B2
run_B2() {
  local C="$OUT/B2_expanding_tube/case"
  gen B2_expanding_tube "$C" || { note "B2 generate" 1; return; }
  local variants="dt1"; [ "$FULL" -eq 1 ] && variants="dt1 dt2 dt4"
  local runs=() labels
  for v in $variants; do
    labels="official"
    local div="${v#dt}"
    read -r DT NSTEPS VTK RST < <("$PY" -c "import sys; sys.path.insert(0,'$HERE/B2_expanding_tube'); \
from params import variant; v=variant($div); print(v['DT'], v['NSTEPS'], v['VTK_INC'], v['RESTART_INC'])")
    for lab in $labels; do
      local R="$OUT/B2_expanding_tube/run_${v}_$lab"; mkdir -p "$R"
      render "$HERE/B2_expanding_tube/case.xml.in" "$R/case.xml" CASE="$C" DT="$DT" NSTEPS="$NSTEPS" \
             VTK_INC="$VTK" RESTART_INC="$RST"
      "$PY" "$HERE/B2_expanding_tube/preflight.py" --case "$C" --xml "$R/case.xml" > "$R/preflight.log" 2>&1
      note "B2 preflight ($v, $lab)" $? ; tail -1 "$R/preflight.log"
      official "$R/case.xml" "$R"; local rcv=$?; note "B2 convert to official input ($v)" $rcv
      [ $rcv -eq 0 ] || continue
      [ "$NOSOLVER" -eq 1 ] && continue
      echo "[run_all] B2 $v $lab: $NSTEPS steps, dt = $DT s"
      solve "$BIN" "$R/case_official.xml" "$R"; note "B2 solver ($v, $lab)" $? info
      "$PY" "$HERE/B2_expanding_tube/analyze.py" "$R" --case "$C" --dt "$DT" --label "$lab" | tail -5
      local rc="${PIPESTATUS[0]}"
      note "B2 expanding tube ($v)" "$rc"
      runs+=("$R:$DT")
    done
  done
  if [ "$NOSOLVER" -eq 0 ] && [ "${#runs[@]}" -ge 2 ]; then
    "$PY" "$HERE/B2_expanding_tube/temporal_order.py" --case "$C" "${runs[@]}" \
         --csv "$OUT/B2_expanding_tube/temporal_order.csv" | tail -12
    note "B2t temporal order" "${PIPESTATUS[0]}"
  fi
}

# ---------------------------------------------------------------------------- B3
run_B3() {
  local C="$OUT/B3_womersley/case"
  local grids="g1"; [ "$FULL" -eq 1 ] && grids="g1 g2 g3"
  "$PY" "$HERE/B3_womersley/womersley.py" | tail -1; note "B3 analytic self-test" "${PIPESTATUS[0]}"
  # shellcheck disable=SC2086
  gen B3_womersley "$C" --grid $grids || { note "B3 generate" 1; return; }
  local NSTEPS; NSTEPS=$("$PY" -c "import sys; sys.path.insert(0,'$HERE/B3_womersley'); \
from params import N_CYCLES, PERIOD, DT; print(int(round(N_CYCLES*PERIOD/DT)))")
  local specs=()
  for gname in $grids; do
    if [ ! -d "$C/mesh_$gname" ]; then
      "$PY" "$HERE/B3_womersley/gen_case.py" --out "$C" --grid "$gname" || { note "B3 generate $gname" 1; continue; }
    fi
    local R="$OUT/B3_womersley/run_$gname"; mkdir -p "$R"
    render "$HERE/B3_womersley/case.xml.in" "$R/case.xml" CASE="$C" GRID="$gname" NSTEPS="$NSTEPS"
    "$PY" "$HERE/B3_womersley/preflight.py" --case "$C" --grid "$gname" --xml "$R/case.xml" > "$R/preflight.log" 2>&1
    note "B3 preflight ($gname)" $? ; tail -1 "$R/preflight.log"
    [ "$NOSOLVER" -eq 1 ] && continue
    echo "[run_all] B3 $gname: $NSTEPS steps"
    solve "$BIN" "$R/case.xml" "$R"; note "B3 solver ($gname)" $? info
    "$PY" "$HERE/B3_womersley/analyze.py" "$R" --label "$gname" | tail -5
    note "B3 Womersley amplitude/phase ($gname)" "${PIPESTATUS[0]}"
    specs+=("$gname:$R")
  done
  if [ "$NOSOLVER" -eq 0 ] && [ "${#specs[@]}" -ge 2 ]; then
    "$PY" "$HERE/B3_womersley/refinement.py" "${specs[@]}" | tail -8
    note "B3r profile error under refinement" "${PIPESTATUS[0]}"
  fi
}

for c in $CASES; do
  echo "=================================================================== $c"
  case "$c" in
    B1) run_B1 ;;
    B2) run_B2 ;;
    B3) run_B3 ;;
    *) echo "unknown case $c" >&2; FAIL=1 ;;
  esac
done


echo
echo "======================== benchmark summary ========================"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "==================================================================="
exit $FAIL
