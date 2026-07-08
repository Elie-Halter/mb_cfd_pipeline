#!/usr/bin/env bash
# End-of-run hook: generate MOTION / VELOCITY / HELICITY movies (+ optional FB-vs-MB
# panel) for a finished svMultiPhysics run, via tools/make_movies.py (PyVista + ffmpeg,
# no ParaView needed).
#
# Usage:
#   run_movies.sh <run_dir> [--wall <wall.vtp>] [--stl-dir <dir>] [--fb-dir <dir>] \
#                 [any other make_movies.py flag]
#
# Example (run dir with results_*.vtu, last cycle 974..1948, dt=1ms, T=0.974s):
#   tools/run_movies.sh <run_dir>/<N>-procs \
#       --wall <mesh>/mesh-surfaces/wall.vtp \
#       --stl-dir <phase_stl_dir> \
#       --cycle-start 974 --cycle-end 1948 --dt 0.001 --period 0.974
#
# Hook it into a chain script (e.g. chain_fullmotion.sh) by appending, AFTER the
# solver step that produces the last VTU of the run:
#   "$PIPELINE_CODE/tools/run_movies.sh" "$RUN_DIR" --wall "$WALL_VTP" \
#       --stl-dir "$STL_DIR" --cycle-start "$CYCLE_START" --cycle-end "$CYCLE_END" \
#       --dt "$DT" --period "$PERIOD" >> "$RUN_DIR/movies.log" 2>&1 &
# (backgrounded with `&` so it doesn't block/slow down a subsequent chain step; or
# run it in the foreground as the very last line if you want the chain to wait for
# the movies before declaring done. Do NOT run it while another svMultiPhysics job
# is active on the same machine -- rendering is CPU-light but frame-by-frame VTU
# reads add I/O; keep --max-frames small if you must render concurrently.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "usage: $0 <run_dir> [make_movies.py options...]" >&2
    exit 1
fi

RUN_DIR="$1"; shift
python3 "$HERE/make_movies.py" "$RUN_DIR" "$@"
