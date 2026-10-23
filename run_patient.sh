#!/usr/bin/env bash
# ============================================================================
#  run_patient.sh -- full FB + MB pipeline for ONE case, production configuration.
#
#  Chains:
#    1. reference mesh (all-tet + surfaces, GID/EID/FaceID)
#    2. wall motion: boundary registration onto the segmented phases
#       (morph/pipeline.py --register-only) -> cap-pinned morph with a 12 mm
#       smoothstep taper (morph/run_pinned_morph.py, gate G1) -> wall-only
#       STANDARD displacement file (tools/extended_to_standard.py --face wall.vtp)
#    3. RCR calibration on the MRI flow split
#       FB/MB XMLs from FB_example.xml / MB_example.xml (mesh equation with
#       Dir 0 on every cap, wall Prescribed_displacement), inflow file with the
#       sign and period checks
#    4. FB run (rigid wall), gate G2 on the outlet flow split
#    5. MB run (wall prescribed, interior moved by the solver's mesh equation)
#    6. post-processing, 7. FB vs MB comparison
#  STOPS at two human GATES: (G1) solver-safe pinned morph, (G2) MRI flow split.
#
#  Usage:  bash run_patient.sh [--dry-run] patients/<case>.env
#    --dry-run   print every command and the generated XMLs; run nothing else
#                (no meshing, no morph, no solver). The XMLs are written to
#                $DRYRUN_XML_DIR (default: a new temporary directory).
#  Resume from a step:  STEP=4 bash run_patient.sh patients/<case>.env
#  (STEP = first step to (re)run: 1=mesh 2=wall motion 3=rcr 4=FB 5=MB 6=post 7=compare)
#
#  The <case>.env file defines the case INPUTS (see patients/TEMPLATE.env). The only
#  non-scripted prerequisite is the set of segmented phase STLs.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --*) echo "unknown option: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done
ENVFILE="${1:?usage: run_patient.sh [--dry-run] patients/<case>.env}"
source "$ENVFILE"
: "${PATIENT:?PATIENT missing in $ENVFILE}"
: "${REF_STL:?}" "${ORIG_SURF:?}" "${PHASES:?}" "${WORK:?}" "${RUNROOT:?}"
: "${T_CYCLE:=0.974}" "${SCALE:=0.1}" "${NPROCS:=4}" "${STEP:=1}"
: "${NSTEPS:=3896}" "${CYC0:=2922}" "${CYC1:=3896}"      # 4 cycles, analysis on the last one
: "${TAPER_L:=12}"                                        # cap taper length (mm)
: "${MORPH_CODE:=${FABLE_CODE:-$REPO/morph}}"             # morph code vendored in the repo
: "${SVMP_BIN:=svmultiphysics}"
PY="${PYTHON:-python3}"

MESH="$WORK/mesh/mesh-complete.mesh.vtu"
SURF="$WORK/mesh/mesh-surfaces"
REG_DIR="$WORK/registration"                  # reg_*.npy from morph/pipeline.py
PIN_DIR="$WORK/pinned_morph"                  # EXTENDED all-node file of the pinned morph
WALL_DISP="$WORK/wall_displacement.txt"       # STANDARD wall-only file given to the solver
FB_DIR="$RUNROOT/FB_${PATIENT}"; MB_DIR="$RUNROOT/MB_${PATIENT}"
FB_XML="$WORK/${PATIENT}_FB.xml";  MB_XML="$WORK/${PATIENT}_MB.xml"
if [ "$DRY" -eq 1 ]; then
  XML_DIR="${DRYRUN_XML_DIR:-$(mktemp -d)}"; mkdir -p "$XML_DIR"
  FB_XML="$XML_DIR/${PATIENT}_FB.xml"; MB_XML="$XML_DIR/${PATIENT}_MB.xml"
else
  mkdir -p "$WORK/mesh" "$FB_DIR" "$MB_DIR"
fi
say(){ echo -e "\n\033[1;36m[$PATIENT] $*\033[0m"; }
gate(){ echo -e "\n\033[1;33m===== GATE: $* =====\033[0m"; }
# run CMD...           : execute, or print it in dry-run mode
# run_log LOG CMD...   : same, output also copied to LOG
run(){ if [ "$DRY" -eq 1 ]; then echo "+ $*"; else "$@"; fi; }
run_log(){ local log="$1"; shift
  if [ "$DRY" -eq 1 ]; then echo "+ $* | tee $log"; else "$@" | tee "$log"; fi; }
[ "$DRY" -eq 1 ] && echo "[dry-run] commands are printed, not executed; XMLs -> $XML_DIR"

# --- 1. REFERENCE MESH (all-tet + surfaces, GID/EID/FaceID) -----------------
if [ "$STEP" -le 1 ]; then
  say "1/7 reference mesh (build_iso_mesh)"
  run "$PY" "$REPO/tools/build_iso_mesh.py" "$REF_STL" "$ORIG_SURF" "$WORK/mesh" "${SURF_HMAX:-0.5}"
fi

# --- 2. WALL MOTION: registration -> pinned + tapered morph -> wall-only file --
if [ "$STEP" -le 2 ]; then
  say "2/7 wall motion: registration, cap-pinned morph (taper ${TAPER_L} mm), wall-only STANDARD file"
  PHASE_ARGS=(); PHASE_T=()
  for p in $PHASES; do PHASE_ARGS+=(--phase "$p"); PHASE_T+=(--phase "${p%%:*}"); done
  run_log "$WORK/registration.log" "$PY" "$MORPH_CODE/pipeline.py" --mesh "$MESH" --out "$REG_DIR" \
        --t-cycle "$T_CYCLE" --scale "$SCALE" --register-only "${PHASE_ARGS[@]}"
  run_log "$WORK/morph.log" "$PY" "$MORPH_CODE/run_pinned_morph.py" --mesh "$MESH" --surf "$SURF" \
        --reg-dir "$REG_DIR" --out "$PIN_DIR" --L "$TAPER_L" \
        --t-cycle "$T_CYCLE" --scale "$SCALE" "${PHASE_T[@]}"
  gate "G1 -- solver-safe pinned morph"
  echo "  Gate = worst inverted element volume vs the solver's mesh-motion critical threshold."
  echo "  A few negligible near-degenerate slivers (|6V| << threshold) are tolerated."
  if [ "$DRY" -eq 0 ]; then
    if grep -q "FOLDS-EXCEED-THRESHOLD" "$WORK/morph.log"; then
      echo -e "\033[1;31m  WARNING: inverted elements EXCEED the solver threshold -- do not start the runs.\033[0m"
      echo "  Remedy: screen the taper length (morph/pin_caps.py sweep), rebuild the reference mesh"
      echo "  (build_iso_mesh removes slivers via mmg3d -optim), or double --n-samples."
      exit 2
    fi
    grep "\[morph-gate\]" "$WORK/morph.log" | tail -1
    echo "  OK: pinned morph is solver-safe."
  fi
  # production prescribes the WALL positions only; the solver's mesh equation moves the interior
  run "$PY" "$REPO/tools/extended_to_standard.py" "$PIN_DIR/displacement_all_nodes.txt" "$WALL_DISP" \
        --face "$SURF/wall.vtp"
fi

# --- 3. RCR CALIBRATION on the MRI flow split -------------------------------
if [ "$STEP" -le 3 ]; then
  say "3/7 RCR calibration (calibrate_rcr) on the MRI flow split"
  run_log "$WORK/rcr.log" "$PY" "$REPO/tools/calibrate_rcr.py" ${RCR_ARGS:-}
  echo "  -> copy the R/C values into the <Add_BC> Neumann entries of the generated XMLs."
fi

# --- XMLs (mesh + wall displacement + number of steps) ----------------------
say "XML: derive from FB_example.xml / MB_example.xml (production settings, caps Dir 0)"
if [ "$DRY" -eq 0 ] && [ ! -f "$WALL_DISP" ]; then
  echo "  WARNING: $WALL_DISP not found yet (run step 2 first)"
fi
"$PY" "$REPO/tools/make_patient_xml.py" \
    --fb-template "$REPO/FB_example.xml" --mb-template "$REPO/MB_example.xml" \
    --mesh "$MESH" --surf "$SURF" --disp "$WALL_DISP" --scale "$SCALE" \
    --nsteps "$NSTEPS" --fb-out "$FB_XML" --mb-out "$MB_XML"
if [ "$DRY" -eq 1 ]; then
  for x in "$FB_XML" "$MB_XML"; do echo; echo "----- $x -----"; cat "$x"; echo; done
fi

# Inlet flow: the XMLs reference it by the RELATIVE name 'inflow.txt' and the solver runs
# from the run directory -> stage the case's flow file (RCR_ARGS --flow) into the FB/MB dirs.
FLOW=$(printf '%s' "${RCR_ARGS:-}" | sed -n 's/.*--flow[ =]\+\([^ ]*\).*/\1/p')
if [ -n "${FLOW:-}" ] && { [ "$DRY" -eq 1 ] || [ -f "$FLOW" ]; }; then
  # SIGN + PERIOD: the solver imposes v = Q*profile*n_out -> an INLET needs Q < 0, and the
  # Fourier period is t_last - t_first of the file. make_inlet_flow.py resamples the (positive,
  # antegrade) MRI waveform over the whole run with the right sign; check_inlet_sign.py refuses
  # a backwards run.
  run "$PY" "$REPO/tools/make_inlet_flow.py" "$FLOW" "$FB_DIR/inflow.txt" --t-cycle "$T_CYCLE" \
      --nsteps "$NSTEPS" ${INLET_FLOW_ARGS:-}
  run cp -f "$FB_DIR/inflow.txt" "$MB_DIR/inflow.txt"
  if [ "$DRY" -eq 1 ]; then
    echo "+ $PY $REPO/tools/check_inlet_sign.py $FB_DIR/inflow.txt --t-cycle $T_CYCLE --nsteps $NSTEPS"
  else
    "$PY" "$REPO/tools/check_inlet_sign.py" "$FB_DIR/inflow.txt" --t-cycle "$T_CYCLE" --nsteps "$NSTEPS" \
      || { echo "  ABORT: inlet flow file would run the aorta backwards"; exit 3; }
    echo "  inlet flow staged -> inflow.txt in FB/MB run dirs (from $(basename "$FLOW"), sign/period verified)"
  fi
else
  echo "  WARNING: no readable --flow file in RCR_ARGS -> the solver will fail to open inflow.txt"
fi

# --- 4. FB RUN (rigid wall) + MRI FLOW-SPLIT GATE ---------------------------
if [ "$STEP" -le 4 ]; then
  say "4/7 FB run (rigid wall)"
  if [ "$DRY" -eq 1 ]; then
    echo "+ (cd $FB_DIR && mpirun -np $NPROCS $SVMP_BIN $FB_XML | tee $FB_DIR/fb.log)"
  else
    ( cd "$FB_DIR" && mpirun -np "$NPROCS" "$SVMP_BIN" "$FB_XML" | tee "$FB_DIR/fb.log" )
  fi
  gate "G2 -- flow split vs MRI (on the FB)"
  run_log "$WORK/split.log" "$PY" "$REPO/tools/extract_flowsplit_FB.py" "$FB_DIR/${NPROCS}-procs" "$SURF" \
      ${OUTLETS:+--outlets $OUTLETS}
  echo "  If any deviation > ~5% vs MRI -> iterate the R values (calibrate_rcr) before the MB."
fi

# --- 5. MB RUN (wall prescribed, interior moved by the mesh equation) -------
if [ "$STEP" -le 5 ]; then
  say "5/7 MB run (wall-only prescribed displacement, caps pinned)"
  if [ "$DRY" -eq 1 ]; then
    echo "+ RUNDIR=$MB_DIR SVMP_NPROCS=$NPROCS SVMP_BIN=$SVMP_BIN bash $REPO/run_MB_aniso.sh $MB_XML | tee $MB_DIR/mb.log"
  else
    RUNDIR="$MB_DIR" SVMP_NPROCS="$NPROCS" SVMP_BIN="$SVMP_BIN" \
      bash "$REPO/run_MB_aniso.sh" "$MB_XML" | tee "$MB_DIR/mb.log"
  fi
  echo "  Check the closed-boundary flux (tools/closed_flux.py) and the CFL at peak systole."
fi

# --- 6. POST-PROCESSING + 7. FB vs MB COMPARISON ----------------------------
if [ "$STEP" -le 6 ]; then
  say "6/7 hemodynamic indices (TAWSS/OSI/helicity) for FB and MB"
  for C in "FB:$FB_DIR" "MB:$MB_DIR"; do
    name="${C%%:*}"; dir="${C##*:}"
    run "$PY" "$REPO/tools/hemo_indices.py" "$dir/${NPROCS}-procs" --wall "$SURF/wall.vtp" \
        --cycle-start "$CYC0" --cycle-end "$CYC1" \
        --out-prefix "$WORK/${name}_${PATIENT}_"
  done
fi
if [ "$STEP" -le 7 ]; then
  say "7/7 FB vs MB COMPARISON"
  run "$PY" "$REPO/tools/compare_FB_MB.py" \
     --fb "$FB_DIR/${NPROCS}-procs" "$SURF/wall.vtp" "$CYC0" "$CYC1" \
     --mb "$MB_DIR/${NPROCS}-procs" "$SURF/wall.vtp" "$CYC0" "$CYC1" \
     --out-prefix "$WORK/cmp_${PATIENT}_"
  if [ "$DRY" -eq 1 ]; then
    echo -e "\n[$PATIENT] dry run done (nothing executed except the XML generation: $XML_DIR)."
  else
    echo -e "\n\033[1;32m[$PATIENT] done. Results in $WORK (cmp_${PATIENT}_FB_vs_MB_wall.vtp).\033[0m"
  fi
fi
