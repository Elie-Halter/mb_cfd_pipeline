#!/usr/bin/env bash
# ============================================================================
#  run_patient.sh — full FB+MB pipeline for ONE patient.
#
#  Chains: reference mesh -> fold-free morph (EXTENDED displacement) -> RCR ->
#          FB run -> MB run -> post-processing -> FB vs MB comparison.
#  STOPS at two human GATES: (G1) fold-free registration, (G2) MRI flow split.
#
#  Usage:  bash run_patient.sh  patients/<patient>.env
#  Resume from a step:  STEP=4 bash run_patient.sh patients/<patient>.env
#  (STEP = first step to (re)run: 1=mesh 2=morph 3=rcr 4=FB 5=MB 6=post 7=compare)
#
#  The <patient>.env file defines the patient-specific INPUTS (see
#  patients/TEMPLATE.env). The only non-scripted prerequisite = the segmented
#  phase STLs.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVFILE="${1:?usage: run_patient.sh patients/<patient>.env}"
source "$ENVFILE"
: "${PATIENT:?PATIENT missing in $ENVFILE}"
: "${REF_STL:?}" "${ORIG_SURF:?}" "${PHASES:?}" "${WORK:?}" "${RUNROOT:?}"
: "${T_CYCLE:=0.974}" "${SCALE:=0.1}" "${NPROCS:=4}" "${STEP:=1}"
: "${FABLE_CODE:=$REPO/morph}"      # morph code vendored in the repo
: "${SVMP_BIN:=svmultiphysics}"

MESH="$WORK/mesh/mesh-complete.mesh.vtu"
SURF="$WORK/mesh/mesh-surfaces"
DISP="$WORK/displacement_${PATIENT}.txt"
FB_DIR="$RUNROOT/FB_${PATIENT}"; MB_DIR="$RUNROOT/MB_${PATIENT}"
FB_XML="$WORK/${PATIENT}_FB.xml";  MB_XML="$WORK/${PATIENT}_MB.xml"
mkdir -p "$WORK/mesh" "$FB_DIR" "$MB_DIR"
say(){ echo -e "\n\033[1;36m[$PATIENT] $*\033[0m"; }
gate(){ echo -e "\n\033[1;33m===== GATE: $* =====\033[0m"; }

# --- 1. REFERENCE MESH (all-tet + surfaces, GID/EID/FaceID) -----------------
if [ "$STEP" -le 1 ]; then
  say "1/7 reference mesh (build_iso_mesh)"
  python3 "$REPO/tools/build_iso_mesh.py" "$REF_STL" "$ORIG_SURF" "$WORK/mesh" "${SURF_HMAX:-0.5}"
fi

# --- 2. FOLD-FREE MORPH -> EXTENDED displacement ----------------------------
if [ "$STEP" -le 2 ]; then
  say "2/7 fold-free registration + rest-shape morph (pipeline.py)"
  PHASE_ARGS=""; for p in $PHASES; do PHASE_ARGS="$PHASE_ARGS --phase $p"; done
  python3 "$FABLE_CODE/pipeline.py" --mesh "$MESH" --out "$WORK" \
        --t-cycle "$T_CYCLE" --scale "$SCALE" $PHASE_ARGS | tee "$WORK/morph.log"
  cp -f "$WORK"/displacement_*.txt "$DISP" 2>/dev/null || true
  gate "G1 — FOLD-FREE registration"
  echo "  Check in $WORK/morph.log: 'folds=0' on ALL phases,"
  echo "  and signed min-volume > 0 after the morph. If folds remain -> widen the lambda schedule"
  echo "  (the pipeline retries on its own; otherwise double --n-samples or tune qmin, see pipeline.py header)."
  if grep -qiE "folds=[1-9]|residual.*fold|J<0|inverted" "$WORK/morph.log"; then
    echo -e "\033[1;31m  WARNING: FOLDS DETECTED — fix before continuing (do not start the runs).\033[0m"; exit 2
  fi
  echo "  OK: no folds detected in the log."
fi

# --- 3. RCR CALIBRATION on the patient's MRI flow split ---------------------
if [ "$STEP" -le 3 ]; then
  say "3/7 RCR calibration (calibrate_rcr) on the patient MRI flow split"
  python3 "$REPO/tools/calibrate_rcr.py" ${RCR_ARGS:-} | tee "$WORK/rcr.log"
  echo "  -> copy the R/C values into the <Add_BC> Neumann entries of the XMLs (FB_example.xml / MB_example.xml)."
fi

# --- 4. GENERATE THE XMLs (mesh + displacement + patient RCR) ---------------
say "XML: derive from the example templates (patient mesh/displacement/RCR)"
python3 "$REPO/tools/make_patient_xml.py" \
    --fb-template "$REPO/FB_example.xml" --mb-template "$REPO/MB_example.xml" \
    --mesh "$MESH" --surf "$SURF" --disp "$DISP" --scale "$SCALE" \
    --nsteps "${NSTEPS:-1948}" --fb-out "$FB_XML" --mb-out "$MB_XML" \
  || { echo "  (make_patient_xml.py missing/needs adapting — edit $FB_XML/$MB_XML by hand from the templates)"; }

# --- 5. FB RUN (rigid wall) -------------------------------------------------
if [ "$STEP" -le 4 ]; then
  say "4/7 FB run (rigid)"
  ( cd "$FB_DIR" && mpirun -np "$NPROCS" "$SVMP_BIN" "$FB_XML" | tee "$FB_DIR/fb.log" )
fi

# --- 6. MRI FLOW-SPLIT GATE on the FB ---------------------------------------
if [ "$STEP" -le 4 ]; then
  gate "G2 — flow split vs MRI (on the FB)"
  python3 "$REPO/tools/extract_flowsplit_FB.py" "$FB_DIR/${NPROCS}-procs" "$SURF" ${OUTLETS:+--outlets $OUTLETS} | tee "$WORK/split.log"
  echo "  If any deviation > ~5% vs MRI -> iterate the R values (calibrate_rcr) before the MB."
fi

# --- 7. MB RUN (moving boundary, GAMG mandatory) ----------------------------
if [ "$STEP" -le 5 ]; then
  say "5/7 MB run (prescribed morph, GAMG via PETSC_OPTIONS)"
  RUNDIR="$MB_DIR" SVMP_NPROCS="$NPROCS" SVMP_BIN="$SVMP_BIN" \
    bash "$REPO/run_MB_aniso.sh" "$MB_XML" | tee "$MB_DIR/mb.log"
  echo "  Watch the CFL at peak systole; if NS diverges -> dt=0.5 ms (x2 wall time)."
fi

# --- 8. POST-PROCESSING + FB vs MB COMPARISON -------------------------------
if [ "$STEP" -le 6 ]; then
  say "6/7 hemodynamic indices (TAWSS/OSI/helicity) for FB and MB"
  for C in "FB:$FB_DIR" "MB:$MB_DIR"; do
    name="${C%%:*}"; dir="${C##*:}"
    python3 "$REPO/tools/hemo_indices.py" "$dir/${NPROCS}-procs" --wall "$SURF/wall.vtp" \
        --cycle-start "${CYC0:-974}" --cycle-end "${CYC1:-1948}" \
        --out-prefix "$WORK/${name}_${PATIENT}_"
  done
fi
if [ "$STEP" -le 7 ]; then
  say "7/7 FB vs MB COMPARISON (the scientific deliverable)"
  python3 "$REPO/tools/compare_FB_MB.py" \
     --fb "$FB_DIR/${NPROCS}-procs" "$SURF/wall.vtp" "${CYC0:-974}" "${CYC1:-1948}" \
     --mb "$MB_DIR/${NPROCS}-procs" "$SURF/wall.vtp" "${CYC0:-974}" "${CYC1:-1948}" \
     --out-prefix "$WORK/cmp_${PATIENT}_"
  echo -e "\n\033[1;32m[$PATIENT] done. Results in $WORK (cmp_${PATIENT}_FB_vs_MB_wall.vtp).\033[0m"
fi
