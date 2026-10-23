# tools/ — Utility scripts

Every script prints its full usage with `--help` (or in its docstring). Grouped by pipeline stage.

## Meshing

| Script | Role | Usage |
|--------|------|-------|
| `cohort_cap_label.py` | Caps and labels the openings of a raw open lumen surface: closed STL + reference cap surfaces `asc/desc/btca/lcca/lsa` (3 smallest openings = branches, larger of the 2 largest = inlet, branches named by their order along the arch); QC render | `python3 cohort_cap_label.py RAW.(stl\|vtp) OUT_DIR` |
| `build_iso_mesh.py` | Reference STL → all-tet mesh + surfaces (GID/EID/FaceID). Uniform, or **radius-based** (`--rbm N`: ~N elements across the local diameter → finer small branches, no SimVascular) | `python3 build_iso_mesh.py <stl> <surfaces> <out> [hmax] [--rbm 6 --hmin 0.2 --hmax 0.8]` |
| `build_fb_at_phase.py` | Rigid-wall mesh frozen at any phase of a prescribed motion (wall-field `.npz` or EXTENDED file), same numbering and connectivity as the moving mesh; `--scan` lists the lumen volume per phase | `python3 build_fb_at_phase.py --mesh ref.vtu --surfaces mesh-surfaces --wall-npz wall.npz --ref-index R --phase K --out-dir out/` |

## Inflow and boundary conditions

| Script | Role | Usage |
|--------|------|-------|
| `make_inlet_flow.py` | Inlet flow file with the solver's sign and period: antegrade-positive MRI waveform → **negative** flow rate (positive = outflow, see CONVENTIONS.md §1), resampled at the time step over the whole run (exact period) | `python3 make_inlet_flow.py mri_inlet.csv flow_rate.txt --t-cycle T --nsteps N [--dt 0.001]` |
| `check_inlet_sign.py` | Pre-flight: refuses an inlet file that would impose an outflow (exit 2) or whose period differs from the cycle (exit 3) | `python3 check_inlet_sign.py flow_rate.txt --t-cycle T [--nsteps N]` |
| `calibrate_rcr.py` | RCR Windkessel calibration on the MRI flow split (Opt 3) | `python3 calibrate_rcr.py --flow <inlet.txt> --map <MAP>` |
| `make_patient_xml.py` | Derives a patient's FB/MB XML files from the templates (paths repointed, comments kept) | `python3 make_patient_xml.py --fb-template … --mb-template … --mesh … --surf … --disp … --fb-out … --mb-out …` |

## Wall motion

The two wall chains themselves are in `morph/` and `chain_image/`.

| Script | Role | Usage |
|--------|------|-------|
| `extended_to_standard.py` | EXTENDED all-node morph file (`morph/`) → STANDARD wall-motion file used in production (the solver then moves the interior with its mesh equation); `--face` keeps only that face's nodes | `python3 extended_to_standard.py disp_all_nodes.txt disp_wall.txt [--face wall.vtp]` |
| `phase_subset_wall.py` | Rebuilds the wall motion from a chosen **subset of phases** (periodic PCHIP through the kept keyframes, resampled at every phase) and reports the error against the full field — how many segmented phases does a run need? | `python3 phase_subset_wall.py wall.npz --keep 0 4 8 --out subset.npz [--ref-index R]` |
| `retime_wall.py` | Shifts a prescribed wall-displacement file in time (re-times the wall against the inflow); the initial mesh must then be the shape of the new sample 0 | `python3 retime_wall.py disp_in.txt disp_out.txt --shift K` |
| `remove_rigid_motion.py` | Removes the rigid-body part (Kabsch, blended to zero at the caps) of a prescribed wall motion while keeping its lumen-volume curve — separates bulk displacement from wall deformation | `python3 remove_rigid_motion.py disp_in.txt disp_out.txt --mesh mesh.vtu --faces mesh-surfaces --caps asc desc btca lcca lsa` |

## Running

| Script | Role | Usage |
|--------|------|-------|
| `svmp_run.sh` | Guarded solver launcher: refuses to start while another solver runs, purges a stale `STOP_SIM`, sets `LD_LIBRARY_PATH` and `PETSC_OPTIONS` (GAMG), starts in a detached tmux session. Binary: `$SVMP_BIN` (default `~/svmp_bin/svmultiphysics`) | `tools/svmp_run.sh <config.xml> <rundir> [session] [nprocs]` |
| `run_provenance.py` | Which configuration produced a run, and is a restart faithful: `compare` two VTUs (round-off vs different), `fingerprint` a run, list `candidates` XMLs compatible with a run's fingerprint | `python3 run_provenance.py compare A.vtu B.vtu` |

## Mass conservation and run checks

| Script | Role | Usage |
|--------|------|-------|
| `closed_flux.py` | **Primary mass check**: closed-boundary flux sum (caps + wall, P1 exact on the deformed geometry, outward normals oriented geometrically) at every saved step, plus the signed cap fluxes (inlet must be negative at systole) | `python3 closed_flux.py <run>/4-procs <mesh-surfaces> <s0> <s1> [--caps …] [--inlet asc]` |
| `mass_balance.py` | Lab-frame balance Q_in − ΣQ_out − dV/dt with the inlet flux measured from the field (fixed caps) | `python3 mass_balance.py --vtu-glob '<run>/4-procs/results_*.vtu' --surf-dir <mesh-surfaces> --dt 0.001 [--cycle-range s0,s1]` |
| `mass_balance_ale.py` | Same with ALE (mesh-relative) cap fluxes, for runs whose caps move | `python3 mass_balance_ale.py --vtu-glob '<run>/4-procs/results_*.vtu' --surf-dir <mesh-surfaces> --dt 0.001 --cycle-range s0,s1` |
| `volume_cycle.py` | Lumen volume against cycle phase from a moving-boundary run; locates a rigid mesh's volume inside the cycle | `python3 volume_cycle.py --vtu-glob 'MB/4-procs/results_*.vtu' --dt 1e-3 --period T --cycle 3 [--reference-vtu FB.vtu]` |
| `extract_flowsplit_FB.py` | Outlet flow split of a run (per-triangle integration, GID mapping, deformed cap areas), compared with the MRI targets | `python3 extract_flowsplit_FB.py <run>/4-procs <mesh>/mesh-surfaces [--outlets desc,btca,lcca,lsa]` |

## Post-processing and FB/MB comparison

| Script | Role | Usage |
|--------|------|-------|
| `checkpoints.py` | Measurements at fixed planes along the aorta: diameter, flow rate, velocity | `python3 checkpoints.py <vtu\|dir> checkpoints.json [--out m.csv]` ; `--make-template <centerline.vtp> N` |
| `hemo_indices.py` | TAWSS / OSI / Helicity over the last cycle (**GID-1** mapping, deformation-proof; wall VTP + volume VTU) | `python3 hemo_indices.py <dir> --wall wall.vtp --cycle-start <start> --cycle-end <end>` |
| `volume_weighted_means.py` | Lumen-mean speed, helicity density, LNH (signed and absolute) and strongly helical volume shares, **volume-weighted** (nodal weight = ¼ of the adjacent tetrahedra volumes) next to plain nodal means | `python3 volume_weighted_means.py results_*.vtu [--csv out.csv]` |
| `compare_FB_MB.py` | **FB vs MB comparison** (ΔTAWSS/ΔOSI, spatial correlation) + wall VTP for figures | `python3 compare_FB_MB.py --fb <dir> wall.vtp <start> <end> --mb <dir> wall.vtp <start> <end>` |
| `gci.py` | **Grid Convergence Index** (Celik 2008) on ≥3 grids — mesh-independence verification | `python3 gci.py --mesh M1.vtu M2.vtu M3.vtu --wall M1_*.vtp M2_*.vtp M3_*.vtp` |
| `make_figures.py` | Figures from the comparison VTP (headless matplotlib hist/scatter + 3D wall maps); also serves the segmentation sensitivity study | `python3 make_figures.py --wall-vtp cmp_*_FB_vs_MB_wall.vtp --out-dir figs/` |
| `make_movies.py` | **Automatic visualization** (no ParaView, PyVista headless + ffmpeg): MOTION, VELOCITY, HELICITY, FB-vs-MB side-by-side; mp4 + gif + hero PNG per type, fixed color scales over the cycle. See `--help` for all options | `python3 make_movies.py <run_dir> --wall wall.vtp [--stl-dir stl/] [--cycle-start S0 --cycle-end S1] [--fb-dir <FB run>] [--types motion,velocity,helicity,fbmb]` |
| `run_movies.sh` | End-of-run hook wrapping `make_movies.py` — append to a chain script after the last VTU is written | `tools/run_movies.sh <run_dir> --wall wall.vtp --stl-dir stl/ --cycle-start S0 --cycle-end S1 --dt 0.001 --period T` |

## Units
Mesh coordinates in mm; solver works in cm (`Mesh_scale_factor 0.1`). Velocity cm/s, pressure
dyne/cm², flow rate mL/s. Flow split = per-triangle integration `Q = Σ(v_tri·area_vec)/100`.
Result VTUs store the deformed points in mm and the velocity in cm/s. Wall→volume mapping uses
GlobalNodeID−1 (the results VTU preserves mesh order), not a positional KDTree — required for the
moving boundary. Inlet flow rates are negative (solver convention, CONVENTIONS.md §1).
