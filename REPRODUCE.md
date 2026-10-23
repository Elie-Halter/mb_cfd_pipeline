# Reproduce

From a clean machine to a moving-boundary (MB) and a rigid-wall (FB) run of one aorta, with the
checks that tell you the runs are sound. Tested on Ubuntu 22.04 / 24.04. The method itself is
summarised in [README.md](README.md); the solver conventions that silently break a run are in
[CONVENTIONS.md](CONVENTIONS.md) — read it first.

**What production does.** The wall motion comes from the images and is prescribed on the wall
nodes only (`Prescribed_displacement`, STANDARD file). The caps are pinned (`Dir 0` in the mesh
equation) and the wall motion is tapered to zero over 12 mm from the cap rings. The interior of
the mesh is moved by the solver's own mesh equation (incremental linear elasticity) at every time
step. The offline rest-shape morph of `morph/` is a validity check — it shows that the imposed wall
motion admits a valid interior mesh — and only its wall positions are used. FB and MB share the
same mesh, so differences are taken node to node.

## 0. Requirements

Linux, MPI (OpenMPI), CMake, a C++/Fortran tool chain, Python 3.10+, ~8 GB RAM, 30–60 min of
build time. Per case: the segmented lumen of one or several cardiac phases, the inlet flow
waveform and the outlet flow split from 4D-flow MRI, and a mean arterial pressure.

## 1. Install

```bash
git clone https://github.com/Elie-Halter/mb_cfd_pipeline.git && cd mb_cfd_pipeline
bash install.sh && source ~/.bashrc && bash check_install.sh
bash tests/run_tests.sh          # no solver, no data: every tool on synthetic cases
```

`install.sh` builds MMG, clones svMultiPhysics at the pinned commit `97ef512`, applies the
fifteen patches of `patches/svMP/` and installs the solver as `~/svmp_bin/svmultiphysics`. There is
no fork to clone. PETSc (`libpetsc-real-dev`) is needed if the mesh equation uses the GAMG
preconditioner.

## 2. Verify the solver on the analytic benchmarks

```bash
bash benchmarks/run_all.sh --no-solver     # generate the cases, run every pre-flight check
bash benchmarks/run_all.sh                 # free stream, expanding tube, Womersley (~2-3 h, 4 cores)
```

Pass criteria and reference values are in [benchmarks/README.md](benchmarks/README.md). The
discriminating result is the expanding tube's discrete mass balance: 3.3e-6 with patch 0014
against 6.66e-2 without it (`benchmarks/build_nogcl.sh` builds the contrast solver,
`SVMP_BIN_NOGCL=... bash benchmarks/run_all.sh --cases "B1 B2"` runs the contrast).

## 3. Reference mesh and boundary data

```bash
python3 tools/cohort_cap_label.py lumen_open.stl ref/                 # caps + cap surfaces (or your own)
python3 tools/build_iso_mesh.py ref/capped.stl ref/mesh-surfaces mesh/ --rbm 6 --hmin 0.2 --hmax 0.8
python3 tools/make_inlet_flow.py mri_inlet.csv inflow.txt --t-cycle T --nsteps N   # NEGATIVE inflow
python3 tools/check_inlet_sign.py inflow.txt --t-cycle T --nsteps N
python3 tools/calibrate_rcr.py --flow mri_inlet.csv --map MAP          # RCR per outlet on the MRI split
```

The mesh is all-tetrahedral (the solver reads one element type), in mm, with GlobalNodeID /
GlobalElementID / ModelFaceID. The inlet flow rate must be **negative**: the solver imposes
`v = Q · profile · n_out`, so a positive Q is an outflow (CONVENTIONS.md §1).
`make_inlet_flow.py` also resamples the waveform over the whole run so that the file's period
(t_last − t_first) cannot differ from the cycle (§2).

## 4. Wall motion — two chains

Both chains end with a wall-motion file in the solver's STANDARD format (positions in cm of the
wall nodes, mapped by GlobalNodeID) and use the same cap pinning and 12 mm geodesic smoothstep taper.

### A. Several segmented phases — surface registration (`morph/`)

```bash
python3 morph/pipeline.py --mesh mesh/mesh-complete.mesh.vtu --out work/ --register-only \
    --phase 0.40:phaseA.stl --phase 0.60:phaseB.stl --phase 0.80:phaseC.stl --t-cycle T
python3 morph/run_pinned_morph.py --mesh mesh/mesh-complete.mesh.vtu --surf mesh/mesh-surfaces \
    --reg-dir work/ --out work_pinned/ --L 12 --phase 0.40 --phase 0.60 --phase 0.80 --t-cycle T
python3 tools/extended_to_standard.py work_pinned/displacement_all_nodes.txt wall_displacement.txt \
    --face mesh/mesh-surfaces/wall.vtp
```

`pipeline.py` registers the reference surface onto every segmented phase (Laplacian N-ICP with
normal shooting, fold-free and iso-topological); without `--register-only` it also runs an
unpinned offline morph, which production does not use. `run_pinned_morph.py` reuses that
registration with pinned caps and the 12 mm taper and runs the offline rest-shape morph over the
cycle (gate G1: inverted elements must stay below the solver's critical volume;
`morph/pin_caps.py sweep` screens the taper length). It writes an EXTENDED all-node file;
`extended_to_standard.py` keeps the wall positions only. See [morph/README.md](morph/README.md).

### B. One segmented phase — image registration (`chain_image/`)

```bash
python3 chain_image/register_phases.py --phases 'phases/phase*.nii.gz' --ref-index R \
    --ref-seg seg_ref.nrrd --out reg/
python3 chain_image/invert_propagate.py --reg-dir reg/ --wall-vtp mesh/mesh-surfaces/wall.vtp --out wall_raw.npz
python3 chain_image/smooth_wall_field.py --in wall_raw.npz --wall-vtp mesh/mesh-surfaces/wall.vtp \
    --period T --out wall_smooth.npz
python3 chain_image/taper_wall_field.py --in wall_smooth.npz --mesh mesh/mesh-complete.mesh.vtu \
    --surfaces mesh/mesh-surfaces --caps asc desc btca lcca lsa --length 12 --out wall_taper.npz
python3 chain_image/write_wall_displacement.py --in wall_taper.npz --mesh mesh/mesh-complete.mesh.vtu \
    --period T --dt 0.001 --cycles 3 --preroll 0.15 --out wall_displacement.txt \
    --flow-in mri_inlet.csv --flow-out inflow.txt
```

The mesh is built on the segmented (reference) phase; elastix B-spline registration maps every
other phase onto it, and the transforms are inverted at the mesh's wall nodes to give their
positions at every phase. The simulation clock starts at the reference phase, so the inflow is
written on the same clock (`--flow-in`, negative). See [chain_image/README.md](chain_image/README.md).

### Wall-motion utilities

- `tools/phase_subset_wall.py` — rebuild the motion from a subset of the phases (how many
  segmented phases does a run need?);
- `tools/retime_wall.py` — shift the wall motion against the inflow;
- `tools/remove_rigid_motion.py` — remove the rigid-body part of the motion;
- `tools/build_fb_at_phase.py` — a rigid mesh frozen at any phase of the motion.

## 5. Solver set-up and runs

Start from `FB_example.xml` / `MB_example.xml`: they are the production inputs with relative
paths (`mesh/...`, `inflow.txt`, `wall_displacement.txt`) and example RCR values;
`tools/make_patient_xml.py` repoints the paths and refuses an MB file that deviates from the
table below or an EXTENDED displacement file. The MB case used in production has:

| Equation | Face | Boundary condition |
|---|---|---|
| fluid | inlet | `Dirichlet`, `Unsteady`, flow-rate file (negative), `Impose_flux true` |
| fluid | outlets | `Neumann`, `RCR` (from `calibrate_rcr.py`) |
| fluid | wall | `Prescribed_displacement` + `Impose_on_state_variable_integral true` (fluid velocity = wall velocity; **not** `Dirichlet 0`, CONVENTIONS.md §3) |
| mesh | wall | `Prescribed_displacement`, the STANDARD wall file of §4 |
| mesh | every cap | `Dir`, value 0 (pinned) |

The mesh equation (`Mesh_stiffness_exponent 2`) moves the interior. Its linear solver was FSILS
for one production case and PETSc with the GAMG preconditioner for the other; GAMG is selected
through `PETSC_OPTIONS="-pc_type gamg -pc_gamg_type agg"`, which `run_MB_aniso.sh` and
`tools/svmp_run.sh` export. The FB case uses the same mesh with a rigid no-slip wall.

```bash
tools/svmp_run.sh FB.xml runs/FB fb 4       # one solver run at a time on a machine
tools/svmp_run.sh MB.xml runs/MB mb 4
```

`run_patient.sh patients/<case>.env` automates chain A in the production configuration
(registration, pinned morph with the 12 mm taper, wall-only STANDARD file), the XML generation
from the examples, the inflow file (with the sign check), the FB and MB runs and the
post-processing for one case. `bash run_patient.sh --dry-run patients/<case>.env` prints every
command and the generated XMLs without running anything else.

## 6. Check the runs

```bash
python3 tools/closed_flux.py runs/MB/4-procs mesh/mesh-surfaces S0 S1        # closed-boundary flux sum
python3 tools/extract_flowsplit_FB.py runs/FB/4-procs mesh/mesh-surfaces      # flow split vs MRI
python3 tools/mass_balance.py --vtu-glob 'runs/MB/4-procs/results_*.vtu' --surf-dir mesh/mesh-surfaces --dt 0.001
python3 tools/volume_cycle.py --vtu-glob 'runs/MB/4-procs/results_*.vtu' --dt 0.001 --period T --cycle 3
```

- the closed-boundary flux sum (caps + wall, P1 exact on the deformed geometry) is near zero at
  every saved step — below 0.01 % of the peak inflow on the paper's runs;
- the inlet flux is negative at systole (otherwise the run is backwards);
- the outlet flow split is within a few percent of the MRI targets;
- the minimum element quality and the inverted-element count stay safe over the cycle.

## 7. Post-process and compare

```bash
python3 tools/hemo_indices.py runs/FB/4-procs --wall mesh/mesh-surfaces/wall.vtp --cycle-start S0 --cycle-end S1
python3 tools/hemo_indices.py runs/MB/4-procs --wall mesh/mesh-surfaces/wall.vtp --cycle-start S0 --cycle-end S1
python3 tools/compare_FB_MB.py --fb runs/FB/4-procs mesh/mesh-surfaces/wall.vtp S0 S1 \
                               --mb runs/MB/4-procs mesh/mesh-surfaces/wall.vtp S0 S1
python3 tools/volume_weighted_means.py runs/MB/4-procs/results_*.vtu --csv means.csv
python3 tools/gci.py --mesh M1.vtu M2.vtu M3.vtu --wall M1_wall.vtp M2_wall.vtp M3_wall.vtp
```

Wall-to-volume mapping uses GlobalNodeID − 1 (the result files keep the mesh order), never a
positional nearest-neighbour search, which is wrong on a moving boundary. Result files store the
deformed points in mm and the velocity in cm/s. All tools are listed in
[tools/README.md](tools/README.md); the solver patches and their role in
[patches/svMP/README.md](patches/svMP/README.md).
