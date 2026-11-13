# mb_cfd_pipeline

Patient-specific moving-boundary (MB) versus rigid-wall (FB) CFD of the thoracic aorta, driven by
4D-flow MRI.

[![tests](https://github.com/Elie-Halter/mb_cfd_pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/Elie-Halter/mb_cfd_pipeline/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built on the **official, unmodified [svMultiPhysics](https://github.com/SimVascular/svMultiPhysics)**
(commit `86729244`) and open-source tools (3D Slicer, Elastix, Gmsh, VTK/PyVista). The solver reads a
single element type, so the mesh is all-tetrahedral — there is no prismatic boundary layer.

The **wall** motion is prescribed from the images; the **interior** of the volume mesh is moved by the
solver's mesh equation (linear elasticity) at every time step, on a fixed all-tetrahedral mesh — no
remeshing and no solution interpolation. Main stages:

1. Wall trajectory over the cycle: one segmented phase is propagated to every MRI phase by
   B-spline image registration (Elastix), and the transformation is inverted node by node so that
   all phases share the nodes of the reference mesh (`chain_image/`); periodic shape-preserving
   (PCHIP) interpolation in time. A second chain, surface registration onto several segmented
   phases (`morph/`), is kept as an option.
2. Cap taper: the caps are held fixed and the motion is faded in over 12 mm along the wall
   (smoothstep), with the taper length checked offline by counting inverted elements.
3. Solver run: FSI equation with a fluid domain only plus the mesh equation (arbitrary
   Lagrangian–Eulerian), wall displacement imposed as a Dirichlet condition from a
   time-and-space file. The pipeline writes its moving-wall input in a simple authoring format
   (`MB_example.xml`, a `Prescribed_displacement` wall); `tools/to_official_solver.py` converts it
   into the official solver's input, and `run_patient.sh` does it automatically.

The moving-mesh set-up is verified on synthetic analytic benchmarks that need no patient data
(`benchmarks/`): free-stream preservation, an analytically expanding tube (moving no-slip and the
discrete mass balance through the closed boundary) and Womersley flow.

## Install

Ubuntu 22.04 / 24.04:

```bash
git clone https://github.com/Elie-Halter/mb_cfd_pipeline.git && cd mb_cfd_pipeline
sudo apt install -y libpetsc-real-dev
bash install.sh && source ~/.bashrc && bash check_install.sh
```

`install.sh` builds MMG, clones svMultiPhysics at commit `86729244` and builds it unmodified as
`~/svmp_bin/svmultiphysics`, and installs the Python requirements (Python 3.10+). PETSc is required:
the mesh equation uses the GAMG preconditioner, which `run_MB_aniso.sh` selects through
`PETSC_OPTIONS`.

## Run

```bash
cp patients/TEMPLATE.env patients/P0001.env   # segmented phases, MRI flow split, paths
bash run_patient.sh patients/P0001.env
```

This chains reference mesh → wall motion with the 12 mm taper → wall-only displacement file → RCR
calibration → FB run → conversion to the official MB input → MB run → post-processing → FB/MB
comparison, in the production configuration of `FB_example.xml` / `MB_example.xml`. Gates stop the
run if the geometry does not sit on the image (G0), if the cap-pinned wall motion folds the mesh
(G1) or if the outlet flow split deviates from MRI (G2). `--dry-run` prints the commands and the
generated XMLs without running them. Step 2 of `run_patient.sh` uses the surface chain (`morph/`);
for the image chain used in the paper, run `chain_image/` stage by stage (see its README), then
`STEP=3 bash run_patient.sh patients/P0001.env`.

The official solver writes its results on the reference coordinates, with the mesh motion in the
`Displacement` array; `tools/deform_vtu.py` writes the deformed meshes for geometric
post-processing (volumes, sections).

Individual stages and the benchmarks are in [REPRODUCE.md](REPRODUCE.md).

## Verify an installation

```bash
bash tests/run_tests.sh                      # no solver, no data: every tool on synthetic cases
bash benchmarks/run_all.sh --no-solver       # generate the analytic benchmarks, convert, pre-flight checks
bash benchmarks/run_all.sh                   # run them with the official solver (~2-3 h on 4 cores)
```

## Verification

Synthetic analytic benchmarks, official svMultiPhysics `86729244` (details, pass criteria and grids
in [benchmarks/README.md](benchmarks/README.md)):

| Case | Quantity | Pass criterion | Result |
|---|---|---|---|
| B1 free stream on a moving mesh | max\|u−U\|/U | < 1e−4 | 1.69e−6 |
| B2 expanding tube | moving no-slip error | < 1e−4 cm/s | 2.68e−6 cm/s |
| B2 expanding tube | induced flow vs −dV/dt | < 2 % | 1.64 % |
| B2 expanding tube | discrete mass balance (closed boundary) | < 2 % | **3.3e−6** |
| B2t (`--full`) | temporal order of the mass residual | p ≥ 2 | 2.94, then 5.2 |
| B3 Womersley, α ≈ 12.9 | flow amplitude, phase (grid g1) | ±5 %, < 5° | 0.978, +2.6° |

## Reproducing the paper

Companion paper: Halter E, Colombo M, et al. *An open-source pipeline for moving boundary aortic
hemodynamics from low-resolution 4D MRI data* (submitted).

- **From this repository alone:** the analytic benchmarks (`bash benchmarks/run_all.sh`), the solver
  (`install.sh`) and its production inputs (`FB_example.xml`, `MB_example.xml` + `tools/to_official_solver.py`),
  the Gmsh mesh generator (`tools/gen_mesh_gmsh.py`), and every tool of the method and of the analysis
  (`chain_image/`, `morph/`, `tools/`), exercised on synthetic cases by `bash tests/run_tests.sh`.
- **Simulation inputs and post-processed results of the two cases** (meshes, solver inputs, wall
  motion, fields at the analysed instants, data behind the figures): in the data set deposited with
  the paper.
- **Not released:** MRI images, segmentations and clinical data.

## Before your first run

svMultiPhysics has a few conventions that silently produce plausible-looking but wrong results if you
get them wrong — the sign of the inlet flow rate, the period of temporal files, the fluid boundary
condition on a moving wall, and cap pinning. They are in [CONVENTIONS.md](CONVENTIONS.md), together
with the checks that tell you a run is sound. Before meshing, check that the lumen surface sits on the
image it was segmented on, at the right size: `tools/check_geometry_vs_image.py` (gate G0 of
`run_patient.sh` when `REF_IMAGE` is set) -- a surface rescaled or mirrored on its way to the mesher
keeps its shape and fails nothing else.

## Layout

```
chain_image/  wall motion: one segmented phase propagated by inverted image registration (Elastix),
              smoothing, cap taper, displacement file — see chain_image/README.md
morph/        optional wall chain: surface registration onto several segmented phases, cap pinning,
              offline rest-shape morph (validity check) — see morph/README.md
tools/        meshing (Gmsh; MMG/TetGen alternative), inflow file, RCR calibration, conversion to the
              official solver input, wall-motion utilities, mass checks, post-processing, FB/MB
              comparison — see tools/README.md
benchmarks/   synthetic analytic benchmarks (free stream, expanding tube, Womersley) — see benchmarks/README.md
tests/        synthetic tests, no patient data and no solver required
patients/     per-patient configuration template for run_patient.sh
legacy/       solver patches of versions <= 0.3, no longer used — see legacy/README.md
```

## Citation

See [CITATION.cff](CITATION.cff) and [CHANGELOG.md](CHANGELOG.md); journal reference to be added on
publication.

## License

MIT, see [LICENSE](LICENSE). Dependencies — svMultiPhysics, Elastix, Gmsh, MMG, TetGen, PETSc — keep
their own.

Developed by Elie Halter, supervised by Dr Monika Colombo, Aarhus University, Department of
Mechanical and Production Engineering.
