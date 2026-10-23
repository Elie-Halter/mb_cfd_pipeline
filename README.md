# mb_cfd_pipeline

Patient-specific moving-boundary (MB) versus rigid-wall (FB) CFD of the thoracic aorta, driven by
multi-phase 4D-flow MRI segmentations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built on [svMultiPhysics](https://github.com/SimVascular/svMultiPhysics) and the
[SimVascular](https://simvascular.github.io/) ecosystem. The solver reads a single element type, so
the mesh is all-tetrahedral — there is no prismatic boundary layer.

The **wall** motion is prescribed from the images; the **interior** of the volume mesh is moved by the
solver's mesh equation (incremental linear elasticity) at every time step, on a fixed all-tetrahedral
mesh — no remeshing and no solution interpolation. Main stages:

1. Wall trajectory over the cycle, by one of two chains: non-rigid registration of a reference
   surface onto each of several segmented phases (N-ICP + normal shooting, fold-free and
   iso-topological; `morph/`), or propagation of a single segmented phase to every phase by
   inverted B-spline image registration (`chain_image/`); periodic shape-preserving (PCHIP)
   interpolation in time.
2. Cap taper: the caps are held fixed and the motion is faded in over 12 mm along the wall
   (smoothstep), with the taper length checked offline by counting inverted elements.
3. Solver run with the wall imposed through `Prescribed_displacement` (STANDARD mode), the interior
   moved by the mesh equation, and the fluid assembled on the deformed geometry (discrete geometric
   conservation law fix, patch 0014).

The solver changes are verified on synthetic analytic benchmarks that need no patient data
(`benchmarks/`): free-stream preservation, an analytically expanding tube (moving no-slip and the
discrete mass balance with and without patch 0014) and Womersley flow.

An offline rest-shape variational morph (Escobar mean-ratio energy with a positive-Jacobian barrier,
`morph/`) is kept as a validity check that the imposed wall admits a valid interior mesh; the
production runs take only the wall positions from the displacement file.

## Install

Ubuntu 22.04 / 24.04:

```bash
git clone https://github.com/Elie-Halter/mb_cfd_pipeline.git && cd mb_cfd_pipeline
bash install.sh && source ~/.bashrc && bash check_install.sh
sudo apt install -y libpetsc-real-dev
```

`install.sh` builds MMG, applies the fifteen solver patches and installs the Python requirements
(Python 3.10+); `check_install.sh` verifies the result. PETSc is required: the mesh-motion equation needs GAMG, which
`run_MB_aniso.sh` exports for you.

## Run

```bash
cp patients/TEMPLATE.env patients/P0001.env   # phase STLs, MRI flow split, paths
bash run_patient.sh patients/P0001.env
```

This chains reference mesh → registration → cap-pinned morph with the 12 mm taper → wall-only
displacement file → RCR calibration → FB run → MB run → post-processing → FB/MB comparison, in the
production configuration of `FB_example.xml` / `MB_example.xml`. Two gates stop the run: G1 if the
pinned morph folds, G2 if the outlet flow split deviates from MRI. `--dry-run` prints the commands
and the generated XMLs without running them. The only input that is not code is the set of
segmented phase STLs.

Individual stages, the second wall chain and the benchmarks are in [REPRODUCE.md](REPRODUCE.md).

## Verify an installation

```bash
bash tests/run_tests.sh                      # no solver, no data: every tool on synthetic cases
bash benchmarks/run_all.sh --no-solver       # generate the analytic benchmarks + pre-flight checks
bash benchmarks/run_all.sh                   # run them with the patched solver (~2-3 h on 4 cores)
```

## Before your first run

svMultiPhysics has a few conventions that silently produce plausible-looking but wrong results if you
get them wrong — the sign of the inlet flow rate, the period of temporal files, the fluid boundary
condition on a prescribed-motion wall, and cap pinning. They are in
[CONVENTIONS.md](CONVENTIONS.md), together with the checks that tell you a run is sound.
Read it once.

## Layout

```
morph/        wall chain 1: surface registration onto several segmented phases, cap pinning,
              offline rest-shape morph (validity check) — see morph/README.md
chain_image/  wall chain 2: one segmented phase propagated by inverted image registration,
              smoothing, cap taper, displacement file — see chain_image/README.md
tools/        meshing, inflow file, RCR calibration, wall-motion utilities, mass checks,
              post-processing, FB/MB comparison — see tools/README.md
benchmarks/   synthetic analytic benchmarks (free stream, expanding tube / DGCL, Womersley)
              — see benchmarks/README.md
patches/      fifteen patches on svMultiPhysics @97ef512 — see patches/svMP/README.md
tests/        synthetic tests, no patient data and no solver required
patients/     per-patient configuration template for run_patient.sh
```

## Patient data

Source code only. Images, segmentations, meshes, displacement fields and results are git-ignored and
are not distributed here. The benchmarks generate their own synthetic meshes.

## Citation

See [CITATION.cff](CITATION.cff). Companion paper: *Moving versus rigid wall in aortic CFD: diastolic
flow in a healthy and a Turner-syndrome aorta* (in preparation); journal reference to be added on publication.

## License

MIT, see [LICENSE](LICENSE). Dependencies — svMultiPhysics, SimVascular, MMG, TetGen, PETSc — keep
their own.

Developed by Elie Halter, supervised by Dr Monika Colombo, Aarhus University, Department of
Mechanical and Production Engineering.
