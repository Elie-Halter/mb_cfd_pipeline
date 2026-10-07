# mb_cfd_pipeline

Patient-specific moving-boundary (MB) versus rigid-wall (FB) CFD of the thoracic aorta, driven by
multi-phase 4D-flow MRI segmentations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built on [svMultiPhysics](https://github.com/SimVascular/svMultiPhysics) and the
[SimVascular](https://simvascular.github.io/) ecosystem. The solver reads a single element type, so
the mesh is all-tetrahedral — there is no prismatic boundary layer.

Wall motion is **prescribed, not solved**: every node position is computed offline and handed to the
solver, so a run involves no mesh-motion PDE, no remeshing and no solution interpolation. Three
stages:

1. Non-rigid registration of the reference surface onto each phase (N-ICP + normal shooting) — a
   fold-free, iso-topological boundary correspondence.
2. Rest-shape variational volume morph (Escobar mean-ratio energy with a positive-Jacobian barrier) —
   the interior follows the wall and every element stays valid.
3. Whole-domain prescription in the solver's `EXTENDED` format.

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

This chains reference mesh → morph → RCR calibration → FB run → MB run → post-processing → FB/MB
comparison. Two gates stop the run: G1 if the registration folds, G2 if the outlet flow split
deviates from MRI. The only input that is not code is the set of segmented phase STLs.

Individual stages are in [REPRODUCE.md](REPRODUCE.md).

## Before your first run

svMultiPhysics has a few conventions that silently produce plausible-looking but wrong results if you
get them wrong — the sign of the inlet flow rate, the period of temporal files, the fluid boundary
condition on a prescribed-motion wall, and cap pinning. They are in
[CONVENTIONS.md](CONVENTIONS.md), together with the checks that tell you a run is sound.
Read it once.

## Layout

```
morph/     prescribed-morph engine (the method) — see morph/README.md
tools/     meshing, RCR calibration, post-processing, FB/MB comparison — see tools/README.md
patches/   fifteen patches on svMultiPhysics @97ef512 — see patches/svMP/README.md
tests/     synthetic tests, no patient data required
```

## Patient data

Source code only. Segmentations, meshes, displacement fields and results are git-ignored and are not
distributed here.

## Citation

See [CITATION.cff](CITATION.cff). Journal reference to be added on publication.

## License

MIT, see [LICENSE](LICENSE). Dependencies — svMultiPhysics, SimVascular, MMG, TetGen, PETSc — keep
their own.

Developed by Elie Halter, supervised by Dr Monika Colombo, Aarhus University, Department of
Mechanical and Production Engineering.
