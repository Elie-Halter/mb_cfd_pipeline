# MB-CFD — all-tetrahedral moving-boundary aortic hemodynamics

**An open-source pipeline for patient-specific moving-boundary (MB) vs fixed-boundary (FB) CFD of the thoracic aorta, driven by multi-phase 4D-flow segmentations.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Solver: svMultiPhysics](https://img.shields.io/badge/solver-svMultiPhysics-blue.svg)](https://github.com/SimVascular/svMultiPhysics)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A free alternative for cardiac-cycle wall motion, built on
**svMultiPhysics + SimVascular / MMG / TetGen**. Because the solver reads a single element type,
the whole pipeline is **100 % tetrahedral** (no prismatic boundary layer).

> **Status:** research code, validated for valid mesh motion on a pilot patient. Patient data are
> **not** distributed with this repository (see [Patient data](#patient-data--ethics)).

---

## 1. What it does
Given a thoracic-aorta lumen segmented at several phases of the cardiac cycle (from 4D-flow MRI),
the pipeline computes the blood-flow field and the wall hemodynamic indices (TAWSS, OSI, helicity,
outlet flow split) for two wall models — **fixed (rigid)** and **moving** — and compares them.

The hard part of moving-boundary CFD is advancing an all-tetrahedral mesh through a large cyclic
wall motion while keeping every element valid **and** preserving the near-wall gradient. We avoid
the usual failures (mesh-motion PDEs invert elements; remeshing smears the solution) with a
**fold-free prescribed morph**, computed offline:

1. **Non-rigid registration** of the reference surface onto each phase (N-ICP + *normal shooting*) →
   a fold-free, iso-topological boundary correspondence.
2. **Rest-shape variational volume morph** (regularised Escobar mean-ratio energy with a
   positive-Jacobian barrier) → the near-wall band follows the wall quasi-rigidly, no remeshing.
3. **Total prescription**: every node position is handed to the solver (`EXTENDED` format), so the
   moving-domain solve runs with **no mesh-motion PDE, no remeshing, no solution interpolation**.

## 2. How to run it

### Install (Ubuntu 22.04 / 24.04)
```bash
git clone https://github.com/Elie-Halter/mb_cfd_pipeline.git && cd mb_cfd_pipeline
bash install.sh && source ~/.bashrc && bash check_install.sh   # MMG (USE_ELAS) + svMultiPhysics + 13 patches + Python
sudo apt install -y libpetsc-real-dev                          # PETSc/GAMG, required for the moving-boundary mesh equation
```
> **GAMG is required** for the mesh-motion equation: it is set via
> `PETSC_OPTIONS="-pc_type gamg -pc_gamg_type agg"`, which `run_MB_aniso.sh` exports for you.
> (Plain `petsc-jacobi` blows up on the fine near-wall mesh.)

### Run a patient (end to end)
```bash
cp patients/TEMPLATE.env patients/P0001.env     # fill in: phase STLs, MRI flow split, paths
bash run_patient.sh patients/P0001.env
```
This chains: reference mesh → fold-free morph → RCR calibration → FB run → MB run →
post-processing → FB-vs-MB comparison. Two gates halt the run if the registration folds (G1) or the
flow split deviates from MRI (G2). The only non-coded input is the **segmented phase STLs**.

### Run individual stages
```bash
# 1) offline morph -> EXTENDED displacement file
python3 morph/pipeline.py --mesh ref.vtu --out work --phase 0.40:A.stl --phase 0.60:B.stl ...
# 2) moving-boundary run (GAMG handled by the wrapper)
bash run_MB_aniso.sh <patient>_MB.xml
# 3) fixed- vs moving-boundary comparison
python3 tools/compare_FB_MB.py --fb FB/4-procs wall.vtp <start> <end> --mb MB/4-procs wall.vtp <start> <end>
python3 tools/make_figures.py  --wall-vtp cmp_*_FB_vs_MB_wall.vtp --out-dir figs/
```

## 3. Expected results
- **Valid mesh motion**: the morph advances the mesh through the full cycle with a positive
  Jacobian everywhere and **no remeshing / no interpolation** (verify with the per-step minimum
  scaled Jacobian and inverted-element count).
- **Mass conservation** over the periodic cycle (cycle-mean inflow = outflow); this is a required
  check and selects the mesh (a well-conditioned near-wall mesh is needed under motion).
- **Flow split** matching the 4D-flow MRI target within a few percent per outlet, with the cardiac
  output (outlet sum) consistent with the prescribed inlet.
- **Wall hemodynamics**: `tools/hemo_indices.py` outputs a wall VTP with TAWSS / OSI / helicity;
  `tools/compare_FB_MB.py` outputs the FB, MB and difference maps (open in ParaView), plus summary
  statistics; `tools/gci.py` reports the grid-convergence index across three meshes.

## 4. Repository layout
```
mb_cfd_pipeline/
├── morph/             fold-free prescribed-morph engine (the method) — see morph/README.md
│   ├── pipeline.py        phase STLs + mesh  ->  EXTENDED displacement file
│   ├── register.py        fold-free non-rigid registration (normal shooting)
│   └── morph_volume.py    rest-shape volume morph (J>0 barrier)
├── tools/             setup & post-processing — see tools/README.md
│   ├── build_iso_mesh.py · calibrate_rcr.py · make_patient_xml.py
│   ├── extract_flowsplit_FB.py · hemo_indices.py · compare_FB_MB.py · make_figures.py · gci.py
├── patches/svMP/      13 patches on svMultiPhysics @97ef512 (key: Prescribed_displacement EXTENDED)
├── patients/TEMPLATE.env       per-patient input config (copy & fill in)
├── run_patient.sh · run_MB_aniso.sh
├── FB_example.xml · MB_example.xml          example svMP solver inputs
└── install.sh · check_install.sh · requirements.txt
```

## Updating / Contributing
Get the latest version (after changes have been pushed):
```bash
git pull
```
Publish your own changes (requires write access to the repository):
```bash
git add -u
git commit -m "describe your change"
git push
```
Simulation outputs and patient data are git-ignored, so local runs never conflict and are
never pushed. Collaborators can be granted write access under **Settings → Collaborators**;
others can fork and open a pull request. Please use **normal commits only** (no history
rewriting / force-push) so everyone can `git pull` cleanly.

## Patient data
This repository contains **source code only**. Any patient-derived data (segmentations, meshes,
displacement fields, results) are **never** versioned (`.gitignore`) and are not provided here.

## Built on / references
This pipeline builds on the following open-source software and methods:

- **svMultiPhysics** — open-source parallel finite-element multi-physics solver. https://github.com/SimVascular/svMultiPhysics
- **SimVascular** — cardiovascular modelling suite. https://simvascular.github.io/
- **MMG** (Dapogny, Dobrzynski, Frey) — anisotropic remeshing. https://www.mmgtools.org/
- **TetGen** (Si, 2015, *ACM Trans. Math. Softw.*) — quality tetrahedral mesh generation.
- **PETSc** (Balay et al.) — linear algebra / GAMG algebraic multigrid. https://petsc.org/
- **Escobar et al. (2003), *CMAME*** — regularised mean-ratio energy for simultaneous untangling and
  smoothing (basis of the rest-shape volume morph).
- **Roache (1994), *J. Fluids Eng.*; Celik et al. (2008), *J. Fluids Eng.*** — Grid Convergence Index.
- Prescribed iso-topological morph for image-based moving-domain CFD (registration → volume morph →
  prescription) follows the established moving-boundary methodology for cyclic vascular geometries.

## Citation
If you use this software, please cite it via [`CITATION.cff`](CITATION.cff)
*(journal reference to be added on publication).*

## License
[MIT](LICENSE) for the source code. Third-party dependencies (above) retain their own licenses.

## Acknowledgements
Developed by **Elie Halter** (elie.halter@gmail.com) as part of a cardiovascular-CFD research
project supervised by **Dr. Monika Colombo**, Aarhus University, Dept. of Mechanical and
Production Engineering.
