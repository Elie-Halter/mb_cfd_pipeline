# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/). Version numbers match `CITATION.cff`.

## [Unreleased]

To be released as 0.3.0 (the version `CITATION.cff` already carries), together with the
submission of the companion paper.

### Added
- `benchmarks/`: synthetic analytic verification suite that needs no patient data — free-stream
  preservation (B1), analytically expanding tube with the discrete mass balance with and without
  patch 0014 (B2, plus its temporal order), Womersley flow (B3, plus a refinement study);
  `run_all.sh` generates, pre-flights, runs and analyses them; `build_nogcl.sh` builds the
  contrast solver without patch 0014.
- `chain_image/`: second wall chain — one segmented phase propagated to every phase by inverted
  B-spline image registration (elastix), periodic smoothing, cap taper, displacement file;
  `tools/phase_subset_wall.py` rebuilds the wall motion from a subset of the phases.
- `run_patient.sh --dry-run`: prints every command and the generated XMLs, runs nothing else.
- `morph/pipeline.py --register-only`: stop after the boundary registration.
- `tools/make_patient_xml.py` checks the MB output against the production configuration
  (moving no-slip wall, `Dir 0` on every cap in the mesh equation) and refuses an EXTENDED
  displacement file; `--inflow` repoints the inflow file.
- GitHub Actions workflow (`.github/workflows/tests.yml`) running `tests/run_tests.sh` on every
  push and pull request; CI badge in the README.
- Tests for the production XML checks and for `run_patient.sh --dry-run`.
- `CHANGELOG.md`; README sections "Reproducing the paper" and "Verification".

### Changed
- `MB_example.xml` and `FB_example.xml` regenerated from the production inputs of the companion
  paper (relative paths, English comments on every non-default setting, labelled example RCR
  values). MB: 3896 steps, 15 nonlinear iterations, backflow coefficient 1.0, larger NS
  linear-solver limits, parabolic inflow profile, RCR initial pressure 113000 dyn/cm²,
  FSILS for the mesh equation and `Dir 0` on all five caps. FB: same changes where they apply,
  8 nonlinear iterations. Inflow file renamed `inflow.txt`, wall file `wall_displacement.txt`.
- `run_patient.sh` now runs the production chain by default: registration → cap-pinned morph
  with the 12 mm smoothstep taper (`morph/run_pinned_morph.py`, gate G1) → wall-only STANDARD
  file (`tools/extended_to_standard.py --face wall.vtp`) → XMLs from the examples. Defaults
  follow production (4 cycles, analysis on the last). The previous path, which fed the unpinned
  all-node EXTENDED file to the MB run, is removed.
- Documentation: every tool listed, both wall chains described, `REPRODUCE.md` rewritten,
  companion-paper title updated (*Moving versus rigid wall in aortic CFD: diastolic flow in
  healthy and Turner-syndrome aortas*); `CITATION.cff` gains a `preferred-citation` block.

### Fixed
- `tools/build_iso_mesh.py` radius map (`--rbm`): `vtkOBBTree` line intersection returns
  non-deterministic hits with VTK 9.7; replaced by `vtkCellLocator` (same map to 2e-6 mm with
  VTK 9.6, about 7× faster).

## [0.2.0] - 2026-10-23

### Added
- `tools/retime_wall.py`, `tools/remove_rigid_motion.py`, `tools/volume_weighted_means.py`
  (analysis tools of the companion paper), `tools/build_fb_at_phase.py` (rigid mesh frozen at any
  phase) and `tools/volume_cycle.py` (lumen volume against cycle phase).
- Cohort tooling: cap labelling (`tools/cohort_cap_label.py`), run provenance
  (`tools/run_provenance.py`), guarded solver launcher (`tools/svmp_run.sh`), MMG size floor.
- `tools/make_inlet_flow.py` and `tools/check_inlet_sign.py`: inlet flow file with the solver's
  sign and exact period; `run_patient.sh` refuses a backwards inflow.
- Container recipes (`Dockerfile`, `apptainer/mbcfd.def`).
- `REPRODUCE.md` (clone → install → run → post-process).

### Changed
- Production method documented as run: wall-only prescribed displacement, caps pinned with a
  12 mm taper, interior moved by the solver's mesh equation, GCL-consistent fluid assembly; the
  offline rest-shape morph is a validity check (README, CITATION, `patches/svMP/README.md`).
- README cut to the essentials; solver pitfalls moved to `CONVENTIONS.md`.

### Fixed
- Inlet flow sign: the solver imposes v = Q · profile · n_out, so an inlet flow rate must be
  negative.
- `tools/mass_balance_ale.py`: sign of the dV/dt term.
- svMultiPhysics patch chain: the fifteen patches apply cleanly with `git am`.

## [0.1.0] - 2026-06-12

First public version (repository history starts 2026-06-15).

### Added
- All-tetrahedral moving-boundary pipeline: fold-free surface registration and offline
  rest-shape morph (`morph/`), meshing without SimVascular (`tools/build_iso_mesh.py`, uniform
  or radius-based with MMG + TetGen, sliver removal), RCR calibration, FB/MB XML generation,
  post-processing (TAWSS, OSI, helicity, FB/MB comparison, GCI) and `run_patient.sh` with the
  G1 (folds) and G2 (flow split) gates.
- svMultiPhysics patches 0001–0015 on commit `97ef512`, among them `Prescribed_displacement`
  (0001), the GCL-consistent deformed geometry (0014) and shared cap/wall ring nodes (0015);
  `install.sh` builds MMG, PETSc support and the patched solver.
- Self-contained test suite on a synthetic case (no patient data, no solver).
- Cap pinning with a geodesic taper and mass-balance checks (`morph/pin_caps.py`,
  `tools/mass_balance*.py`).

### Fixed
- Compatibility with SciPy ≥ 1.14, NumPy 2.0 and CMake ≥ 4.
- `run_patient.sh`: inflow staging and displacement-file copy.

[Unreleased]: https://github.com/Elie-Halter/mb_cfd_pipeline/compare/d4d3bd2...HEAD
[0.2.0]: https://github.com/Elie-Halter/mb_cfd_pipeline/compare/f10a971...d4d3bd2
[0.1.0]: https://github.com/Elie-Halter/mb_cfd_pipeline/commit/f10a971
