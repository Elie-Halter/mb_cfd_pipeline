# tests/ — self-contained test suite

Validates the pipeline **without any patient data or solver run**: every tool is exercised on a
tiny synthetic case (a small all-tet cylinder with fake Velocity/WSS/Displacement), plus
compile/import smoke tests and the meshing toolchain.

```bash
bash tests/run_tests.sh        # run everything, prints PASS/FAIL + a summary
```

Or run a single component test:
```bash
python3 tests/test_imports.py    # every morph/ and tools/ file compiles + key modules import
python3 tests/test_postproc.py   # hemo_indices, extract_flowsplit_FB, compare_FB_MB, make_figures, gci, make_patient_xml
python3 tests/test_meshing.py    # radius-based size map + MMG/TetGen toolchain (gated on the binaries)
python3 tests/test_benchmarks.py # analytic Womersley, P1 integrals, benchmark templates and case generators
python3 tests/test_chain_image.py # image-registration wall chain + phase subsets on a synthetic wall field
```

| File | What it covers |
|------|----------------|
| `synth.py` | builds the tiny synthetic case (mesh + surfaces + fake result VTUs) |
| `test_imports.py` | all `morph/`, `chain_image/`, `tools/` and `benchmarks/` Python compiles; key library modules import |
| `test_postproc.py` | each post-processing tool runs end-to-end and writes its output |
| `test_meshing.py` | `build_iso_mesh` radius-based size map; `mmgs_O3` + `tetgen` (skipped if not installed) |
| `test_chain_image.py` | transform inversion on an analytic field; periodic PCHIP; `chain_image/` smooth → taper → write on a noisy synthetic wall field (rings pinned, clock, negative inflow); `tools/phase_subset_wall.py` |
| `test_benchmarks.py` | Womersley closed form vs its ODE; P1 divergence/flux/volume integrals on exact fields; `benchmarks/*/case.xml.in` render; B1/B2 generators pass their pre-flight checks (gmsh-gated) |

Notes:
- Needs only the Python environment (`requirements.txt`). The meshing test skips cleanly if
  `mmgs_O3` / `tetgen` are absent.
- A full solver end-to-end run is **not** part of this suite (it needs a built svMultiPhysics and a
  real mesh); `bash benchmarks/run_all.sh` runs the analytic benchmarks with the solver.
