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
```

| File | What it covers |
|------|----------------|
| `synth.py` | builds the tiny synthetic case (mesh + surfaces + fake result VTUs) |
| `test_imports.py` | all `morph/` and `tools/` Python compiles; key library modules import |
| `test_postproc.py` | each post-processing tool runs end-to-end and writes its output |
| `test_meshing.py` | `build_iso_mesh` radius-based size map; `mmgs_O3` + `tetgen` (skipped if not installed) |

Notes:
- Needs only the Python environment (`requirements.txt`). The meshing test skips cleanly if
  `mmgs_O3` / `tetgen` are absent.
- A full solver end-to-end run is **not** part of this suite (it needs a built svMultiPhysics and a
  real mesh); validate that separately with `check_install.sh` + a real case.
