# tools/ — Utility scripts

| Script | Role | Usage |
|--------|------|-------|
| `calibrate_rcr.py` | RCR Windkessel calibration on the MRI flow split (Opt 3) | `python3 calibrate_rcr.py --flow <inlet.txt> --map <MAP>` |
| `extract_flowsplit_FB.py` | Flow split of an FB run (per-triangle integration), compared to MRI targets | `python3 extract_flowsplit_FB.py <run>/4-procs <mesh>/mesh-surfaces` |
| `checkpoints.py` | Measurements at fixed planes along the aorta: diameter, flow rate, velocity | `python3 checkpoints.py <vtu\|dir> checkpoints.json [--out m.csv]` ; `--make-template <centerline.vtp> N` |
| `hemo_indices.py` | TAWSS / OSI / Helicity over the last cycle (**GID-1** mapping, deformation-proof; wall VTP + volume VTU) | `python3 hemo_indices.py <dir> --wall wall.vtp --cycle-start <start> --cycle-end <end>` |
| `compare_FB_MB.py` | **FB vs MB comparison** (ΔTAWSS/ΔOSI, spatial correlation) + wall VTP for figures | `python3 compare_FB_MB.py --fb <dir> wall.vtp <start> <end> --mb <dir> wall.vtp <start> <end>` |
| `make_figures.py` | Paper figures from the comparison VTP (headless matplotlib hist/scatter + 3D wall maps); also serves the **segmentation sensitivity** study (segA vs segB) | `python3 make_figures.py --wall-vtp cmp_*_FB_vs_MB_wall.vtp --out-dir figs/` |
| `gci.py` | **Grid Convergence Index** (Celik 2008) on ≥3 grids — mesh-independence verification | `python3 gci.py --mesh M1.vtu M2.vtu M3.vtu --wall M1_*.vtp M2_*.vtp M3_*.vtp` |
| `make_patient_xml.py` | Derives a patient's FB/MB XML files from the templates (paths repointed, comments kept) | `python3 make_patient_xml.py --fb-template … --mb-template … --mesh … --surf … --disp … --fb-out … --mb-out …` |
| `build_iso_mesh.py` | Reference STL → all-tet mesh + surfaces (GID/EID/FaceID). Uniform, or **radius-based** (`--rbm N`: ~N elements across the local diameter → finer small branches, no SimVascular) | `python3 build_iso_mesh.py <stl> <surfaces> <out> [hmax] [--rbm 6 --hmin 0.2 --hmax 0.8]` |

## Units
Mesh coordinates in mm; solver works in cm (`Mesh_scale_factor 0.1`). Velocity cm/s, pressure
dyne/cm², flow rate mL/s. Flow split = per-triangle integration `Q = Σ(v_tri·area_vec)/100`.
Wall→volume mapping uses GlobalNodeID−1 (the results VTU preserves mesh order), not a positional
KDTree — required for the moving boundary.
