# morph/ — fold-free prescribed-morph engine (core of the method)

Implementation of all-tetrahedral mesh advance under large prescribed displacement, without remeshing
or interpolation. For the overview, see the "Method in brief" section of the root `README.md`.

| File | Role |
|------|------|
| `pipeline.py` | **driver**, multi-instance: N STL phases + reference mesh → EXTENDED displacement file |
| `register.py` | fold-free non-rigid registration (Laplacian N-ICP + normal shooting) |
| `morph_volume.py` | rest-shape volumetric morph (Escobar energy, J>0 barrier, L-BFGS, local polish) |
| `check_write_any.py` | signed-J validation (samples + midpoints) + EXTENDED writing (periodic PCHIP) |
| `build_aniso.py`, `build_solver_mesh.py` | near-wall anisotropic mesh / solver mesh construction |
| `analyze_residuals.py` | infeasibility proof (LP) of inversions trapped by the prescribed boundary |
| `morph_any.py`, `morph_aniso.py`, `remap_reg.py`, `diag_pinned.py`, `check_*` | variants / diagnostics |

## Usage
```bash
python3 morph/pipeline.py --mesh ref.vtu --out workdir \
    --phase 0.40:A.stl --phase 0.60:B.stl --phase 0.80:C.stl --phase 0.95:D.stl
# -> workdir/displacement_*.txt (EXTENDED format, all nodes) + reg/morph reports
```
Auto-tuning of the parameters (λ, ray_len, qmin, n_samples, …) is documented at the top of `pipeline.py`.

## Dependencies
numpy, scipy, pyvista, meshio. CPU only (L-BFGS). ~1h for 454k tets / 32 samples, <3 GB RAM.

> Same-directory imports (`import register`, `from morph_volume import …`): run
> `python3 morph/pipeline.py …` from the repo root (the script's directory is on sys.path).
