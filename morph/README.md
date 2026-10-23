# morph/ — wall chain 1: surface registration and the offline rest-shape morph

Wall chain 1 of the pipeline: a reference surface registered onto several segmented phases, and an
offline all-tetrahedral rest-shape morph that advances the whole mesh under the registered motion
without remeshing or interpolation. In production the morph is a validity check (the imposed wall
admits a valid interior mesh at every sample); only its wall positions are prescribed
(`tools/extended_to_standard.py`) and the solver's mesh equation moves the interior. The other wall
chain (one segmented phase, image registration) is in `chain_image/`. Overview: root `README.md`.

| File | Role |
|------|------|
| `pipeline.py` | **driver**, multi-instance: N STL phases + reference mesh → EXTENDED displacement file |
| `register.py` | fold-free non-rigid registration (Laplacian N-ICP + normal shooting) |
| `morph_volume.py` | rest-shape volumetric morph (Escobar energy, J>0 barrier, L-BFGS, local polish) |
| `check_write_any.py` | signed-J validation (samples + midpoints) + EXTENDED writing (periodic PCHIP) |
| `build_aniso.py`, `build_solver_mesh.py` | near-wall anisotropic mesh / solver mesh construction |
| `pin_caps.py` | cap pinning + geodesic smoothstep taper of the wall motion (default L = 12 mm); `sweep` screens taper lengths by inverted-element count |
| `run_pinned_morph.py` | re-runs the morph from an existing registration with pinned caps and the taper (the production wall motion) |
| `analyze_residuals.py` | infeasibility proof (LP) of inversions trapped by the prescribed boundary |
| `morph_any.py`, `morph_aniso.py`, `remap_reg.py`, `diag_pinned.py`, `check_*` | variants / diagnostics |

## Usage
```bash
python3 morph/pipeline.py --mesh ref.vtu --out workdir \
    --phase 0.40:A.stl --phase 0.60:B.stl --phase 0.80:C.stl --phase 0.95:D.stl
# -> workdir/displacement_*.txt (EXTENDED format, all nodes) + reg/morph reports
python3 morph/run_pinned_morph.py --mesh ref.vtu --surf mesh-surfaces --reg-dir workdir \
    --out pinned --L 12 --phase 0.40 --phase 0.60 --phase 0.80 --phase 0.95
python3 tools/extended_to_standard.py pinned/displacement_all_nodes.txt displacement_wall.txt \
    --face mesh-surfaces/wall.vtp        # wall positions only: the file prescribed in production
```
Auto-tuning of the parameters (λ, ray_len, qmin, n_samples, …) is documented at the top of `pipeline.py`.

## Dependencies
numpy, scipy, pyvista, meshio. CPU only (L-BFGS). ~1h for 454k tets / 32 samples, <3 GB RAM.

> Same-directory imports (`import register`, `from morph_volume import …`): run
> `python3 morph/pipeline.py …` from the repo root (the script's directory is on sys.path).
