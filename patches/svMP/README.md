# svMultiPhysics patches

Modifications of our fork, to be applied on upstream **commit `97ef512`**
("Encapsulate Newton iteration in Integrator class (#450)"):
```bash
git clone https://github.com/SimVascular/svMultiPhysics.git
cd svMultiPhysics
git checkout 97ef51223e5a079bdee018bd9c0490c861c07fe0
git am <repo>/patches/svMP/*.patch     # install.sh does this automatically
```
Regenerated from `97ef512..HEAD`.
Older versions (before defect fixes) are in `_old_*/`.

## Usefulness verdict (each patch serves the MB run)

| Patch | Role | Verdict |
|-------|------|---------|
| 0001 | BC `Prescribed_displacement` (wall positions from file) | 🟢 **ESSENTIAL** — core of the MB |
| 0002 | Jacobian monitoring in `construct_mesh` | ⚪ Diagnostic (mesh quality log) |
| 0003 | CFL monitoring in `construct_fluid` (initial version) | ⚪ made effective by 0011 |
| 0004 | `l_elas`: per-element `stiff_factor` parameter | 🟢 Tezduyar base stiffness |
| 0005 | `Parameters`: XML `Mesh_stiffness_exponent` | 🟢 exposes Opt 4b |
| 0006 | propagates `Mesh_stiffness_exponent` to the runtime | 🟢 same |
| 0007 | stiffness diffusion + fail-fast on collapsed elements | 🟢 mesh quality + remesh trigger |
| 0008 | element skip + spatial stiffness (tweaks) | 🟢 remesh robustness |
| 0009 | relaxes `J_MIN_CRITICAL` to 1e-12 (mesh_scale_factor compat) | 🟢 avoids false collapses |
| 0010 | **distance-to-wall stiffness** (Fluent §3.3.1, Opt 4a) | 🟢 quasi-rigid boundary layer |
| 0011 | **real CFL monitor** (fixes the dead code in 0003) | 🟢 detects MB instability |
| 0012 | **O(N²) fix** for writing the deformed mesh in the VTUs | 🟢 fast saves + deformed visualization |
| 0013 | `EXTENDED` mode (wall + boundary-layer displacement) + MPI | 🟢 **the moving-boundary BC** — prescribed displacement, all nodes |

**Summary:** no unused modifications remain. CFL (0003) became functional via 0011,
and visualization (0012) no longer penalizes performance. The EXTENDED mode (0013) carries
the moving-boundary prescription used in production.

## Affected svMP files
`ComMod.h`, `Parameters.cpp/.h`, `consts.h`, `distribute.cpp`, `fluid.cpp`,
`l_elas.cpp/.h`, `mesh.cpp`, `read_files.cpp/.h`, `set_bc.cpp`, `vtk_xml.cpp`.
