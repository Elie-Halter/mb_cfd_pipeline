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
| 0004 | `l_elas`: per-element `stiff_factor` parameter | 🟡 Tezduyar base stiffness — active, but nearly uniform in our meshes (see below) |
| 0005 | `Parameters`: XML `Mesh_stiffness_exponent` | 🟢 exposes Opt 4b |
| 0006 | propagates `Mesh_stiffness_exponent` to the runtime | 🟢 same |
| 0007 | stiffness diffusion + fail-fast on collapsed elements | 🟢 mesh quality + remesh trigger |
| 0008 | element skip + spatial stiffness (tweaks) | 🟢 remesh robustness |
| 0009 | relaxes `J_MIN_CRITICAL` to 1e-12 (mesh_scale_factor compat) | 🟢 avoids false collapses |
| 0010 | **distance-to-wall stiffness** (Fluent §3.3.1, Opt 4a) | ⚪ **not used in production** (off unless `SVMP_WALL_STIFF_EXP` is set) |
| 0011 | **real CFL monitor** (fixes the dead code in 0003) | 🟢 detects MB instability |
| 0012 | **O(N²) fix** for writing the deformed mesh in the VTUs | 🟢 fast saves + deformed visualization |
| 0013 | `EXTENDED` mode (wall + boundary-layer / all-node displacement) + MPI | ⚪ **not used in production** — only active when the file carries the `EXTENDED` keyword |
| 0014 | **GCL-consistent deformed geometry** in `construct_fluid` (deformed geometry at n+α_f, `Dg`) | 🟢 **ESSENTIAL** — the discrete-GCL / mass-conservation lock (fork commit `f3e2bc8`) |
| 0015 | prescribed-disp **shared cap∩wall ring nodes** (map all face nodes) | 🟢 fixes ring-node pinning / adjacent-cell inversion under moving caps |

**Summary — what the production runs actually use (verified 2026-10-21).** The wall is imposed
through `Prescribed_displacement` in **STANDARD** mode (0001 + 0015: wall-face nodes only, positions
interpolated linearly in time); the caps are Dirichlet-0; the **interior is moved by the solver's mesh
equation** (incremental linear elasticity) at every step; the fluid uses the GCL-consistent deformed
geometry (0014). On the moving-wall run of case 4826 the solver's wall displacement equals the file to
round-off while the interior differs from the file's (offline-morph) interior values, which confirms
that only the wall is taken from the file even when the file lists every node.

The Jacobian-scaled stiffness (0004–0009, `stiff_factor = (1+eps)/(J^chi+eps)`, eps = 1e-4) uses the
element's *absolute* Jacobian (6 x volume in mesh units). With `Mesh_scale_factor` 0.1 (cm), J^2 << eps
for almost every element, so the factor is nearly uniform (x1.3 between the 1st and 99th percentile
elements on the 4826 production mesh): it does not materially stiffen distorted elements. Mesh validity
comes from cap pinning with a geodesic taper (`morph/pin_caps.py`); the collapsed-element skip
(J < 1e-12) remains as a safety net. Normalising J by a reference value is left as future work.

Diagnostics: 0002, 0011. Performance: 0012. Not used in production: 0003 (superseded), 0010, 0013.

## Affected svMP files
`ComMod.h`, `Parameters.cpp/.h`, `consts.h`, `distribute.cpp`, `fluid.cpp`,
`l_elas.cpp/.h`, `mesh.cpp`, `read_files.cpp/.h`, `set_bc.cpp`, `vtk_xml.cpp`.
