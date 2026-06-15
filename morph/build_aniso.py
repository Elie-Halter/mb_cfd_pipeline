"""
ANISOTROPIC all-tet near-wall band via an mmg3d metric, surface unchanged (-nosurf).
Metric: M = R diag(h_n^-2, h_t^-2, h_t^-2) R^T, n = direction toward the nearest wall point
(≈ normal); band d<BAND: h_n graded H_N_WALL -> H_ISO; isotropic H_ISO elsewhere.
-nosurf => surface nodes/triangles are preserved EXACTLY => the registered snapshots remain
valid (same boundary positions, matched by coordinates).

Usage: python3 morph/build_aniso.py [h_n_wall=0.18] [h_t=0.7] [band=2.0] [out=<out_dir>/aniso]
Outputs: <out>.mesh/.sol input, <out>.o.mesh mmg3d output, stats (n tets, quality, RAM).
"""
import sys, subprocess, time
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree

ROOT = "."


def write_medit(path, pts, tets, tris, tri_ref):
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\nDimension 3\n")
        f.write(f"Vertices\n{len(pts)}\n")
        np.savetxt(f, np.column_stack([pts, np.zeros(len(pts), int)]), fmt="%.10g %.10g %.10g %d")
        f.write(f"Triangles\n{len(tris)}\n")
        np.savetxt(f, np.column_stack([tris + 1, tri_ref]), fmt="%d")
        f.write(f"Tetrahedra\n{len(tets)}\n")
        np.savetxt(f, np.column_stack([tets + 1, np.ones(len(tets), int)]), fmt="%d")
        f.write("End\n")


def main(h_n_wall=0.18, h_t=0.7, band=2.0, out=f"{ROOT}/work/aniso"):
    g = pv.read(f"{ROOT}/mesh.vtu")
    X = np.asarray(g.points).astype(float)
    tet = g.cells_dict[10].astype(np.int64)
    surf = g.extract_surface().triangulate()
    tri = np.asarray(surf.point_data["vtkOriginalPointIds"]).astype(np.int64)[
        surf.faces.reshape(-1, 4)[:, 1:]]

    wall = pv.read(f"{ROOT}/mesh-surfaces/wall.vtp")
    wall = wall.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    wpts = np.asarray(wall.points); wn = np.asarray(wall.point_data["Normals"])
    tree = cKDTree(wpts)
    d, j = tree.query(X)
    # normal direction: toward the nearest wall point; fall back to the wall normal if too close
    v = X - wpts[j]
    nv = np.linalg.norm(v, axis=1)
    n = np.where(nv[:, None] > 0.05, v / np.maximum(nv, 1e-12)[:, None], wn[j])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]

    H_ISO = 1.0
    t_ = np.clip(d / band, 0, 1)
    hn = h_n_wall + (H_ISO - h_n_wall) * t_ ** 1.2
    ht = np.where(d < band, h_t, H_ISO)
    # tensor M = (1/hn²-1/ht²) n nᵀ + (1/ht²) I  (6 medit components: 11,12,22,13,23,33)
    a = 1 / hn ** 2 - 1 / ht ** 2
    b = 1 / ht ** 2
    M = a[:, None] * np.column_stack([n[:, 0] * n[:, 0], n[:, 0] * n[:, 1], n[:, 1] * n[:, 1],
                                      n[:, 0] * n[:, 2], n[:, 1] * n[:, 2], n[:, 2] * n[:, 2]])
    M[:, 0] += b; M[:, 2] += b; M[:, 5] += b

    write_medit(f"{out}.mesh", X, tet, tri, np.ones(len(tri), int))
    with open(f"{out}.sol", "w") as f:
        f.write("MeshVersionFormatted 2\nDimension 3\nSolAtVertices\n")
        f.write(f"{len(X)}\n1 3\n")
        np.savetxt(f, M, fmt="%.8g")
        f.write("End\n")
    print(f"metric written; target estimate: h_n={h_n_wall} (wall) h_t={h_t} band {band}mm")

    t0 = time.time()
    import os
    opts = os.environ.get("MMG_OPTS", "-nosurf -hgrad 1.3").split()
    r = subprocess.run(["mmg3d_O3", "-in", f"{out}.mesh", "-sol", f"{out}.sol",
                        "-out", f"{out}.o.mesh", "-v", "4"] + opts,
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    print(r.stderr[-1000:])
    print(f"mmg3d: {time.time()-t0:.0f}s, rc={r.returncode}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*([float(a[0])] if len(a) > 0 else []),
         *([float(a[1])] if len(a) > 1 else []),
         *([float(a[2])] if len(a) > 2 else []))
