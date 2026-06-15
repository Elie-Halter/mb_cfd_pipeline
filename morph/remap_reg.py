"""
Re-maps the registered snapshots (defined on the reference iso mesh boundary) onto the
boundary of a REMESHED mesh (surface modified by mmg3d without -nosurf, gap <= hausd).
For each new boundary node: nearest point on the old surface (triangle + barycentrics)
-> P1 interpolation of each phase's registered position field.
Usage: python3 morph/remap_reg.py <mesh.o.mesh> <out_prefix>
Outputs: <out_prefix>_bidx.npy, <out_prefix>_<tag>.npy (boundary positions per phase)
"""
import sys
import numpy as np, pyvista as pv, vtk, meshio

ROOT = "."
TAGS = ["p8", "p12", "p16", "p0"]


def main(mesh_path, out_prefix):
    vid = np.load(f"{ROOT}/work/reg_bnd_idx.npy")
    tri = np.load(f"{ROOT}/work/reg_tri.npy")
    g = pv.read(f"{ROOT}/mesh.vtu")
    P0 = np.asarray(g.points)[vid]
    old = pv.PolyData(P0, np.column_stack([np.full(len(tri), 3), tri]).ravel())
    loc = vtk.vtkCellLocator(); loc.SetDataSet(old); loc.BuildLocator()

    m = meshio.read(mesh_path)
    X = m.points.astype(float)
    btri = m.cells_dict["triangle"].astype(np.int64)
    bidx = np.unique(btri)
    pts = X[bidx]

    cell = np.empty(len(pts), np.int64); bary = np.empty((len(pts), 3)); dist = np.empty(len(pts))
    c = vtk.mutable(0); sub = vtk.mutable(0); d2 = vtk.mutable(0.0)
    p = [0.0, 0.0, 0.0]; gc = vtk.vtkGenericCell()
    for k in range(len(pts)):
        loc.FindClosestPoint(pts[k], p, gc, c, sub, d2)
        cell[k] = c.get(); dist[k] = np.sqrt(d2.get())
        a, b, cc = P0[tri[cell[k]]]
        v0, v1, v2 = b - a, cc - a, np.asarray(p) - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        bary[k] = [1 - w1 - w2, w1, w2]
    print(f"{len(pts)} boundary nodes remapped ; max dist to old surface={dist.max():.3f} "
          f"p99={np.percentile(dist, 99):.3f}")
    np.save(f"{out_prefix}_bidx.npy", bidx)
    # phase 0 = interpolated positions of P0 (≈ pts, up to hausd) -> keep exact pts
    np.save(f"{out_prefix}_ref.npy", pts)
    for tag in TAGS:
        Y = np.load(f"{ROOT}/work/reg_{tag}.npy")
        D = Y - P0                                   # registered displacement field
        Dn = np.einsum('ki,kij->kj', bary, D[tri[cell]])
        np.save(f"{out_prefix}_{tag}.npy", pts + Dn)
    print(f"wrote {out_prefix}_{{bidx,ref,{','.join(TAGS)}}}.npy")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
