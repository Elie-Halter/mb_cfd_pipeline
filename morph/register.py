"""
Non-rigid FOLD-FREE registration v3: reference boundary surface (wall+caps) -> phase STL.

  min_D  Σ w_i ||x_i + D_i - q_i||² + λ Σ_(ij) ||D_i - D_j||²,   λ annealed 300 -> 1.

Fold-free by construction: q_i obtained by NORMAL SHOOTING (ray intersection along the
deformed vertex ±normal with the STL, normal-compatibility filter) -> the pull-back has no
spurious tangential component (which is what folded the closest-point correspondence at the
branch saddles). Closest-point fallback if there is no intersection. Cleanup passes:
  - relax: 1-ring smoothing of folds + ray reprojection (fine steps),
  - harmonic patch WITHOUT reprojection on residual folds (we tolerate a local ~0.3 mm
    deviation from the target, below segmentation noise, to guarantee 0 folds).
Chaining: ref->phase_1->phase_2->phase_3; ref->phase_0.

Outputs: <out_dir>/reg_<tag>.npy, reg_bnd_idx.npy, reg_tri.npy, reg_report.txt
"""
import json
import numpy as np, pyvista as pv, vtk
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from vtk.util.numpy_support import vtk_to_numpy

ROOT = "."


def load_source(mesh_path=f"{ROOT}/mesh.vtu"):
    g = pv.read(mesh_path)
    X0 = np.asarray(g.points)
    surf = g.extract_surface().triangulate()
    vid = np.asarray(surf.point_data["vtkOriginalPointIds"]).astype(np.int64)
    tri = surf.faces.reshape(-1, 4)[:, 1:].copy()
    surf2 = surf.compute_normals(cell_normals=True, point_normals=False,
                                 auto_orient_normals=True, inplace=False)
    n_auto = np.asarray(surf2.cell_data["Normals"])
    P = np.asarray(surf.points)
    n_raw = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    n_raw /= np.maximum(np.linalg.norm(n_raw, axis=1), 1e-30)[:, None]
    fl = np.einsum('ij,ij->i', n_raw, n_auto) < 0
    tri[fl] = tri[fl][:, [0, 2, 1]]
    return X0, vid, tri


def tri_geom(P, tri):
    n = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    a2 = np.linalg.norm(n, axis=1)
    nn = n / np.maximum(a2, 1e-30)[:, None]
    l2 = ((P[tri[:, 1]] - P[tri[:, 0]]) ** 2).sum(1) + \
         ((P[tri[:, 2]] - P[tri[:, 1]]) ** 2).sum(1) + \
         ((P[tri[:, 0]] - P[tri[:, 2]]) ** 2).sum(1)
    q = 2 * np.sqrt(3) * a2 / np.maximum(l2, 1e-30)
    return nn, a2, q


def vnormals(P, tri, nv):
    fn, a2, _ = tri_geom(P, tri)
    vn = np.zeros((nv, 3))
    for k in range(3):
        np.add.at(vn, tri[:, k], fn * a2[:, None])
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]


def intrinsic_folds(P, tri, nv):
    fn, a2, q = tri_geom(P, tri)
    vn = vnormals(P, tri, nv)
    nt = vn[tri].mean(1)
    nt /= np.maximum(np.linalg.norm(nt, axis=1), 1e-30)[:, None]
    return (np.einsum('ij,ij->i', fn, nt) < 0.0) | (q < 0.02), q


class Target:
    def __init__(self, name):
        path = name if "/" in name else f"{ROOT}/phases/{name}"
        s = pv.read(path).triangulate().clean()
        s = s.compute_normals(cell_normals=True, point_normals=True, auto_orient_normals=True)
        self.cn = np.asarray(s.cell_data["Normals"])
        self.obb = vtk.vtkOBBTree(); self.obb.SetDataSet(s); self.obb.BuildLocator()
        self.loc = vtk.vtkCellLocator(); self.loc.SetDataSet(s); self.loc.BuildLocator()

    def closest(self, pts):
        q = np.empty_like(pts); cid = np.empty(len(pts), np.int64); dist = np.empty(len(pts))
        c = vtk.mutable(0); sub = vtk.mutable(0); d2 = vtk.mutable(0.0)
        p = [0.0, 0.0, 0.0]; gc = vtk.vtkGenericCell()
        for k in range(len(pts)):
            self.loc.FindClosestPoint(pts[k], p, gc, c, sub, d2)
            q[k] = p; cid[k] = c.get(); dist[k] = np.sqrt(d2.get())
        return q, cid, dist

    def ray_project(self, Y, VN, ray_len, ncmin=0.3):
        """nearest ±normal intersection with normal compatibility; ok=False otherwise."""
        q = Y.copy(); ok = np.zeros(len(Y), bool)
        pts = vtk.vtkPoints(); ids = vtk.vtkIdList()
        for k in range(len(Y)):
            p = Y[k]; n = VN[k]
            bestd = np.inf
            for s in (1.0, -1.0):
                if self.obb.IntersectWithLine(p, p + (s * ray_len) * n, pts, ids):
                    for m in range(pts.GetNumberOfPoints()):
                        c = ids.GetId(m)
                        if np.dot(self.cn[c], n) <= ncmin:
                            continue
                        x = np.asarray(pts.GetPoint(m))
                        d = np.linalg.norm(x - p)
                        if d < bestd:
                            bestd = d; q[k] = x; ok[k] = True
        return q, ok


def relax(Y, tri, adj, deg, tgt, nv, rounds=40, qmin=0.0, log=lambda *a: None):
    """1-ring smoothing of folds (and triangles q<qmin) + ray reprojection."""
    for r in range(rounds):
        fold, qq = intrinsic_folds(Y, tri, nv)
        fold = fold | (qq < qmin)
        if not fold.any():
            return Y, 0
        bad = np.zeros(nv, bool); bad[tri[fold].ravel()] = True
        bad |= adj.dot(bad.astype(float)) > 0
        avg = adj.dot(Y) / deg[:, None]
        Y[bad] = 0.5 * Y[bad] + 0.5 * avg[bad]
        VN = vnormals(Y, tri, nv)
        q, ok = tgt.ray_project(Y[bad], VN[bad], ray_len=1.5)
        sel = np.where(bad)[0][ok]
        Y[sel] = 0.5 * Y[sel] + 0.5 * q[ok]
        if r % 10 == 0:
            log(f"    relax {r}: folds={int(fold.sum())} nodes affected={int(bad.sum())}")
    fold, _ = intrinsic_folds(Y, tri, nv)
    return Y, int(fold.sum())


def harmonic_patch_fix(Y, tri, adj, deg, nv, log=lambda *a: None):
    """residual folds: local harmonic positions (2-ring), WITHOUT reprojection."""
    for it in range(6):
        fold, _ = intrinsic_folds(Y, tri, nv)
        if not fold.any():
            return Y, 0
        bad = np.zeros(nv, bool); bad[tri[fold].ravel()] = True
        for _ in range(2):
            bad |= adj.dot(bad.astype(float)) > 0
        idx = np.where(bad)[0]
        L = sp.diags(deg) - adj
        Lff = L[idx][:, idx].tocsc()
        Lfb = L[idx][:, ~bad].tocsr()
        rhs = -Lfb.dot(Y[~bad])
        try:
            sol = spl.spsolve(Lff, rhs)
        except Exception:
            break
        Y[idx] = sol if sol.ndim == 2 else sol.reshape(-1, 3)
        log(f"    harmonic patch {it}: folds={int(fold.sum())} nodes={len(idx)}")
    fold, _ = intrinsic_folds(Y, tri, nv)
    return Y, int(fold.sum())


def register(Yinit, Xref, tri, edges, adj, deg, tgt, hbar,
             lam_schedule=(300, 100, 30, 10, 3, 1), n_inner=3, log=print):
    nv = len(Yinit)
    i, j = edges[:, 0], edges[:, 1]
    wv = sp.coo_matrix((np.ones(len(edges)), (i, j)), shape=(nv, nv)); wv = wv + wv.T
    L = sp.diags(np.asarray(wv.sum(1)).ravel()) - wv
    Y = Yinit.copy()
    for lam in lam_schedule:
        for it in range(n_inner):
            _, _, dist0 = tgt.closest(Y)
            ray_len = float(np.clip(5 * np.median(dist0) + 0.5, 2.0, 10.0))
            VN = vnormals(Y, tri, nv)
            q, ok = tgt.ray_project(Y, VN, ray_len)
            w = ok.astype(float)
            A = (sp.diags(w) + (lam * hbar ** -2) * L).tocsc()
            solve = spl.factorized(A)
            rhs = w[:, None] * (q - Xref)
            D = np.column_stack([solve(rhs[:, k]) for k in range(3)])
            Y = Xref + D
        if lam <= 10:
            Y, nf = relax(Y, tri, adj, deg, tgt, nv, rounds=20, log=log)
        _, _, dist = tgt.closest(Y)
        fold, _ = intrinsic_folds(Y, tri, nv)
        log(f"  λ={lam:g}: hits={int(ok.sum())}/{nv} dist med={np.median(dist):.3f} "
            f"p99={np.percentile(dist, 99):.2f} max={dist.max():.2f} folds={int(fold.sum())}")

    Y, nf = relax(Y, tri, adj, deg, tgt, nv, rounds=60, log=log)
    if nf:
        Y, nf = harmonic_patch_fix(Y, tri, adj, deg, nv, log=log)
    # quality: unfold AND un-squeeze the triangles crushed by the registration (q<0.15),
    # by sliding along the target (no loss of fidelity)
    Y, _ = relax(Y, tri, adj, deg, tgt, nv, rounds=80, qmin=0.15, log=log)
    if intrinsic_folds(Y, tri, nv)[0].any():
        Y, _ = harmonic_patch_fix(Y, tri, adj, deg, nv, log=log)
    fold_i, qq = intrinsic_folds(Y, tri, nv)
    q, cid, dist = tgt.closest(Y)
    stats = dict(folds=int(fold_i.sum()), min_tri_q=float(qq.min()),
                 dist_med=float(np.median(dist)), dist_p99=float(np.percentile(dist, 99)),
                 dist_max=float(dist.max()))
    return Y, stats


def main():
    X0, vid, tri = load_source()
    nb = len(vid)
    P0 = X0[vid]
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(1); edges = np.unique(e, axis=0)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(nb, nb))
    adj = (adj + adj.T).tocsr()
    deg = np.asarray(adj.sum(1)).ravel()
    l0 = np.linalg.norm(P0[edges[:, 0]] - P0[edges[:, 1]], axis=1)
    hbar = float(l0.mean())
    print(f"boundary surface: {nb} nodes, {len(tri)} tris, h̄={hbar:.3f}")
    np.save(f"{ROOT}/work/reg_bnd_idx.npy", vid)
    np.save(f"{ROOT}/work/reg_tri.npy", tri)

    fold0, _ = intrinsic_folds(P0, tri, nb)
    print(f"intrinsic folds on the reference surface (sanity, should be ~0): {int(fold0.sum())}")

    report, results = {}, {}
    chain = [("p8", "phase_1_capped.stl", None),
             ("p12", "phase_2_capped.stl", "p8"),
             ("p16", "phase_3_capped.stl", "p12"),
             ("p0", "phase_0_capped.stl", None)]
    for tag, name, init_tag in chain:
        print(f"\n=== registration -> {name} (init: {init_tag or 'ref'}) ===")
        tgt = Target(name)
        Yinit = results[init_tag].copy() if init_tag else P0.copy()
        Y, stats = register(Yinit, P0, tri, edges, adj, deg, tgt, hbar)
        r = np.linalg.norm(Y[edges[:, 0]] - Y[edges[:, 1]], axis=1) / l0
        stats["edge_ratio_min_p1_p99_max"] = [float(r.min()), float(np.percentile(r, 1)),
                                              float(np.percentile(r, 99)), float(r.max())]
        stats["disp_max"] = float(np.linalg.norm(Y - P0, axis=1).max())
        print(f"  -> folds={stats['folds']} min_q={stats['min_tri_q']:.3f} "
              f"dist(med/p99/max)={stats['dist_med']:.3f}/{stats['dist_p99']:.2f}/{stats['dist_max']:.2f} "
              f"edge ratio={np.round(stats['edge_ratio_min_p1_p99_max'], 2)} |D|max={stats['disp_max']:.2f}")
        np.save(f"{ROOT}/work/reg_{tag}.npy", Y)
        results[tag] = Y
        report[tag] = stats
    with open(f"{ROOT}/work/reg_report.txt", "w") as f:
        json.dump(report, f, indent=1)
    print("\nOK")


if __name__ == "__main__":
    main()
