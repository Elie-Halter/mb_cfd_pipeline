"""
Morph survival test for the ANISO mesh (<out_dir>/aniso.o.mesh):
 - boundary preserved by -nosurf => the registered snapshots (reg_*.npy, indexed on the iso
   mesh boundary) transfer by exact coordinate matching;
 - same pipeline as morph_volume (harmonic + rest-shape energy), reduced n_samples.
 - first checks the near-wall resolution of the aniso mesh (normal vs tangential h).
Usage: python3 morph/morph_aniso.py [n_samples=8] [maxiter=80]
"""
import sys, time
import numpy as np, pyvista as pv, meshio
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline

ROOT = "."
sys.path.insert(0, f"{ROOT}/morph")
from morph_volume import Energy, signed_vol, local_polish

KNOTS = [(0.00, None), (0.40, "p8"), (0.60, "p12"), (0.80, "p16"), (0.95, "p0")]


def main(n_samples=8, maxiter=80):
    m = meshio.read(f"{ROOT}/work/aniso.o.mesh")
    X0 = m.points.astype(float)
    tet = m.cells_dict["tetra"].astype(np.int64)
    v0 = signed_vol(X0, tet)
    if np.median(v0) < 0:
        tet = tet[:, [0, 2, 1, 3]]
        v0 = signed_vol(X0, tet)
    print(f"aniso: {len(X0)} pts, {len(tet)} tets, minV={v0.min():.2e}, inverted={(v0<=0).sum()}")

    tri = m.cells_dict["triangle"].astype(np.int64)
    bidx = np.unique(tri)

    # near-wall resolution: edges sorted by wall distance, normal/tangential components
    wall = pv.read(f"{ROOT}/mesh-surfaces/wall.vtp")
    wall = wall.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    wtree = cKDTree(np.asarray(wall.points))
    wn = np.asarray(wall.point_data["Normals"])
    d, j = wtree.query(X0)
    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]])
    e.sort(1); e = np.unique(e, axis=0)
    mid_d = 0.5 * (d[e[:, 0]] + d[e[:, 1]])
    ev = X0[e[:, 1]] - X0[e[:, 0]]
    el = np.linalg.norm(ev, axis=1)
    nmid = wn[j[e[:, 0]]]
    en_ = np.abs(np.einsum('ij,ij->i', ev, nmid))
    band = mid_d < 0.5
    print(f"band <0.5mm: {band.sum()} edges ; |normal component| med={np.median(en_[band]):.3f} "
          f"len med={np.median(el[band]):.3f} (target h_n~0.18) ; "
          f"near-normal edges (|cos|>0.7) med={np.median(el[band][np.abs(en_[band]/el[band])>0.7]):.3f}")

    # transfer of the registered snapshots onto the aniso boundary
    vid_iso = np.load(f"{ROOT}/work/reg_bnd_idx.npy")
    g = pv.read(f"{ROOT}/mesh.vtu")
    Piso = np.asarray(g.points)[vid_iso]
    t2 = cKDTree(Piso)
    dd, match = t2.query(X0[bidx])
    print(f"aniso->iso boundary matching: max dist = {dd.max():.2e} (should be ~0)")
    assert dd.max() < 1e-6, "surface not preserved?!"

    P0b = X0[bidx]
    snaps = [P0b] + [np.load(f"{ROOT}/work/reg_{tag}.npy")[match] for _, tag in KNOTS[1:]] + [P0b]
    ts = [t for t, _ in KNOTS] + [1.0]
    spline = CubicSpline(ts, np.stack(snaps), axis=0, bc_type="periodic")

    free_mask = np.ones(len(X0), bool); free_mask[bidx] = False
    n = len(X0)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    L = sp.diags(np.asarray(adj.sum(1)).ravel()) - adj
    fidx = np.where(free_mask)[0]
    Lff = L[fidx][:, fidx].tocsr()
    Lfb = L[fidx][:, bidx].tocsr()

    def harmonic(db):
        out = np.empty((len(fidx), 3))
        rhs = -Lfb.dot(db)
        for k in range(3):
            out[:, k], _ = spl.cg(Lff, rhs[:, k], tol=1e-8, maxiter=400)
        return out

    en = Energy(X0, tet, free_mask)
    tt = np.arange(n_samples) / n_samples
    X = X0.copy()
    worst = 0
    prev = X0.copy()
    for k in range(1, n_samples):
        t0 = time.time()
        B = spline(tt[k])
        Xi = X.copy()
        Xi[fidx] += harmonic(B - X[bidx])
        Xi[bidx] = B
        ninv0 = int((en.detA(Xi) <= 0).sum())
        X = en.solve(Xi, [1e-3] if ninv0 == 0 else [2e-2, 5e-3, 1e-3], maxiter, log=print)
        v = signed_vol(X, tet)
        if (v <= 0).any():
            X = local_polish(X, tet, free_mask, X0, log=print)
            v = signed_vol(X, tet)
        ninv = int((v <= 0).sum())
        worst = max(worst, ninv)
        # midpoint with the previous sample
        vm = signed_vol(0.5 * (prev + X), tet)
        print(f"[{k}/{n_samples-1}] t={tt[k]:.3f} init_inv={ninv0} -> inv={ninv} "
              f"minV={v.min():.2e} mid_inv={(vm<=0).sum()} ({time.time()-t0:.0f}s)")
        prev = X.copy()
        np.save(f"{ROOT}/work/asnap_{k:02d}.npy", X.astype(np.float32))
    vm = signed_vol(0.5 * (prev + X0), tet)
    print(f"wrap-around mid_inv={(vm<=0).sum()} ; worst inv over cycle (samples) = {worst}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
         int(sys.argv[2]) if len(sys.argv) > 2 else 80)
