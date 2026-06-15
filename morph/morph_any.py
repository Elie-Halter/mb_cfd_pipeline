"""
Generic cycle morph for a tet mesh + remapped boundary snapshots (remap_reg.py).
Usage: python3 morph/morph_any.py <mesh.o.mesh> <reg_prefix> [n_samples=8] [maxiter=80] [snap_prefix]
"""
import sys, time
import numpy as np
import meshio
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.interpolate import CubicSpline, PchipInterpolator

ROOT = "."
sys.path.insert(0, f"{ROOT}/morph")
from morph_volume import Energy, signed_vol, local_polish, smooth_bnd

KNOT_T = [0.00, 0.40, 0.60, 0.80, 0.95, 1.00]
TAGS = ["p8", "p12", "p16", "p0"]


def main(mesh_path, prefix, n_samples=8, maxiter=80, snap_prefix=None):
    m = meshio.read(mesh_path)
    X0 = m.points.astype(float)
    tet = m.cells_dict["tetra"].astype(np.int64)
    v0 = signed_vol(X0, tet)
    if np.median(v0) < 0:
        tet = tet[:, [0, 2, 1, 3]]
        v0 = signed_vol(X0, tet)
    bidx = np.load(f"{prefix}_bidx.npy")
    print(f"{mesh_path}: {len(X0)} pts {len(tet)} tets, ref inverted={(v0<=0).sum()}, "
          f"boundary {len(bidx)}")

    P0b = np.load(f"{prefix}_ref.npy")
    snaps = [P0b] + [np.load(f"{prefix}_{t}.npy") for t in TAGS] + [P0b]
    # periodic PCHIP (anti-overshoot): extended knots
    arr = np.stack(snaps)
    text = [-0.2, -0.05] + KNOT_T + [1.4]
    aext = np.concatenate([arr[[3, 4]], arr, arr[[1]]], axis=0)
    spline = PchipInterpolator(text, aext, axis=0)

    # boundary triangles in local numbering (smoothing of the interpolated boundary)
    tri = m.cells_dict["triangle"].astype(np.int64)
    l2g = -np.ones(len(X0), np.int64); l2g[bidx] = np.arange(len(bidx))
    triL = l2g[tri]
    eB = np.vstack([triL[:, [0, 1]], triL[:, [1, 2]], triL[:, [2, 0]]])
    eB.sort(1); eB = np.unique(eB, axis=0)
    adjB = sp.coo_matrix((np.ones(len(eB)), (eB[:, 0], eB[:, 1])),
                         shape=(len(bidx), len(bidx)))
    adjB = (adjB + adjB.T).tocsr()
    degB = np.asarray(adjB.sum(1)).ravel()

    free_mask = np.ones(len(X0), bool); free_mask[bidx] = False
    n = len(X0)
    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]])
    e.sort(1); e = np.unique(e, axis=0)
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
    X = X0.copy(); prev = X0.copy()
    worst = 0; worst_mid = 0
    # the remeshed mesh starts on a boundary ≈P0 (hausd gap): already consistent (ref.npy=pts)
    for k in range(1, n_samples):
        t0 = time.time()
        B = smooth_bnd(np.asarray(spline(tt[k])), triL, adjB, degB)
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
        vm = signed_vol(0.5 * (prev + X), tet)
        worst = max(worst, ninv); worst_mid = max(worst_mid, int((vm <= 0).sum()))
        print(f"[{k}/{n_samples-1}] t={tt[k]:.3f} init_inv={ninv0} -> inv={ninv} "
              f"minV={v.min():.2e} mid_inv={(vm<=0).sum()} ({time.time()-t0:.0f}s)")
        prev = X.copy()
        if snap_prefix:
            np.save(f"{snap_prefix}_{k:02d}.npy", X.astype(np.float32))
    vm = signed_vol(0.5 * (prev + X0), tet)
    print(f"wrap-around mid_inv={(vm<=0).sum()} ; worst (sample)={worst} worst (mid)={worst_mid}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 8, int(a[4]) if len(a) > 4 else 80,
         a[5] if len(a) > 5 else None)
