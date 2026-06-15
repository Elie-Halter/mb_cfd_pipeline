"""
Iso-topological volume morph: advances the full tet mesh along the cycle.
- Boundary: periodic spline over {ref, reg_p8, reg_p12, reg_p16, reg_p0} (registered snapshots).
- Interior, at each time sample: init = previous solution + harmonic extension
  (graph Laplacian) of the boundary increment, then L-BFGS minimization of Escobar's
  regularized mean-ratio energy RELATIVE TO THE REFERENCE SHAPE of each tet
  (W = reference tet edges => as-rigid-as-possible morphing, det->0 barrier, the near-wall
  band follows the wall without a prescribed/free interface). Vectorized analytic gradient.
- Local polish (2-ring) if inversions persist after the global pass.
Outputs: <out_dir>/morph_snapshots.npz (positions at each sample), morph_report.txt
Usage: python3 morph/morph_volume.py [n_samples=32] [maxiter=80]
"""
import sys, time, json
import numpy as np, pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, PchipInterpolator

ROOT = "."
KNOTS = [(0.00, None), (0.40, "p8"), (0.60, "p12"), (0.80, "p16"), (0.95, "p0")]


def load():
    g = pv.read(f"{ROOT}/mesh.vtu")
    X0 = np.asarray(g.points).astype(float)
    tet = g.cells_dict[10].astype(np.int64)
    gid = np.asarray(g.point_data["GlobalNodeID"]).astype(int)
    vid = np.load(f"{ROOT}/work/reg_bnd_idx.npy")
    return X0, tet, gid, vid


def signed_vol(P, tet):
    a, b, c, d = P[tet[:, 0]], P[tet[:, 1]], P[tet[:, 2]], P[tet[:, 3]]
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6


class Energy:
    """E = Σ_e ||S_e M_e||²_F / (3 σ^{2/3}), σ=(det+sqrt(det²+4δ²))/2, M_e = W_e^{-1} (ref)."""

    def __init__(self, X0, tet, free_mask):
        self.tet = tet
        n = len(X0)
        W = np.stack([X0[tet[:, 1]] - X0[tet[:, 0]],
                      X0[tet[:, 2]] - X0[tet[:, 0]],
                      X0[tet[:, 3]] - X0[tet[:, 0]]], axis=2)        # columns
        self.M = np.linalg.inv(W)
        self.free = np.where(free_mask)[0]
        self.n = n
        self.loc = -np.ones(n, np.int64)
        self.loc[self.free] = np.arange(len(self.free))

    def detA(self, X):
        S = np.stack([X[self.tet[:, 1]] - X[self.tet[:, 0]],
                      X[self.tet[:, 2]] - X[self.tet[:, 0]],
                      X[self.tet[:, 3]] - X[self.tet[:, 0]]], axis=2)
        A = S @ self.M
        return np.linalg.det(A)

    def fg(self, xf, Xfull, delta):
        X = Xfull.copy()
        X[self.free] = xf.reshape(-1, 3)
        t = self.tet
        S = np.stack([X[t[:, 1]] - X[t[:, 0]], X[t[:, 2]] - X[t[:, 0]], X[t[:, 3]] - X[t[:, 0]]], axis=2)
        A = S @ self.M
        a0, a1, a2 = A[:, :, 0], A[:, :, 1], A[:, :, 2]
        det = np.einsum('ij,ij->i', a0, np.cross(a1, a2))
        rt = np.sqrt(det * det + 4 * delta * delta)
        sig = 0.5 * (det + rt)
        fro = np.einsum('ijk,ijk->i', A, A)
        s23 = sig ** (2.0 / 3.0)
        E = float(np.sum(fro / (3 * s23)))
        # dE/dA
        dsig = 0.5 * (1 + det / rt)
        cof = np.stack([np.cross(a1, a2), np.cross(a2, a0), np.cross(a0, a1)], axis=2)
        c1 = (2.0 / (3 * s23))[:, None, None] * A
        c2 = (fro * (2.0 / 9.0) * sig ** (-5.0 / 3.0) * dsig)[:, None, None] * cof
        dA = c1 - c2
        dS = np.einsum('nij,nkj->nik', dA, self.M)          # dA @ M^T
        G = np.zeros((self.n, 3))
        np.add.at(G, t[:, 1], dS[:, :, 0])
        np.add.at(G, t[:, 2], dS[:, :, 1])
        np.add.at(G, t[:, 3], dS[:, :, 2])
        np.add.at(G, t[:, 0], -dS.sum(axis=2))
        return E, G[self.free].ravel()

    def solve(self, Xinit, deltas, maxiter, log=lambda *a: None):
        X = Xinit.copy()
        x = X[self.free].ravel()
        for d in deltas:
            res = minimize(lambda v: self.fg(v, X, d), x, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "maxcor": 8})
            x = res.x
            Xc = X.copy(); Xc[self.free] = x.reshape(-1, 3)
            ninv = int((self.detA(Xc) <= 0).sum())
            log(f"    δ={d:g}: E={res.fun:.0f} ninv={ninv} it={res.nit}")
            if ninv == 0 and d <= 1e-3:
                break
        X[self.free] = x.reshape(-1, 3)
        return X


def local_polish(X, tet, free_mask, X0, rounds=3, log=lambda *a: None):
    """targeted 2-ring untangle around inverted tets (restricted rest-shape energy)."""
    for r in range(rounds):
        v = signed_vol(X, tet)
        bad = np.where(v <= 0)[0]
        if len(bad) == 0:
            return X
        nodes = set(tet[bad].ravel().tolist())
        for _ in range(2):
            m = np.isin(tet, list(nodes)).any(1)
            nodes |= set(tet[m].ravel().tolist())
        rtets = np.where(np.isin(tet, list(nodes)).any(1))[0]
        sub_free = np.zeros(len(X), bool)
        sub_free[list(nodes)] = True
        sub_free &= free_mask
        en = Energy(X0, tet[rtets], sub_free)
        X = en.solve(X, [3e-2, 1e-2, 3e-3, 1e-3, 3e-4], maxiter=200,
                     log=lambda s: log("   [polish]" + s))
    return X


def smooth_bnd(B, tri, adjB, degB, qmin=0.10, rounds=12, cap=0.25):
    """intrinsic relaxation of the interpolated boundary: smooths the triangles q<qmin
    (1-ring), total deviation capped at `cap` mm. No reprojection (between the knots the
    surface is an interpolation, not data)."""
    B0 = B.copy()
    nv = len(B)
    for _ in range(rounds):
        n = np.cross(B[tri[:, 1]] - B[tri[:, 0]], B[tri[:, 2]] - B[tri[:, 0]])
        a2 = np.linalg.norm(n, axis=1)
        l2 = ((B[tri[:, 1]] - B[tri[:, 0]]) ** 2).sum(1) + \
             ((B[tri[:, 2]] - B[tri[:, 1]]) ** 2).sum(1) + \
             ((B[tri[:, 0]] - B[tri[:, 2]]) ** 2).sum(1)
        q = 2 * np.sqrt(3) * a2 / np.maximum(l2, 1e-30)
        bad_t = q < qmin
        if not bad_t.any():
            break
        bad = np.zeros(nv, bool); bad[tri[bad_t].ravel()] = True
        bad |= adjB.dot(bad.astype(float)) > 0
        avg = adjB.dot(B) / degB[:, None]
        B[bad] = 0.5 * B[bad] + 0.5 * avg[bad]
        d = B - B0
        dn = np.linalg.norm(d, axis=1)
        over = dn > cap
        if over.any():
            B[over] = B0[over] + d[over] * (cap / dn[over])[:, None]
    return B


def main(n_samples=32, maxiter=80, smooth=True):
    X0, tet, gid, vid = load()
    nb = len(vid)
    free_mask = np.ones(len(X0), bool); free_mask[vid] = False
    P0 = X0[vid]
    snaps = [P0] + [np.load(f"{ROOT}/work/reg_{tag}.npy") for _, tag in KNOTS[1:]] + [P0]
    ts = [t for t, _ in KNOTS] + [1.0]
    if smooth:
        # periodic PCHIP (no overshoot): extended knots
        text = [-0.2, -0.05] + ts + [1.4]
        aext = np.concatenate([np.stack(snaps)[[3, 4]], np.stack(snaps), np.stack(snaps)[[1]]], axis=0)
        spline = PchipInterpolator(text, aext, axis=0)
        btri = np.load(f"{ROOT}/work/reg_tri.npy")
        eB = np.vstack([btri[:, [0, 1]], btri[:, [1, 2]], btri[:, [2, 0]]])
        eB.sort(1); eB = np.unique(eB, axis=0)
        adjB = sp.coo_matrix((np.ones(len(eB)), (eB[:, 0], eB[:, 1])), shape=(nb, nb))
        adjB = (adjB + adjB.T).tocsr()
        degB = np.asarray(adjB.sum(1)).ravel()
    else:
        spline = CubicSpline(ts, np.stack(snaps), axis=0, bc_type="periodic")

    v0 = signed_vol(X0, tet)
    if np.median(v0) < 0:                       # robustness across cases: VTK may
        tet = tet[:, [0, 2, 1, 3]]              # emit a globally negative orientation
        v0 = signed_vol(X0, tet)                # (cf. pipeline.py / morph_any.py)
    assert (v0 > 0).all(), f"{int((v0 <= 0).sum())} reference tets non-positive after orientation"
    print(f"ref: {len(tet)} tets, minV={v0.min():.5f}, boundary {nb}, free {int(free_mask.sum())}")

    # graph Laplacian (harmonic init of the increments)
    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]])
    e.sort(1); e = np.unique(e, axis=0)
    n = len(X0)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    L = sp.diags(np.asarray(adj.sum(1)).ravel()) - adj
    fidx = np.where(free_mask)[0]
    Lff = L[fidx][:, fidx].tocsr()
    Lfb = L[fidx][:, vid].tocsr()

    def harmonic(db):
        out = np.empty((len(fidx), 3))
        rhs = -Lfb.dot(db)
        for k in range(3):
            out[:, k], info = spl.cg(Lff, rhs[:, k], tol=1e-8, maxiter=400)
        return out

    en = Energy(X0, tet, free_mask)
    tt = np.arange(n_samples) / n_samples
    X = X0.copy()
    all_pos = [X0.copy()]
    report = []
    for k in range(1, n_samples):
        t0 = time.time()
        B = np.asarray(spline(tt[k]))
        if smooth:
            B = smooth_bnd(B, btri, adjB, degB)
        db = B - X[vid]
        Xi = X.copy()
        Xi[fidx] += harmonic(db)
        Xi[vid] = B
        d0 = en.detA(Xi)
        ninv0 = int((d0 <= 0).sum())
        deltas = [1e-3] if ninv0 == 0 else [2e-2, 5e-3, 1e-3]
        X = en.solve(Xi, deltas, maxiter, log=print)
        v = signed_vol(X, tet)
        if (v <= 0).any():
            X = local_polish(X, tet, free_mask, X0, log=print)
            v = signed_vol(X, tet)
        ninv = int((v <= 0).sum())
        rep = dict(t=float(tt[k]), ninv_init=ninv0, ninv=ninv, minV=float(v.min()),
                   minJr=float(en.detA(X).min()), secs=round(time.time() - t0, 1))
        report.append(rep)
        print(f"[{k:2d}/{n_samples-1}] t={tt[k]:.4f} init_inv={ninv0:5d} -> inv={ninv:4d} "
              f"minV={rep['minV']:.5f} minJ/J0={rep['minJr']:.4f} ({rep['secs']}s)")
        all_pos.append(X.copy())
        np.save(f"{ROOT}/work/snap_{k:02d}.npy", X.astype(np.float32))

    np.savez_compressed(f"{ROOT}/work/morph_snapshots.npz",
                        pos=np.stack(all_pos).astype(np.float32), t=tt, vid=vid)
    with open(f"{ROOT}/work/morph_report.txt", "w") as f:
        json.dump(report, f, indent=1)

    # validation: J at the midpoints of the linear interpolation between samples (+ wrap-around)
    print("\n=== validation of inter-sample linear interpolation ===")
    worst = 0
    P = np.stack(all_pos + [all_pos[0]])
    for k in range(n_samples):
        for al in (0.25, 0.5, 0.75):
            v = signed_vol((1 - al) * P[k] + al * P[k + 1], tet)
            worst = max(worst, int((v <= 0).sum()))
    print(f"worst number of inversions at the intermediate points: {worst}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32,
         int(sys.argv[2]) if len(sys.argv) > 2 else 80,
         (sys.argv[3] != "0") if len(sys.argv) > 3 else True)
