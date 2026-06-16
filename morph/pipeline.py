"""
Multi-instance PIPELINE: N phase surfaces (STL) + 1 reference tetrahedral volume mesh
  ->  EXTENDED displacement file (all nodes, periodic) for svMP.

    python3 morph/pipeline.py --mesh ref.vtu --out outdir \
        --phase 0.40:phase_A.stl --phase 0.60:phase_B.stl --phase 0.95:phase_C.stl \
        [--n-samples 32] [--maxiter 80] [--t-cycle 0.974] [--scale 0.1] [--no-smooth]

Assumptions: the reference mesh corresponds to phase t=0 of the cycle; phase t given as a
cycle fraction in (0,1); STL closed (wall + caps). The vtu mesh must carry GlobalNodeID
(otherwise it is created = index+1).

Stages (all offline):
 1. Fold-free REGISTRATION of the boundary (wall+caps) onto each phase — Laplacian N-ICP +
    normal shooting. Automatic chaining: phases sorted by cyclic temporal distance to the
    reference, init = already-registered phase closest in time (otherwise reference).
 2. Volume MORPH: periodic PCHIP + smoothing of the interpolated boundary, harmonic
    extension + rest-shape minimization (J>0 barrier), local polish.
 3. VALIDATION of signed J (samples + midpoints) + EXTENDED write-out.

AUTO-TUNED parameters (and how to change them if needed):
 - h̄ (mean surface edge length) -> Laplacian scale: lambda_effective = lambda/h̄²; default
   lambda schedule (300..1), widen upward (3000..) if phases move >50% of the diameter.
 - ray_len (normal shooting) = clip(5·median(dist), 2, 10) — tracks convergence.
 - dmax (correspondence rejection) = max(2, 5·median(dist)).
 - qmin=0.15 (surface quality relaxation): raise to 0.2 if the solver remains sensitive,
   lower to 0.1 if the registration cannot converge in fidelity.
 - n_samples=32: double if the phase-to-phase displacement exceeds ~10 near-wall edges.
 - boundary smoothing cap 0.25 mm (smooth_bnd): ~half the near-wall edge length.
Robustness against stubborn folds: ray-based relax -> local harmonic patch (no reprojection);
if a phase keeps folds, the pipeline RE-RUNS its registration with a doubled lambda schedule.
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register as REG
from morph_volume import Energy, signed_vol, local_polish, smooth_bnd
import check_write_any


def cyc_dist(a, b):
    d = abs(a - b)
    return min(d, 1 - d)


def stage_register(mesh, phases, out, log=print):
    X0, vid, tri = REG.load_source(mesh)
    nb = len(vid)
    P0 = X0[vid]
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(1); edges = np.unique(e, axis=0)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(nb, nb))
    adj = (adj + adj.T).tocsr()
    deg = np.asarray(adj.sum(1)).ravel()
    hbar = float(np.linalg.norm(P0[edges[:, 0]] - P0[edges[:, 1]], axis=1).mean())
    log(f"[reg] boundary {nb} nodes, h̄={hbar:.3f}")
    np.save(f"{out}/reg_bnd_idx.npy", vid)
    np.save(f"{out}/reg_tri.npy", tri)

    done = {0.0: P0}
    order = sorted(phases, key=lambda p: cyc_dist(p[0], 0.0))
    for t, path in order:
        init_t = min(done, key=lambda u: cyc_dist(u, t))
        log(f"[reg] phase t={t} <- init t={init_t}")
        tgt = REG.Target(path)
        Y = done[init_t].copy()
        for trial, sched in enumerate([(300, 100, 30, 10, 3, 1),
                                       (1000, 600, 300, 100, 60, 30, 10, 3, 1)]):
            Y, stats = REG.register(Y, P0, tri, edges, adj, deg, tgt, hbar,
                                    lam_schedule=sched, log=log)
            if stats["folds"] == 0:
                break
            log(f"[reg] t={t}: {stats['folds']} folds -> retry with widened schedule")
        log(f"[reg] t={t}: folds={stats['folds']} dist(med/p99/max)="
            f"{stats['dist_med']:.3f}/{stats['dist_p99']:.2f}/{stats['dist_max']:.2f}")
        if stats["folds"]:
            log(f"[reg] WARNING residual folds at t={t} — the morph will flag them")
        np.save(f"{out}/reg_t{t:.4f}.npy", Y)
        done[t] = Y
    return X0, vid, tri


def stage_morph(mesh, phases, out, X0, vid, tri, n_samples, maxiter, smooth, log=print):
    import pyvista as pv
    g = pv.read(mesh)
    tet = g.cells_dict[10].astype(np.int64)
    v0 = signed_vol(X0, tet)
    if np.median(v0) < 0:
        tet = tet[:, [0, 2, 1, 3]]
    n = len(X0)
    free_mask = np.ones(n, bool); free_mask[vid] = False
    fidx = np.where(free_mask)[0]

    ts = [0.0] + sorted(p[0] for p in phases) + [1.0]
    P0 = X0[vid]
    arr = np.stack([P0] + [np.load(f"{out}/reg_t{t:.4f}.npy") for t in ts[1:-1]] + [P0])
    text = [ts[-3] - 1, ts[-2] - 1] + ts + [1 + ts[1]]
    aext = np.concatenate([arr[[-3, -2]], arr, arr[[1]]], axis=0)
    spline = PchipInterpolator(text, aext, axis=0)

    eB = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    eB.sort(1); eB = np.unique(eB, axis=0)
    nb = len(vid)
    adjB = sp.coo_matrix((np.ones(len(eB)), (eB[:, 0], eB[:, 1])), shape=(nb, nb))
    adjB = (adjB + adjB.T).tocsr()
    degB = np.asarray(adjB.sum(1)).ravel()

    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]])
    e.sort(1); e = np.unique(e, axis=0)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    L = sp.diags(np.asarray(adj.sum(1)).ravel()) - adj
    Lff = L[fidx][:, fidx].tocsr()
    Lfb = L[fidx][:, vid].tocsr()

    def harmonic(db):
        o = np.empty((len(fidx), 3))
        rhs = -Lfb.dot(db)
        import inspect; _kw = {"rtol": 1e-8} if "rtol" in inspect.signature(spl.cg).parameters else {"tol": 1e-8}
        for k in range(3):
            o[:, k], _ = spl.cg(Lff, rhs[:, k], maxiter=400, **_kw)
        return o

    en = Energy(X0, tet, free_mask)
    tt = np.arange(n_samples) / n_samples
    X = X0.copy()
    worst = 0
    for k in range(1, n_samples):
        t0 = time.time()
        B = np.asarray(spline(tt[k]))
        if smooth:
            B = smooth_bnd(B, tri, adjB, degB)
        Xi = X.copy()
        Xi[fidx] += harmonic(B - X[vid])
        Xi[vid] = B
        ninv0 = int((en.detA(Xi) <= 0).sum())
        X = en.solve(Xi, [1e-3] if ninv0 == 0 else [2e-2, 5e-3, 1e-3], maxiter, log=log)
        v = signed_vol(X, tet)
        if (v <= 0).any():
            X = local_polish(X, tet, free_mask, X0, log=log)
            v = signed_vol(X, tet)
        ninv = int((v <= 0).sum())
        worst = max(worst, ninv)
        log(f"[morph {k:2d}/{n_samples-1}] t={tt[k]:.4f} init={ninv0} -> inv={ninv} "
            f"minV={v.min():.2e} ({time.time()-t0:.0f}s)")
        np.save(f"{out}/snap_{k:02d}.npy", X.astype(np.float32))
    log(f"[morph] worst sample = {worst}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="reference tet mesh (vtu, GlobalNodeID)")
    ap.add_argument("--phase", action="append", required=True,
                    help="t_frac:path.stl (repeatable), t in (0,1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--t-cycle", type=float, default=0.974)
    ap.add_argument("--scale", type=float, default=0.1, help="mesh unit -> solver unit")
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    phases = []
    for p in a.phase:
        t, path = p.split(":", 1)
        phases.append((float(t), path))

    X0, vid, tri = stage_register(a.mesh, phases, a.out)
    stage_morph(a.mesh, phases, a.out, X0, vid, tri, a.n_samples, a.maxiter, not a.no_smooth)
    check_write_any.main(a.mesh, f"{a.out}/snap", a.n_samples,
                         f"{a.out}/displacement_all_nodes.txt", a.t_cycle, a.scale)


if __name__ == "__main__":
    main()
