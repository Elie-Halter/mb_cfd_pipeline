"""
Cap-pinned morph driver: reuse an EXISTING registration (reg_*.npy) and produce a
pinned + wall-tapered EXTENDED displacement, WITHOUT re-running the ICP registration.

caps -> anchored (pinned) ; wall displacement tapered to the anchor over a geodesic
distance L (mm) from the cap rings (see pin_caps.py for the rationale). The cap anchor
is re-imposed AFTER smooth_bnd so the caps stay exactly on their prescribed motion.

Anchor per cap:
  - default: Dirichlet 0 (cap frozen at its reference position) -- Dell'Agnello-style.
  - --rigid-cap NAME: that cap follows its mean RIGID TRANSLATION instead of freezing.
    The cap stays a flat disk that translates with the bulk motion (e.g. the ascending
    root pulled by the heart), which is physiological AND keeps mass_balance exact: a
    rigid cap has a uniform mesh velocity, so its relative flux is Q_abs - v_cap.A_vec,
    a single noise-free term (pair with tools/mass_balance.py --rigid-cap NAME).
Each boundary node is anchored toward its NEAREST cap's anchor; w=smoothstep(geo/L).

Usage:
  python3 morph/run_pinned_morph.py --mesh ref.vtu --surf <surf_dir> --reg-dir <work> \
      --out <outdir> --L 12 --phase 0.40 ... [--rigid-cap asc] \
      [--n-samples 32] [--maxiter 80] [--t-cycle 0.974] [--scale 0.1] [--no-smooth]
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from morph_volume import Energy, signed_vol, local_polish, smooth_bnd
import check_write_any
import pin_caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--surf", required=True, help="mesh-surfaces dir (wall/asc/desc/btca/lcca/lsa .vtp)")
    ap.add_argument("--reg-dir", required=True, help="dir with reg_bnd_idx/reg_tri/reg_t*.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=float, default=12.0, help="taper geodesic length (mm)")
    ap.add_argument("--rigid-cap", action="append", default=[],
                    help="cap name(s) anchored to their mean rigid TRANSLATION instead of frozen at 0")
    ap.add_argument("--phase", action="append", type=float, required=True, help="phase t in (0,1)")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--t-cycle", type=float, default=0.974)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    smooth = not a.no_smooth
    log = print

    import pyvista as pv
    g = pv.read(a.mesh)
    X0 = np.asarray(g.points).astype(float)
    tet = g.cells_dict[10].astype(np.int64)
    if np.median(signed_vol(X0, tet)) < 0:
        tet = tet[:, [0, 2, 1, 3]]

    vid = np.load(f"{a.reg_dir}/reg_bnd_idx.npy")
    tri = np.load(f"{a.reg_dir}/reg_tri.npy")
    P0 = X0[vid]; nb = len(vid)

    # cap mask (local) + geodesic distance from cap rings, using the SAME vid/tri as the reg arrays
    gid = np.asarray(g.point_data["GlobalNodeID"]).astype(int); order = np.argsort(gid)
    def idx_of(n):
        s = pv.read(f"{a.surf}/{n}.vtp"); sg = np.asarray(s.point_data["GlobalNodeID"]).astype(int)
        return order[np.searchsorted(gid[order], sg)]
    import heapq
    CAPS = ["asc", "desc", "btca", "lcca", "lsa"]
    v2l = -np.ones(len(X0), np.int64); v2l[vid] = np.arange(nb)
    cap_masks = {}
    for n in CAPS:
        m = np.zeros(nb, bool); loc = v2l[idx_of(n)]; m[loc[loc >= 0]] = True
        cap_masks[n] = m
    cap_local = np.zeros(nb, bool)
    for m in cap_masks.values():
        cap_local |= m

    # boundary graph (edge-length weighted) + per-cap geodesic
    e0 = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]); e0.sort(1); e0 = np.unique(e0, axis=0)
    elen = np.linalg.norm(P0[e0[:, 0]] - P0[e0[:, 1]], axis=1)
    nbr = [[] for _ in range(nb)]
    for (u, v), Lw in zip(e0, elen):
        nbr[u].append((v, Lw)); nbr[v].append((u, Lw))

    def dijkstra(srcs):
        d = np.full(nb, np.inf); pq = []
        for i in srcs:
            d[i] = 0.0; heapq.heappush(pq, (0.0, int(i)))
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > d[u]: continue
            for v, w in nbr[u]:
                nd = dd + w
                if nd < d[v]: d[v] = nd; heapq.heappush(pq, (nd, v))
        return d

    dcap = np.stack([dijkstra(np.where(cap_masks[n])[0]) for n in CAPS])  # (ncap, nb)
    nearest = np.argmin(dcap, axis=0)                                     # index into CAPS
    gdist = dcap.min(axis=0)
    w_taper = pin_caps.taper_weights(gdist, cap_local, a.L)
    rigid = set(a.rigid_cap)
    log(f"[pin] caps={int(cap_local.sum())} nodes ; rigid={sorted(rigid) or 'none'} ; "
        f"taper L={a.L} mm ; free(w=1): {int((w_taper >= 0.999).sum())}/{nb}")

    # per-phase anchored+tapered boundary: node anchored toward its NEAREST cap's anchor
    #   anchor = mean rigid translation of that cap if --rigid-cap, else 0 (frozen)
    phases = sorted(a.phase)
    def pin(reg):
        disp = reg - P0
        anchor = np.zeros_like(disp)
        for ci, n in enumerate(CAPS):
            if n in rigid:
                anchor[nearest == ci] = disp[cap_masks[n]].mean(0)
        return P0 + anchor + w_taper[:, None] * (disp - anchor)
    regs = {t: pin(np.load(f"{a.reg_dir}/reg_t{t:.4f}.npy")) for t in phases}

    ts = [0.0] + phases + [1.0]
    arr = np.stack([P0] + [regs[t] for t in phases] + [P0])
    text = [ts[-3] - 1, ts[-2] - 1] + ts + [1 + ts[1]]
    aext = np.concatenate([arr[[-3, -2]], arr, arr[[1]]], axis=0)
    spline = PchipInterpolator(text, aext, axis=0)

    # boundary adjacency for smoothing
    eB = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]); eB.sort(1); eB = np.unique(eB, axis=0)
    adjB = sp.coo_matrix((np.ones(len(eB)), (eB[:, 0], eB[:, 1])), shape=(nb, nb)); adjB = (adjB + adjB.T).tocsr()
    degB = np.asarray(adjB.sum(1)).ravel()

    # interior harmonic operator
    n = len(X0)
    free_mask = np.ones(n, bool); free_mask[vid] = False
    fidx = np.where(free_mask)[0]
    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]]); e.sort(1); e = np.unique(e, axis=0)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)); adj = (adj + adj.T).tocsr()
    Lap = sp.diags(np.asarray(adj.sum(1)).ravel()) - adj
    Lff = Lap[fidx][:, fidx].tocsr(); Lfb = Lap[fidx][:, vid].tocsr()
    import inspect
    cgkw = {"rtol": 1e-8} if "rtol" in inspect.signature(spl.cg).parameters else {"tol": 1e-8}
    def harmonic(db):
        o = np.empty((len(fidx), 3)); rhs = -Lfb.dot(db)
        for k in range(3):
            o[:, k], _ = spl.cg(Lff, rhs[:, k], maxiter=400, **cgkw)
        return o

    en = Energy(X0, tet, free_mask)
    tt = np.arange(a.n_samples) / a.n_samples
    X = X0.copy(); worst = 0
    for k in range(1, a.n_samples):
        t0 = time.time()
        B = np.asarray(spline(tt[k]))
        cap_tgt = B[cap_local].copy()                      # interpolated cap motion (rigid/frozen)
        if smooth:
            B = smooth_bnd(B, tri, adjB, degB)
        B[cap_local] = cap_tgt                             # re-impose exact cap anchor after smoothing
        Xi = X.copy()
        Xi[fidx] += harmonic(B - X[vid])
        Xi[vid] = B
        ninv0 = int((en.detA(Xi) <= 0).sum())
        X = en.solve(Xi, [1e-3] if ninv0 == 0 else [2e-2, 5e-3, 1e-3], a.maxiter, log=log)
        X[vid] = B                                         # keep boundary (incl. pinned caps) exact
        v = signed_vol(X, tet)
        if (v <= 0).any():
            X = local_polish(X, tet, free_mask, X0, log=log); X[vid] = B; v = signed_vol(X, tet)
        ninv = int((v <= 0).sum()); worst = max(worst, ninv)
        log(f"[pmorph {k:2d}/{a.n_samples-1}] t={tt[k]:.4f} init={ninv0} -> inv={ninv} "
            f"minV={v.min():.2e} ({time.time()-t0:.0f}s)")
        np.save(f"{a.out}/snap_{k:02d}.npy", X.astype(np.float32))
    log(f"[pmorph] worst sample = {worst}")

    # per-cap check across the cycle: |disp| (rigid caps move, frozen ~0) and PLANARITY
    # residual = deviation from the cap's own rigid translation (should be ~0 => stays a flat disk)
    log("[pin] per-cap motion across cycle (mm):")
    snaps = [np.load(f"{a.out}/snap_{k:02d}.npy") for k in range(1, a.n_samples)]
    for n in CAPS:
        m = cap_masks[n]
        dmax = 0.0; planar = 0.0
        for S in snaps:
            d = S[vid][m] - P0[m]
            dmax = max(dmax, float(np.linalg.norm(d, axis=1).max()))
            planar = max(planar, float(np.linalg.norm(d - d.mean(0), axis=1).max()))
        tag = "RIGID" if n in rigid else "frozen"
        log(f"    {n:5s} [{tag}]: max|disp|={dmax:5.2f}  max non-rigid residual={planar:.2e} (planarity)")
    check_write_any.main(a.mesh, f"{a.out}/snap", a.n_samples,
                         f"{a.out}/displacement_all_nodes.txt", a.t_cycle, a.scale)


if __name__ == "__main__":
    main()
