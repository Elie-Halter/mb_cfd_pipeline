"""
Cap pinning + geodesic wall taper for the prescribed morph.

Physiological/numerical rationale: the aorta is anchored at its truncation planes
(root near the heart proximally, branch/diaphragm tethering distally). The EXTENDED
morph lets the caps translate (asc ~6 mm here), which (a) invalidates the fixed-cap
mass-balance diagnostic and (b) is not the intended Dirichlet-0 cap BC (Dell'Agnello).

This module pins cap nodes to their reference position (Dirichlet 0) and tapers the
WALL displacement smoothly to zero over a geodesic distance L (mm) from the cap rings,
so the prescribed boundary stays fold-free. Interior follows via the existing harmonic
extension + rest-shape stages of pipeline.py.

Provides:
  load_geom(mesh, surf_dir)            -> X0, vid(local->vol), tri(local), cap_local(mask), gdist(mm)
  taper_weights(gdist, cap_local, L)   -> w in [0,1] per boundary node (w=0 on caps)
  pin_reg_array(P0_local, reg, w)      -> pinned/tapered boundary positions
  harmonic_extend(...)                 -> interior positions for a cheap fold proxy

CLI (cheap L-sweep on the max-motion snapshot):
  python3 morph/pin_caps.py sweep <mesh.vtu> <surf_dir> <work_dir> <snap_idx> L1 L2 ...
"""
import os, sys, heapq
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def load_geom(mesh, surf_dir):
    import pyvista as pv
    g = pv.read(mesh)
    X0 = np.asarray(g.points).astype(float)
    gid = np.asarray(g.point_data["GlobalNodeID"]).astype(int)
    order = np.argsort(gid)

    def idx_of(name):
        s = pv.read(f"{surf_dir}/{name}.vtp")
        sg = np.asarray(s.point_data["GlobalNodeID"]).astype(int)
        return order[np.searchsorted(gid[order], sg)]

    # boundary surface = the same extract_surface order used by register.load_source
    surf = g.extract_surface().triangulate()
    vid = np.asarray(surf.point_data["vtkOriginalPointIds"]).astype(np.int64)  # local->vol
    tri = surf.faces.reshape(-1, 4)[:, 1:].copy()                              # local indices
    nb = len(vid)
    v2l = -np.ones(X0.shape[0], np.int64); v2l[vid] = np.arange(nb)            # vol->local

    cap_vol = np.unique(np.concatenate([idx_of(n) for n in ["asc", "desc", "btca", "lcca", "lsa"]]))
    cap_local = np.zeros(nb, bool)
    loc = v2l[cap_vol]; cap_local[loc[loc >= 0]] = True

    # geodesic distance (mm) along the boundary graph, source = all cap nodes (dist 0)
    P0 = X0[vid]
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]); e.sort(1)
    e = np.unique(e, axis=0)
    elen = np.linalg.norm(P0[e[:, 0]] - P0[e[:, 1]], axis=1)
    nbr = [[] for _ in range(nb)]
    for (a, b), L in zip(e, elen):
        nbr[a].append((b, L)); nbr[b].append((a, L))
    gdist = np.full(nb, np.inf)
    pq = []
    for i in np.where(cap_local)[0]:
        gdist[i] = 0.0; heapq.heappush(pq, (0.0, int(i)))
    while pq:
        d, u = heapq.heappop(pq)
        if d > gdist[u]:
            continue
        for v, w in nbr[u]:
            nd = d + w
            if nd < gdist[v]:
                gdist[v] = nd; heapq.heappush(pq, (nd, v))
    return X0, vid, tri, cap_local, gdist


def taper_weights(gdist, cap_local, L):
    """smoothstep ramp 0->1 over geodesic distance [0, L]; caps forced to 0 (pinned)."""
    if L <= 0:
        w = (~cap_local).astype(float)          # hard pin (no taper)
    else:
        x = np.clip(gdist / L, 0.0, 1.0)
        w = x * x * (3 - 2 * x)                  # smoothstep
    w[cap_local] = 0.0
    return w


def harmonic_extend(X0, tet, vid, bnd_disp):
    """Cheap fold proxy: harmonic interior extension of a prescribed boundary displacement."""
    n = len(X0)
    free = np.ones(n, bool); free[vid] = False
    fidx = np.where(free)[0]
    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]]); e.sort(1)
    e = np.unique(e, axis=0)
    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    L = sp.diags(np.asarray(adj.sum(1)).ravel()) - adj
    Lff = L[fidx][:, fidx].tocsr(); Lfb = L[fidx][:, vid].tocsr()
    D = np.zeros((n, 3)); D[vid] = bnd_disp
    rhs = -Lfb.dot(bnd_disp)
    import inspect
    kw = {"rtol": 1e-7} if "rtol" in inspect.signature(spl.cg).parameters else {"tol": 1e-7}
    for k in range(3):
        D[fidx, k], _ = spl.cg(Lff, rhs[:, k], maxiter=500, **kw)
    return X0 + D


def _signed_vol(P, tet):
    a, b, c, d = P[tet[:, 0]], P[tet[:, 1]], P[tet[:, 2]], P[tet[:, 3]]
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6


def sweep(mesh, surf_dir, work_dir, snap_idx, Ls):
    import pyvista as pv
    g = pv.read(mesh); tet = g.cells_dict[10].astype(np.int64)
    X0, vid, tri, cap_local, gdist = load_geom(mesh, surf_dir)
    sgn = 1.0 if np.median(_signed_vol(X0, tet)) > 0 else -1.0
    snap = np.load(f"{work_dir}/snap_{snap_idx:02d}.npy").astype(float)
    bnd_disp_full = (snap - X0)[vid]                      # moving-cap registered boundary disp
    capd = np.linalg.norm(bnd_disp_full[cap_local], axis=1)
    print(f"snapshot #{snap_idx}: cap |d| mean={capd.mean():.2f} max={capd.max():.2f} mm ; "
          f"geodesic max={gdist[np.isfinite(gdist)].max():.1f} mm")
    print(f"{'L(mm)':>6} {'pinned_caps':>11} {'inv_tets':>9} {'minV(mm3)':>11} {'|6V|cm3':>10}  verdict")
    for L in Ls:
        w = taper_weights(gdist, cap_local, L)
        bnd_disp = w[:, None] * bnd_disp_full             # taper wall, pin caps
        P = harmonic_extend(X0, tet, vid, bnd_disp)
        v = sgn * _signed_vol(P, tet)
        ninv = int((v <= 0).sum())
        worst6V = abs(6.0 * min(0.0, v.min())) * 1e-3
        verdict = "SOLVER-SAFE" if worst6V < 1e-3 else "FOLDS>THRESH"
        capres = np.linalg.norm(bnd_disp[cap_local], axis=1).max()
        print(f"{L:6.1f} {capres:11.2e} {ninv:9d} {v.min():11.2e} {worst6V:10.2e}  {verdict}")


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        mesh, surf_dir, work_dir, si = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
        Ls = [float(x) for x in sys.argv[6:]] or [0, 4, 6, 8, 12, 16]
        sweep(mesh, surf_dir, work_dir, si, Ls)
