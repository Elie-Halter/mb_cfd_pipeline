"""
wallfield.py -- shared helpers for the image-registration wall chain.

Wall-field file (.npz), the common currency of this directory and of
tools/build_fb_at_phase.py and tools/phase_subset_wall.py:
    positions  (n_phases, n_wall_nodes, 3)  wall-node positions in mm, one row per phase
    gid        (n_wall_nodes,)              1-based GlobalNodeID of each wall node in the mesh
    ref_phase  ()                           optional: index of the phase the mesh was built on

Phases are assumed evenly spaced over the cycle, phase k at t_k = k T / n_phases, unless a
caller passes explicit phase times.
"""
import numpy as np


# ---------------------------------------------------------------- I/O
def load_field(path):
    z = np.load(path)
    pos = np.asarray(z["positions"], dtype=float)
    gid = np.asarray(z["gid"]).astype(np.int64)
    ref = int(z["ref_phase"]) if "ref_phase" in z.files else None
    if pos.ndim != 3 or pos.shape[1] != len(gid) or pos.shape[2] != 3:
        raise ValueError(f"{path}: positions {pos.shape} inconsistent with gid {gid.shape}")
    return pos, gid, ref


def save_field(path, positions, gid, ref_phase=None):
    kw = dict(positions=np.asarray(positions, dtype=float), gid=np.asarray(gid).astype(np.int64))
    if ref_phase is not None:
        kw["ref_phase"] = int(ref_phase)
    np.savez_compressed(path, **kw)


def phase_times(n_phases, period):
    return np.arange(n_phases) / n_phases * period


# ---------------------------------------------------------------- periodic interpolation
def periodic_pchip(t_keys, values, period, wrap="pad"):
    """Shape-preserving periodic interpolant through (t_keys, values[k]); returns f(t) for any
    t (taken modulo the period). values: (n_keys, ...).
      wrap="pad"    : two keyframes are replicated on each side of the period, so the slopes
                      at the wrap see both neighbours (used for phase subsets);
      wrap="append" : the first keyframe is repeated at t = T only (closure used to write the
                      production displacement files)."""
    from scipy.interpolate import PchipInterpolator
    t_keys = np.asarray(t_keys, dtype=float)
    v = np.asarray(values, dtype=float)
    o = np.argsort(t_keys)
    t_keys, v = t_keys[o], v[o]
    shape = v.shape[1:]
    flat = v.reshape(len(t_keys), -1)
    if wrap == "pad":
        npad = min(2, len(t_keys))
        te = np.concatenate([t_keys[-npad:] - period, t_keys, t_keys[:npad] + period])
        ve = np.concatenate([flat[-npad:], flat, flat[:npad]], axis=0)
    elif wrap == "append":
        if abs(t_keys[0]) > 1e-12:
            raise ValueError("wrap='append' needs a keyframe at t = 0")
        te = np.append(t_keys, period)
        ve = np.concatenate([flat, flat[:1]], axis=0)
    else:
        raise ValueError(wrap)
    cs = PchipInterpolator(te, ve, axis=0)

    def f(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return cs(np.mod(t, period)).reshape((len(t),) + shape)
    return f


# ---------------------------------------------------------------- wall surface graph
class WallGraph:
    """Triangle topology of the wall surface in wall-field row indexing."""

    def __init__(self, wall_vtp, gid):
        import pyvista as pv
        w = pv.read(wall_vtp)
        tri_local = w.faces.reshape(-1, 4)[:, 1:]
        gn = np.asarray(w.point_data["GlobalNodeID"]).astype(np.int64)
        g2i = {int(g): i for i, g in enumerate(gid)}
        missing = [int(g) for g in gn if int(g) not in g2i]
        if missing:
            raise ValueError(f"{len(missing)} wall.vtp nodes are absent from the wall field")
        idx = np.array([g2i[int(g)] for g in gn])
        self.n = len(gid)
        self.tri = idx[tri_local]                                   # (m,3) field rows
        e = np.vstack([self.tri[:, [0, 1]], self.tri[:, [1, 2]], self.tri[:, [2, 0]]])
        e.sort(axis=1)
        self.edges = np.unique(e, axis=0)
        deg = np.bincount(self.edges.ravel(), minlength=self.n)
        if (deg == 0).any():
            raise ValueError(f"{int((deg == 0).sum())} wall-field nodes are not on wall.vtp triangles")
        nb = np.concatenate([self.edges, self.edges[:, ::-1]])
        nb = nb[np.argsort(nb[:, 0], kind="stable")]
        self.flat = nb[:, 1]
        self.deg = deg
        self.off = np.concatenate([[0], np.cumsum(deg)])

    def lap(self, D):
        """Umbrella Laplacian of a per-node vector field D (n,3)."""
        return np.add.reduceat(D[self.flat], self.off[:-1], axis=0) / self.deg[:, None] - D

    def rings(self, seed, k):
        """Node set within k edge-rings of the seed nodes (boolean mask)."""
        m = np.zeros(self.n, bool); m[list(seed)] = True
        for _ in range(k):
            m2 = m.copy()
            src = np.repeat(m, self.deg)
            m2[self.flat[src]] = True
            m = m2
        return m

    def taubin(self, D, npairs=10, lam=0.5, mu=-0.53, mask=None):
        for _ in range(npairs):
            for f in (lam, mu):
                L = self.lap(D)
                if mask is not None:
                    L = L * mask[:, None]
                D = D + f * L
        return D

    def smooth(self, D, nit, lam=0.5, mask=None):
        for _ in range(nit):
            L = self.lap(D)
            if mask is not None:
                L = L * mask[:, None]
            D = D + lam * L
        return D

    def enclosed_volume_ml(self, P):
        a, b, c = P[self.tri[:, 0]], P[self.tri[:, 1]], P[self.tri[:, 2]]
        return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0) / 1000.0


def cycle_metrics(graph, base, D, period, dt=0.002, t0=0.0, wrap="append"):
    """Scan the interpolated cycle of displacement D (n_phases, n, 3) relative to `base` at
    t0, t0 + dt, ... (mod period): worst triangle-normal ratio (< 0 = folded), folded
    triangles, max edge strain per edge."""
    f = periodic_pchip(phase_times(len(D), period), D, period, wrap=wrap)
    tri, e = graph.tri, graph.edges
    N0 = np.cross(base[tri[:, 1]] - base[tri[:, 0]], base[tri[:, 2]] - base[tri[:, 0]])
    A0 = np.einsum("ij,ij->i", N0, N0)
    el = np.linalg.norm(base[e[:, 0]] - base[e[:, 1]], axis=1)
    worst, worst_t, folded = np.inf, None, set()
    smax = np.zeros(len(e))
    for t in (np.arange(0.0, period, dt) + t0) % period:
        d = f(t)[0]
        P = base + d
        N = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
        r = np.einsum("ij,ij->i", N, N0) / A0
        if r.min() < worst:
            worst, worst_t = float(r.min()), float(t)
        folded.update(np.where(r < 0)[0].tolist())
        smax = np.maximum(smax, np.linalg.norm(d[e[:, 0]] - d[e[:, 1]], axis=1) / el)
    return dict(worst_ratio=worst, worst_t=worst_t, folded=sorted(folded), edge_strain=smax)


# ---------------------------------------------------------------- transform inversion
def invert_points(P, disp, tol=1e-3, max_iter=300):
    """Solve x + disp(x) = P for x, node by node (damped fixed point with per-node step
    control). disp: callable (n,3) -> (n,3), the displacement of the forward map
    T(x) = x + disp(x). Returns (x, residual_norm)."""
    P = np.asarray(P, dtype=float)
    x = P.copy()
    rn_prev = np.full(len(P), np.inf)
    step = np.ones((len(P), 1))
    rn = rn_prev
    for _ in range(max_iter):
        r = x + disp(x) - P
        rn = np.linalg.norm(r, axis=1)
        worse = rn > rn_prev
        step[worse] *= 0.5
        step[~worse] = np.minimum(step[~worse] * 1.2, 1.0)
        x = x - step * r
        rn_prev = rn
        if rn.max() < tol:
            break
    rn = np.linalg.norm(x + disp(x) - P, axis=1)
    return x, rn


def idw_fill(P, X, bad, k=12):
    """Replace the displacement X - P of the `bad` nodes by an inverse-distance-squared
    average of the k nearest good nodes."""
    from scipy.spatial import cKDTree
    good = ~bad
    if not bad.any():
        return X
    tree = cKDTree(P[good])
    d, i = tree.query(P[bad], k=min(k, int(good.sum())))
    d, i = np.atleast_2d(d), np.atleast_2d(i)
    w = 1.0 / np.maximum(d, 1e-6) ** 2
    dg = (X - P)[good]
    X = X.copy()
    X[bad] = P[bad] + (dg[i] * w[..., None]).sum(1) / w.sum(1)[:, None]
    return X
