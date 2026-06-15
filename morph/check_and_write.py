"""
Final checks on the morphed cycle + writing of the EXTENDED displacement file (all nodes).
- inversions / min J-ratio per sample (already in morph_report, recomputed here)
- near-wall preservation: length ratio of edges whose 2 nodes are both <1mm from the wall
- cap planarity per sample (RMS distance to the fitted plane)
- file: <out_dir>/displacement.txt (ABSOLUTE positions in cm, periodic)
Usage: python3 morph/check_and_write.py [out_txt]
"""
import sys
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree

ROOT = "."
T_CYCLE = 0.974
SCALE = 0.1


def main(out_txt=f"{ROOT}/work/displacement.txt"):
    g = pv.read(f"{ROOT}/mesh.vtu")
    X0 = np.asarray(g.points).astype(float)
    tet = g.cells_dict[10].astype(np.int64)
    gid = np.asarray(g.point_data["GlobalNodeID"]).astype(int)
    z = np.load(f"{ROOT}/work/morph_snapshots.npz")
    pos, tt, vid = z["pos"].astype(float), z["t"], z["vid"]
    order = np.argsort(gid)

    def vols(P):
        a, b, c, d = P[tet[:, 0]], P[tet[:, 1]], P[tet[:, 2]], P[tet[:, 3]]
        return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6

    v0 = vols(X0)
    wall = pv.read(f"{ROOT}/mesh-surfaces/wall.vtp")
    wgid = np.asarray(wall.point_data["GlobalNodeID"]).astype(int)
    wall_i = order[np.searchsorted(gid[order], wgid)]
    dwall = cKDTree(X0[wall_i]).query(X0)[0]
    e = np.vstack([tet[:, [0, 1]], tet[:, [0, 2]], tet[:, [0, 3]],
                   tet[:, [1, 2]], tet[:, [1, 3]], tet[:, [2, 3]]])
    e.sort(1); e = np.unique(e, axis=0)
    nw_e = e[(dwall[e[:, 0]] < 1.0) & (dwall[e[:, 1]] < 1.0)]
    l0 = np.linalg.norm(X0[nw_e[:, 0]] - X0[nw_e[:, 1]], axis=1)

    caps = {}
    for n in ["asc", "desc", "btca", "lcca", "lsa"]:
        s = pv.read(f"{ROOT}/mesh-surfaces/{n}.vtp")
        cg = np.asarray(s.point_data["GlobalNodeID"]).astype(int)
        caps[n] = order[np.searchsorted(gid[order], cg)]

    print("smpl   t      ninv  minV/V0    edges<1mm p1/p50/p99   caps RMS-plane max")
    for k in range(len(pos)):
        P = pos[k]
        v = vols(P)
        r = np.linalg.norm(P[nw_e[:, 0]] - P[nw_e[:, 1]], axis=1) / l0
        pl = 0.0
        for n, ci in caps.items():
            Q = P[ci] - P[ci].mean(0)
            pl = max(pl, float(np.linalg.svd(Q, full_matrices=False)[1][2] / np.sqrt(len(ci))))
        print(f"[{k:2d}] t={tt[k]:.4f} {int((v<=0).sum()):5d} {v.min()/np.median(v0):9.5f} "
              f"  {np.percentile(r,1):.3f}/{np.percentile(r,50):.3f}/{np.percentile(r,99):.3f}"
              f"      {pl:.3f}")

    # EXTENDED file, periodic (last sample = first one at t=T_CYCLE)
    n = len(X0)
    tt_out = np.append(tt * T_CYCLE, T_CYCLE)
    P_out = np.concatenate([pos, pos[:1]], axis=0)
    with open(out_txt, "w") as f:
        f.write(f"{n} {len(tt_out)}\nEXTENDED\n")
        np.savetxt(f, gid, fmt="%d")
        for k, t in enumerate(tt_out):
            f.write(f"{t:.6f}\n")
            blk = np.column_stack([gid, P_out[k] * SCALE])
            np.savetxt(f, blk, fmt="%d %.6f %.6f %.6f")
    print(f"\nwrote {out_txt} ({len(tt_out)} samples, {n} nodes)")


if __name__ == "__main__":
    main(*sys.argv[1:])
