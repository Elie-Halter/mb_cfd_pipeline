#!/usr/bin/env python3
"""Remove the rigid-body part of a prescribed wall motion, keeping the lumen-volume curve.

For each sample k, the best-fit rotation + translation (Kabsch) that maps the wall nodes farther than
L from the caps at sample 0 onto sample k is undone; the correction is blended to zero at the caps with
a smoothstep over L (caps stay fixed). Removing the rigid part alone changes the lumen volume (the
translation against fixed caps stretches the end segments), so the remaining deformation is rescaled
per sample (secant iterations) until the lumen volume equals that of the original sample.
Used to test whether the moving-wall effect comes from bulk displacement or from wall deformation.

Inputs: displacement file (positions, svMultiPhysics format, sample 0 = initial mesh), the volume mesh
and its face VTPs (GlobalNodeID 1-based). Units: positions in the file's units (cm), --taper in the
same units.

Usage: python3 remove_rigid_motion.py disp_in.txt disp_out.txt --mesh mesh-complete.mesh.vtu
       --faces mesh-surfaces --wall wall --caps asc desc btca lcca lsa --taper 1.2 --scale 0.1
"""
import argparse, json
from itertools import islice
import numpy as np, pyvista as pv
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--mesh", required=True); ap.add_argument("--faces", required=True)
ap.add_argument("--wall", default="wall"); ap.add_argument("--caps", nargs="+", required=True)
ap.add_argument("--taper", type=float, default=1.2, help="blend length, file units (default 1.2 = 12 mm in cm)")
ap.add_argument("--scale", type=float, default=0.1, help="mesh-to-file unit factor (mm mesh, cm file: 0.1)")
a = ap.parse_args()

wall = pv.read(f"{a.faces}/{a.wall}.vtp"); gw = np.asarray(wall.point_data["GlobalNodeID"]).astype(int) - 1
caps = np.vstack([np.asarray(pv.read(f"{a.faces}/{c}.vtp").points) for c in a.caps]) * a.scale
vol = pv.read(a.mesh); tet = vol.cells.reshape(-1, 5)[:, 1:]; gv = np.asarray(vol.point_data["GlobalNodeID"]).astype(int) - 1


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - ca @ R.T


def volume(P):
    X = P[gv]; p0, p1, p2, p3 = X[tet[:, 0]], X[tet[:, 1]], X[tet[:, 2]], X[tet[:, 3]]
    return abs(np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)).sum() / 6.0)


log = []
with open(a.src) as f, open(a.dst, "w") as g:
    n, nt = map(int, f.readline().split()); g.write(f"{n} {nt}\n")
    for k in range(nt):
        t = f.readline(); raw = np.fromstring("".join(islice(f, n)), sep=" ").reshape(n, 4)
        ids = raw[:, 0].astype(int) - 1; X = np.empty((n, 3)); X[ids] = raw[:, 1:]
        if k == 0:
            X0 = X.copy(); dcap = cKDTree(caps).query(X0)[0]
            s = np.clip(dcap / a.taper, 0, 1); w = s * s * (3 - 2 * s)
            far = np.zeros(n, bool); far[gw] = True; far &= dcap > a.taper
        R, tr = kabsch(X0[far], X[far])
        Xn = (1 - w)[:, None] * X + w[:, None] * ((X - tr) @ R)
        Vt, V0 = volume(X), volume(X0)
        a0, v0, a1, v1 = 0.0, V0, 1.0, volume(Xn)
        for _ in range(3):
            if abs(v1 - v0) < 1e-12:
                break
            a0, v0, a1 = a1, v1, a1 + (Vt - v1) * (a1 - a0) / (v1 - v0); v1 = volume(X0 + a1 * (Xn - X0))
        alpha = a1 if 0 < k < nt - 1 else 1.0
        Xn = X0 + alpha * (Xn - X0)
        g.write(t); np.savetxt(g, np.c_[ids + 1, Xn[ids]], fmt=["%d", "%.9f", "%.9f", "%.9f"])
        log.append(dict(k=k, t=float(t), alpha=float(alpha), V_target=Vt, V_out=volume(Xn)))
json.dump(log, open(a.dst + ".log.json", "w"), indent=1)
print(f"wrote {a.dst} ({nt} samples); log in {a.dst}.log.json")
