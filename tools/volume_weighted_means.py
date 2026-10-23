#!/usr/bin/env python3
"""Lumen-mean flow descriptors, nodal and VOLUME-WEIGHTED, on result VTUs.

Plain nodal means (np.mean over mesh nodes) over-weight the wall, where tetrahedral meshes are finer.
Here each node gets the weight w_i = 1/4 of the volume of the tetrahedra that share it, computed on the
geometry of each VTU (a moving-wall VTU carries the deformed coordinates).

Reported per VTU: mean speed, mean |H| (H = u.omega), mean |LNH| and signed mean LNH
(LNH = u.omega / (|u||omega|)), and the volume share with LNH > +thr and < -thr; nodal versions alongside.

Usage: python3 volume_weighted_means.py results_1530.vtu [more.vtu ...] [--thr 0.6] [--csv out.csv]
"""
import argparse, csv
import numpy as np, pyvista as pv


def metrics(path, thr=0.6):
    m = pv.read(path)
    v = np.asarray(m.point_data["Velocity"], float)
    w = (np.asarray(m.point_data["Vorticity"], float) if "Vorticity" in m.point_data
         else np.asarray(m.compute_derivative(scalars="Velocity", vorticity=True).point_data["vorticity"]))
    vn = np.linalg.norm(v, axis=1); H = (v * w).sum(1); lnh = H / (vn * np.linalg.norm(w, axis=1) + 1e-9)
    tet = m.cells_dict[10]; X = np.asarray(m.points, float)
    vt = np.abs(np.einsum("ij,ij->i", np.cross(X[tet[:, 1]] - X[tet[:, 0]], X[tet[:, 2]] - X[tet[:, 0]]),
                          X[tet[:, 3]] - X[tet[:, 0]])) / 6.0
    wn = np.bincount(tet.ravel(), weights=np.repeat(vt / 4.0, 4), minlength=len(X)); wn /= wn.sum()
    return dict(meanV=float(vn.mean()), meanAbsH=float(np.abs(H).mean()), meanAbsLNH=float(np.abs(lnh).mean()),
                meanLNH=float(lnh.mean()),
                meanV_vol=float((wn * vn).sum()), meanAbsH_vol=float((wn * np.abs(H)).sum()),
                meanAbsLNH_vol=float((wn * np.abs(lnh)).sum()), meanLNH_vol=float((wn * lnh).sum()),
                fracPos_vol=float(wn[lnh > thr].sum()), fracNeg_vol=float(wn[lnh < -thr].sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vtu", nargs="+"); ap.add_argument("--thr", type=float, default=0.6); ap.add_argument("--csv")
    a = ap.parse_args()
    rows = [dict(file=p, **metrics(p, a.thr)) for p in a.vtu]
    for r in rows:
        print(f"{r['file']}: v {r['meanV']:.3f} -> {r['meanV_vol']:.3f} (vol) | |H| {r['meanAbsH']:.3g} -> "
              f"{r['meanAbsH_vol']:.3g} | |LNH| {r['meanAbsLNH']:.3f} -> {r['meanAbsLNH_vol']:.3f} | "
              f"LNH>+{a.thr}: {100 * r['fracPos_vol']:.1f}% <-{a.thr}: {100 * r['fracNeg_vol']:.1f}%")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
