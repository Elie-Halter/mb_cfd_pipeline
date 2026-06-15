#!/usr/bin/env python3
"""
FB vs MB comparison -- the central scientific deliverable.

For each case (rigid-wall FB, moving-boundary MB) we compute TAWSS & OSI on the wall
(via the validated gid-1 logic from hemo_indices), then bring them onto a COMMON wall
and output: global stats (mean/median FB, MB, delta, %delta), spatial correlation, and
a wall VTP carrying FB/MB/delta for the ParaView figures.

Two inter-case mapping regimes:
  - SAME mesh (FB and MB on the same mesh, e.g. FB-aniso vs MB-aniso) -> EXACT node-to-
    node comparison by GlobalNodeID. This is the scientifically clean comparison (only
    the wall motion differs, not the resolution).
  - DIFFERENT meshes (e.g. FB-iso vs MB-aniso) -> the MB field is projected onto the FB
    wall by nearest-point on the REFERENCE geometry (TAWSS/OSI are time-averaged nodal
    scalars defined on the reference wall). This is FLAGGED: the difference then mixes
    motion + near-wall resolution.

Usage:
  python3 compare_FB_MB.py \
     --fb  <fb_results_dir>  <fb_wall.vtp>  <cyc_start> <cyc_end> \
     --mb  <mb_results_dir>  <mb_wall.vtp>  <cyc_start> <cyc_end> \
     [--dt 0.001] [--out-prefix cmp_]
"""
import sys, os, argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hemo_indices import cycle_vtus, wss_indices   # validated gid-1 logic


def case_indices(results_dir, wall_vtp, s0, s1, dt):
    """(wall, TAWSS, OSI) for one case."""
    vtus = cycle_vtus(results_dir, s0, s1)
    if not vtus:
        raise SystemExit(f"no VTU in [{s0},{s1}] under {results_dir}")
    wall = pv.read(wall_vtp)
    print(f"  {os.path.basename(results_dir)}: {len(vtus)} snapshots, wall {wall.n_points} nodes")
    tawss, osi = wss_indices(vtus, wall, dt)
    return wall, tawss, osi


def same_mesh(wa, wb):
    if wa.n_points != wb.n_points:
        return False
    ga = wa.point_data.get("GlobalNodeID"); gb = wb.point_data.get("GlobalNodeID")
    if ga is not None and gb is not None:
        return np.array_equal(np.asarray(ga), np.asarray(gb))
    # no GID: geometric test (same coords)
    return np.allclose(wa.points, wb.points, atol=1e-6)


def stat_line(name, a):
    return f"{name:14} mean {np.mean(a):8.3f}  med {np.median(a):8.3f}  p95 {np.percentile(a,95):8.3f}  max {np.max(a):8.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb", nargs=4, required=True, metavar=("DIR", "WALL", "S0", "S1"))
    ap.add_argument("--mb", nargs=4, required=True, metavar=("DIR", "WALL", "S0", "S1"))
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--out-prefix", default="cmp_")
    args = ap.parse_args()

    print("FB:")
    wfb, tawss_fb, osi_fb = case_indices(args.fb[0], args.fb[1], int(args.fb[2]), int(args.fb[3]), args.dt)
    print("MB:")
    wmb, tawss_mb, osi_mb = case_indices(args.mb[0], args.mb[1], int(args.mb[2]), int(args.mb[3]), args.dt)

    # --- bring MB onto the FB wall (common base = FB) ---
    if same_mesh(wfb, wmb):
        mode = "same mesh (node-to-node, exact)"
        tawss_mb_on_fb, osi_mb_on_fb = tawss_mb, osi_mb
    else:
        mode = "DIFFERENT meshes -> MB->FB projection on the reference (mixes motion+resolution)"
        d, idx = cKDTree(wmb.points).query(wfb.points, k=1)
        print(f"  [proj] nearest-point dist MB->FB: med {np.median(d):.3g} mm, p95 {np.percentile(d,95):.3g} mm")
        tawss_mb_on_fb = tawss_mb[idx]
        osi_mb_on_fb = osi_mb[idx]

    dtawss = tawss_mb_on_fb - tawss_fb
    dosi = osi_mb_on_fb - osi_fb
    rel = np.divide(dtawss, np.maximum(tawss_fb, 1e-9)) * 100.0

    # --- VTP output for figures ---
    wfb["FB_TAWSS"] = tawss_fb;  wfb["MB_TAWSS"] = tawss_mb_on_fb
    wfb["dTAWSS"] = dtawss;      wfb["dTAWSS_pct"] = rel
    wfb["FB_OSI"] = osi_fb;      wfb["MB_OSI"] = osi_mb_on_fb;  wfb["dOSI"] = dosi
    out = args.out_prefix + "FB_vs_MB_wall.vtp"
    wfb.save(out)

    # spatial correlation FB/MB (is the pattern preserved?)
    corr = float(np.corrcoef(tawss_fb, tawss_mb_on_fb)[0, 1])

    print(f"\n--- FB vs MB COMPARISON  (mapping: {mode}) ---")
    print(stat_line("TAWSS FB", tawss_fb))
    print(stat_line("TAWSS MB", tawss_mb_on_fb))
    print(stat_line("dTAWSS", dtawss))
    print(f"{'dTAWSS rel':14} mean {np.mean(rel):+7.1f}%  med {np.median(np.abs(rel)):6.1f}% (|.|)  "
          f"spatial corr FB/MB {corr:+.3f}")
    print(stat_line("OSI FB", osi_fb))
    print(stat_line("OSI MB", osi_mb_on_fb))
    print(f"{'dOSI':14} mean {np.mean(dosi):+.4f}  med {np.median(dosi):+.4f}  "
          f"max|.| {np.max(np.abs(dosi)):.4f}")
    print(f"\n-> {out}  (FB_TAWSS, MB_TAWSS, dTAWSS, dTAWSS_pct, FB_OSI, MB_OSI, dOSI)")


if __name__ == "__main__":
    main()
