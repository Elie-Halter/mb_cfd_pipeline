#!/usr/bin/env python3
"""
B1 -- free-stream analysis: max|u - U|/U and the L2 norm of div(u) at every saved step.

Usage: python3 analyze.py <run_dir> [--label patched|nogcl] [--csv out.csv]
  <run_dir> holds results_###.vtu (directly or under <n>-procs/).

Pass criterion: worst-case max|u - U|/U < 1e-4. Reference value (patched solver, 4 ranks,
200 steps): 1.69e-6, the floor of the iterative linear solver. The binary WITHOUT patch 0014
gives the same value: a uniform field makes every term of the nodal ALE residual vanish
whatever the integration geometry, so this test is necessary but does not discriminate the
DGCL fix -- B2 does.
"""
import argparse
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
import benchtools as bt
from params import U_INF, DT, PASS_MAX_REL_ERR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--label", default="patched")
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--csv", help="output CSV (default <run_dir>/b1_metrics_<label>.csv)")
    a = ap.parse_args()
    import pyvista as pv
    files = bt.find_results(a.run_dir)
    if not files:
        sys.exit(f"[B1] no results_*.vtu under {a.run_dir}")
    U = np.array([U_INF, 0.0, 0.0])
    rows, worst = [], 0.0
    for fp in files:
        g = pv.read(fp)
        u = np.asarray(g.point_data["Velocity"])[:, :3]
        rel = np.linalg.norm(u - U, axis=1).max() / U_INF
        div, vol = bt.cell_divergence(bt.result_points_cm(g), g.cells_dict[10], u)
        l2div = float(np.sqrt(np.sum(div ** 2 * vol) / vol.sum()))
        step = bt.step_of(fp)
        rows.append((step, step * a.dt, rel, l2div))
        worst = max(worst, rel)
        print(f"  step {step:4d}  t={step * a.dt:6.3f}  max|u-U|/U = {rel:.3e}   L2(div u) = {l2div:.3e} 1/s")
    out = a.csv or os.path.join(a.run_dir, f"b1_metrics_{a.label}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_s", "max_rel_vel_err", "L2_div_1_per_s"])
        w.writerows(rows)
    ok = worst < PASS_MAX_REL_ERR
    print(f"\n[B1] {a.label}: worst max|u-U|/U = {worst:.3e} over {len(files)} steps -> {out}")
    print(f"[B1] {'PASS' if ok else 'FAIL'} (< {PASS_MAX_REL_ERR:g}; reference 1.69e-6 for both binaries)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
