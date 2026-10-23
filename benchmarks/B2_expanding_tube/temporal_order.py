#!/usr/bin/env python3
"""
B2t -- observed temporal order of the discrete mass residual on the expanding tube.

Same geometry and prescribed motion, time step divided by 1, 2 and 4. For each run:
  err_c_max = max_{t >= 0.5} |Q_caps + Q_wall| / max|dV/dt|
  err_c_rms = rms of the same quantity
  err_b_max = max_{t >= 0.5} |Q_caps + dV/dt| / max|dV/dt|   (induced flow; spatial error)
and the observed order p = log2(err(dt) / err(dt/2)) between successive runs.

Usage:
  python3 temporal_order.py --case <case_dir> RUN_DIR:DT [RUN_DIR:DT ...] [--csv out.csv]

Reference values (patched solver, 4 ranks): err_c_max = 3.31e-6 / 4.49e-7 / 3.12e-8 at
dt = 5 / 2.5 / 1.25 ms -> p = 2.88 then 3.85 (err_c_rms: 2.97 then 4.40); err_b_max ~ 1.64 % at
every dt (p ~ 0: the induced-flow error is the spatial polygonal-volume deficit). Tightening
the fluid tolerances x50 at dt = 5 ms leaves err_c unchanged (3.306e-6), so the residual is a
discretisation error, not an iterative-solver floor. Pass: p >= 2 on the first refinement.
"""
import argparse
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
from analyze import series
from params import SETTLE


def metrics(rows):
    tt = np.array([r["t_s"] for r in rows])
    res = np.array([r["Q_caps"] + r["Q_wall"] for r in rows])
    flow = np.array([r["Q_caps"] - r["minus_dVa_dt"] for r in rows])
    scale = np.abs([r["minus_dVa_dt"] for r in rows]).max()
    est = tt >= SETTLE
    return dict(n=len(rows), err_c_max=np.abs(res[est]).max() / scale,
                err_c_rms=np.sqrt(np.mean(res[est] ** 2)) / scale,
                err_b_max=np.abs(flow[est]).max() / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("runs", nargs="+", help="RUN_DIR:DT, in decreasing dt order")
    ap.add_argument("--csv")
    a = ap.parse_args()
    res = []
    for spec in a.runs:
        rd, dt = spec.rsplit(":", 1)
        m = metrics(series(rd, os.path.abspath(a.case), float(dt)))
        m.update(run=rd, dt=float(dt)); res.append(m)
    print()
    for m in res:
        print(f"dt={m['dt']:.5f} n={m['n']:3d}  err_c_max={m['err_c_max']:.3e}  "
              f"err_c_rms={m['err_c_rms']:.3e}  err_b_max={m['err_b_max']:.3e}")
    ok = True
    print("\n# observed order p = log2(err(dt)/err(dt/2)):")
    for i in range(len(res) - 1):
        r0, r1 = res[i], res[i + 1]
        ratio = r0["dt"] / r1["dt"]
        for key in ("err_c_max", "err_c_rms", "err_b_max"):
            p = np.log(r0[key] / r1[key]) / np.log(ratio) if r1[key] > 0 else float("nan")
            print(f"  {r0['dt']:.5f} -> {r1['dt']:.5f}  {key:10s} p = {p:+.2f}")
            if i == 0 and key == "err_c_max":
                ok = p >= 2.0
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "dt", "n_snap", "err_c_max", "err_c_rms", "err_b_max"])
            for m in res:
                w.writerow([m["run"], m["dt"], m["n"], m["err_c_max"], m["err_c_rms"], m["err_b_max"]])
        print(f"-> {a.csv}")
    if len(res) > 1:
        print(f"[B2t] {'PASS' if ok else 'FAIL'} (p >= 2 on the first refinement of err_c_max; reference 2.88)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
