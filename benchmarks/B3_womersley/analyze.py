#!/usr/bin/env python3
"""
B3 -- Womersley analysis of one run: flow-rate amplitude and phase, profile L2 error.

Usage: python3 analyze.py <run_dir> [--label g1] [--csv out.csv] [--npz profiles.npz]

On the LAST complete cycle present in the results:
  - flow rate Q(t) through the mid-plane (slice z = L/2, P1 flux): fundamental fitted by least
    squares -> amplitude ratio and phase shift against the analytic Q(t)
    PASS: |amplitude - 1| < 5 % and |phase| < 5 deg;
  - area-weighted space-time L2 error of u_z in the slab |z - L/2| < 3 mm (informative; it is a
    near-wall resolution limit of isotropic P1 grids and decreases under refinement, see
    refinement.py);
  - periodicity: the same L2 on the previous cycle.
Reference values (patched solver, 4 ranks): grid g1 (~38k tets) amplitude 0.978, phase +2.6 deg,
L2 27.5 %; grid g2 (~122k) amplitude 1.015, phase +4.3 deg, L2 13.9 %.
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
from params import (R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP, PERIOD, DT, SLAB_HALF_CM,
                    PASS_AMP_REL, PASS_PHASE_DEG)
from womersley import Womersley


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--label", default="run")
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--csv")
    ap.add_argument("--npz", help="optional: save the sampled profiles (for figures)")
    a = ap.parse_args()
    import pyvista as pv
    files = bt.find_results(a.run_dir)
    if not files:
        sys.exit(f"[B3] no results_*.vtu under {a.run_dir}")
    tmax = bt.step_of(files[-1]) * a.dt
    n_cyc = int(np.floor(tmax / PERIOD + 1e-9))
    if n_cyc < 2:
        sys.exit(f"[B3] need at least 2 complete cycles (t_max = {tmax:.3f} s)")
    t_start, t_prev = (n_cyc - 1) * PERIOD, (n_cyc - 2) * PERIOD
    w = Womersley(R0_CM, LT_CM, RHO, NU, OMEGA, P_AMP)
    u_peak = np.abs(w.uhat(np.linspace(0, R0_CM, 400))).max()
    num = den = num_p = den_p = 0.0
    rows, TT, QS, QA = [], [], [], []
    prof = {"t": [], "r": None, "u_sim": [], "u_ana": []}
    for fp in files:
        step = bt.step_of(fp)
        t = step * a.dt
        if t <= t_prev + 1e-9 or t > n_cyc * PERIOD + 1e-9:
            continue
        g = pv.read(fp)
        g_cm = g.copy()
        g_cm.points = bt.result_points_cm(g)
        pts = np.asarray(g_cm.points)
        u = np.asarray(g.point_data["Velocity"])[:, :3]
        m = np.abs(pts[:, 2] - LT_CM / 2) < SLAB_HALF_CM
        r = np.hypot(pts[m, 0], pts[m, 1])
        uz = u[m, 2]
        ua = w.u(np.minimum(r, R0_CM), t)
        if t <= t_start + 1e-9:                       # previous cycle: periodicity only
            num_p += float(np.sum(r * (uz - ua) ** 2)); den_p += float(np.sum(r * ua ** 2))
            continue
        num += float(np.sum(r * (uz - ua) ** 2)); den += float(np.sum(r * ua ** 2))
        sl = g_cm.slice(normal=(0, 0, 1), origin=(0, 0, LT_CM / 2)).triangulate()
        Q = bt.tri_flux(np.asarray(sl.points), sl.faces.reshape(-1, 4)[:, 1:],
                        np.asarray(sl.point_data["Velocity"])[:, :3], (0, 0, 1))
        Qa = float(w.Q(t))
        e_snap = np.sqrt(np.mean((uz - ua) ** 2)) / u_peak
        rows.append((step, t, e_snap, Q, Qa))
        TT.append(t); QS.append(Q); QA.append(Qa)
        if prof["r"] is None:
            prof["r"] = r
        prof["t"].append(t); prof["u_sim"].append(uz); prof["u_ana"].append(ua)
        print(f"  step {step:4d} t={t:5.3f}  RMS(profile)/u_peak={e_snap:6.3%}  Q={Q:+8.3f} vs {Qa:+8.3f} cm3/s")
    if den == 0.0 or len(TT) < 4:
        sys.exit("[B3] no node/snapshot selected -- check the slab position and point units")
    E = np.sqrt(num / den)
    E_prev = np.sqrt(num_p / den_p) if den_p > 0 else float("nan")
    TT, QS, QA = map(np.asarray, (TT, QS, QA))
    amp_s, ph_s = bt.fit_sincos(TT, QS, OMEGA)
    amp_a, ph_a = bt.fit_sincos(TT, QA, OMEGA)
    ratio = amp_s / amp_a
    dphi = (np.degrees(ph_s - ph_a) + 180) % 360 - 180
    out = a.csv or os.path.join(a.run_dir, f"b3_metrics_{a.label}.csv")
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "t_s", "rms_profile_over_upeak", "Q_sim", "Q_ana"])
        wr.writerows(rows)
    if a.npz:
        np.savez(a.npz, r=prof["r"], t=np.asarray(prof["t"]), u_sim=np.asarray(prof["u_sim"]),
                 u_ana=np.asarray(prof["u_ana"]))
    ok = abs(ratio - 1) < PASS_AMP_REL and abs(dphi) < PASS_PHASE_DEG
    print(f"\n[B3] {a.label} -> {out}  (cycle {n_cyc})")
    print(f"[B3] periodicity: L2 cycle {n_cyc - 1} = {E_prev:.3%} vs cycle {n_cyc} = {E:.3%}")
    print(f"[B3] flow-rate amplitude sim/analytic = {ratio:.4f} ({'PASS' if abs(ratio - 1) < PASS_AMP_REL else 'FAIL'} +-5 %)")
    print(f"[B3] flow-rate phase = {dphi:+.2f} deg ({'PASS' if abs(dphi) < PASS_PHASE_DEG else 'FAIL'} < 5 deg)")
    print(f"[B3] area-weighted L2 profile error (slab) = {E:.2%} (informative; decreases under refinement)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
