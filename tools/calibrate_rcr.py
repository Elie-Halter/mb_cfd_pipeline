#!/usr/bin/env python3
"""
Option 3 — RCR Windkessel calibration to the 4D-flow MRI flow split.

Method (Les et al. 2010; Pirola et al. 2017): for N parallel outlets driven by a
common mean aortic pressure (MAP), the time-averaged flow to each outlet is

    Q_i = f_i * Q_mean        (f_i = MRI flow-split fraction)

so the total resistance that reproduces that split at the target MAP is

    R_total_i = MAP / Q_i = (MAP / Q_mean) / f_i = R_TPR / f_i

The 3-element (RCR) split keeps the structure the baseline model already used
(example values, taken from a validated FB model): every outlet had
    R_prox / R_total ~ 0.091      (characteristic impedance fraction)
    tau = R_dist * C  = 1.5 s     (diastolic decay constant, identical on all)
We preserve both and only rescale the totals -> the split changes, the pressure
level and the diastolic decay are unchanged.

Units: pressure dyne/cm^2, flow mL/s == cm^3/s, R dyne*s/cm^5, C cm^5/dyne.
"""
import argparse
import numpy as np

# ---- targets ---------------------------------------------------------------
# Example 4D-flow MRI flow split (fractions of cardiac output) -- replace with
# your cohort's measured values.
MRI_SPLIT = {           # outlet name : fraction
    "desc": 0.753,
    "bcca": 0.126,      # brachiocephalic / innominate (FB mesh calls it 'bcca')
    "lcca": 0.047,
    "lsa":  0.075,
}
MMHG_TO_DYNE = 1333.22  # 1 mmHg = 1333.22 dyne/cm^2

# Example structure constants (from a validated baseline model).
R_PROX_FRAC = 0.0909    # R_prox = R_PROX_FRAC * R_total
TAU_DIST    = 1.50      # s ; C = TAU_DIST / R_dist


def cycle_mean_flow(flow_file):
    """Time-averaged inlet flow over one cycle (trapezoid). Line 1 = header."""
    d = np.loadtxt(flow_file, skiprows=1)
    t, q = d[:, 0], d[:, 1]
    return np.trapz(q, t) / (t[-1] - t[0])


def calibrate(q_mean, map_mmhg):
    map_dyne = map_mmhg * MMHG_TO_DYNE
    r_tpr = map_dyne / q_mean          # total peripheral resistance (all outlets)
    s = sum(MRI_SPLIT.values())
    rows = {}
    for name, f in MRI_SPLIT.items():
        f_n = f / s                    # renormalise (targets sum to 1.001)
        r_total = r_tpr / f_n
        r_prox = R_PROX_FRAC * r_total
        r_dist = r_total - r_prox
        cap = TAU_DIST / r_dist
        rows[name] = dict(frac=f_n, r_total=r_total, r_prox=r_prox,
                          r_dist=r_dist, cap=cap)
    return r_tpr, map_dyne, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default=None,
                    help="inlet flow_rate_pulsatile.txt (for Q_mean)")
    ap.add_argument("--qmean", type=float, default=None,
                    help="override cycle-mean inlet flow [mL/s]")
    ap.add_argument("--map", type=float, default=93.0,
                    help="target mean arterial pressure [mmHg] "
                         "(default is an example value -- set your patient's MAP)")
    args = ap.parse_args()

    if args.qmean is not None:
        q_mean = args.qmean
    elif args.flow:
        q_mean = cycle_mean_flow(args.flow)
    else:
        raise SystemExit("need --flow or --qmean")

    r_tpr, map_dyne, rows = calibrate(q_mean, args.map)

    print(f"# Q_mean (cycle avg)   = {q_mean:8.3f} mL/s  ({q_mean*60/1000:.2f} L/min)")
    print(f"# MAP target           = {args.map:6.1f} mmHg ({map_dyne:.0f} dyne/cm^2)")
    print(f"# R_TPR (all outlets)  = {r_tpr:8.1f} dyne.s/cm^5")
    print(f"# R_prox/R_total       = {R_PROX_FRAC};  tau_dist = R_dist*C = {TAU_DIST} s")
    print()
    hdr = f"{'outlet':6} {'split%':>7} {'R_prox':>10} {'Capacitance':>13} {'R_dist':>10}"
    print(hdr); print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:6} {r['frac']*100:6.1f}% {r['r_prox']:10.1f} "
              f"{r['cap']:13.4e} {r['r_dist']:10.1f}")
    print()
    print("# --- XML RCR blocks (copy into each <Add_BC>) ---")
    for name, r in rows.items():
        print(f"  <!-- {name}: split {r['frac']*100:.1f}% -->")
        print(f"  <Proximal_resistance> {r['r_prox']:.1f} </Proximal_resistance>")
        print(f"  <Capacitance> {r['cap']:.4e} </Capacitance>")
        print(f"  <Distal_resistance> {r['r_dist']:.1f} </Distal_resistance>")


if __name__ == "__main__":
    main()
