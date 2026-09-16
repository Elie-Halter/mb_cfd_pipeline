#!/usr/bin/env python3
"""
Pre-flight: refuse to run if the inlet flow file would impose an OUTFLOW.

svMultiPhysics: v = Q * profile * n_out  ->  inlet flow rate must be NEGATIVE (see
tools/make_inlet_flow.py). Also warns when the file period (t_last - t_first, fft.cpp) differs
from the cardiac cycle by more than 0.5 %.

Usage: python3 tools/check_inlet_sign.py flow_rate.txt --t-cycle 0.974 [--nsteps N]
Exit code 0 = OK, 2 = would run backwards, 3 = period mismatch.
"""
import argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("flow"); ap.add_argument("--t-cycle", type=float, required=True)
ap.add_argument("--nsteps", type=int, default=None); a = ap.parse_args()
d = np.loadtxt(a.flow, skiprows=1); t, q = d[:, 0], d[:, 1]
_tr = getattr(np, "trapezoid", None) or np.trapz
qmean = _tr(q, t) / (t[-1] - t[0]); T_file = t[-1] - t[0]
print(f"{a.flow}: {len(t)} samples, span {t[0]:.3f}..{t[-1]:.3f} s (fft period = {T_file:.4f} s), mean Q = {qmean:+.2f} mL/s")
rc = 0
if qmean > 0:
    print("ERROR: mean inlet flow is POSITIVE -> the solver will push fluid OUT through the inlet (backwards aorta)."); rc = 2
run_span = (a.nsteps or 0) * 0.001
if T_file < run_span * 0.999 and abs(T_file - a.t_cycle) / a.t_cycle > 0.005:
    print(f"ERROR: file period {T_file:.4f} s differs from T_cycle {a.t_cycle} s by {100*abs(T_file-a.t_cycle)/a.t_cycle:.1f} % "
          f"(the solver uses t_last - t_first as the period). Resample with tools/make_inlet_flow.py."); rc = rc or 3
if rc == 0: print("OK: inflow negative, period consistent.")
raise SystemExit(rc)
