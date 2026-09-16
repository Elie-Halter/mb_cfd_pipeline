#!/usr/bin/env python3
"""
Build the svMultiPhysics INLET flow file with the correct SIGN and PERIOD.

svMultiPhysics imposes a Dirichlet flow-rate BC as  v = Q(t) * profile * n_out  (set_bc.cpp,
`set_bc_dir_l`: lY = dirY * gx * nV, nV = OUTWARD nodal normal). Hence a POSITIVE Q(t) is an
OUTFLOW: to prescribe an inlet the flow rate must be NEGATIVE (every bundled
tests/cases/fluid/*/lumen_inlet.flow is negative). MRI waveforms are usually reported positive in
the antegrade direction, so a naive copy runs the whole aorta backwards (blood entering through
the descending aorta and the branches, leaving through the root) while the outlet flow split,
|Q| waveforms and r-values still look right. Symptoms: negative pressures at RCR outlets,
"pressure gauge offsets", mass-balance residuals that only close with a flipped sign.

Second trap: the solver's Fourier fit (fft.cpp) takes the period as t_last - t_first of the
file. A 30-point file sampled at 0.001..0.616 s imposes T = 0.615 s, not the true cycle. Here
the waveform is resampled at the solver time step over the WHOLE run (+50 steps) with a
periodic PCHIP, so the period is exact and no wrap-around ever happens.

Input: 2-column text (t [s or ms], Q [mL/s], antegrade POSITIVE), header lines starting with
a non-digit are skipped, ';' ',' or whitespace separators. Output: svMP temporal-values file.

Usage:
  python3 tools/make_inlet_flow.py mri_inlet.csv flow_rate.txt --t-cycle 0.974 --nsteps 3896 \
        [--dt 0.001] [--modes 128] [--time-unit auto|s|ms] [--positive-is-inflow]
Sanity print: mean and peak of the written signal (must be NEGATIVE).
"""
import argparse, re, numpy as np
from scipy.interpolate import PchipInterpolator

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--t-cycle", type=float, required=True); ap.add_argument("--nsteps", type=int, required=True)
ap.add_argument("--dt", type=float, default=0.001); ap.add_argument("--modes", type=int, default=128)
ap.add_argument("--time-unit", default="auto", choices=["auto", "s", "ms"])
ap.add_argument("--positive-is-inflow", action="store_true", default=True,
                help="input convention (default): antegrade flow is positive -> written NEGATIVE")
ap.add_argument("--negative-is-inflow", dest="positive_is_inflow", action="store_false",
                help="input already in svMP convention (inflow negative) -> written as is")
ap.add_argument("--col", type=int, default=1, help="column index of Q (0 = time)")
a = ap.parse_args()

rows = []
for line in open(a.src):
    s = line.strip()
    if not s or not re.match(r"^[-+]?\d", s): continue
    parts = re.split(r"[;,\s]+", s)
    rows.append((float(parts[0]), float(parts[a.col])))
t, q = np.array(rows).T
unit = a.time_unit
if unit == "auto": unit = "ms" if t.max() > 10.0 else "s"
if unit == "ms": t = t / 1000.0
t = t - t[0]
if t[-1] >= a.t_cycle: raise SystemExit(f"last sample t={t[-1]:.4f} >= T={a.t_cycle}: samples must lie inside one cycle")
cs = PchipInterpolator(np.append(t, a.t_cycle), np.append(q, q[0]))      # periodic wrap
tt = np.arange(a.nsteps + 50) * a.dt
qs = cs(tt % a.t_cycle)
if a.positive_is_inflow: qs = -qs
with open(a.dst, "w") as f:
    f.write(f"{len(tt)} {a.modes}\n")
    for x, y in zip(tt, qs): f.write(f"{x:.6f} {y:.6f}\n")
print(f"{a.dst}: {len(tt)} samples, dt {a.dt} s, period {a.t_cycle} s, {a.modes} modes")
print(f"  mean Q = {qs.mean():+.2f} mL/s, extreme = {qs[np.abs(qs).argmax()]:+.1f} mL/s at t = {tt[np.abs(qs).argmax()] % a.t_cycle:.3f} s")
if qs.mean() > 0:
    raise SystemExit("ERROR: mean flow is POSITIVE -> this would be an OUTFLOW in svMultiPhysics. Check --positive-is-inflow / your input sign.")
print("  OK: inflow is negative (svMultiPhysics convention).")
