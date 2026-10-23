#!/usr/bin/env python3
"""
write_wall_displacement.py -- step 5 of the image-registration wall chain: periodic
interpolation in time and the solver's Prescribed_displacement file.

The wall field (n_phases evenly spaced phases) is interpolated in time by a shape-preserving
periodic PCHIP of the displacement relative to the reference phase, and written at every solver
time step of the whole run (STANDARD format: "n_nodes n_times", then per time "t" and
"gid x y z" lines with ABSOLUTE positions in cm = mm x 0.1). Sampling the whole run means the
solver never wraps the file in time (it interpolates linearly and takes the last time as the
period).

Simulation clock: t_sim = 0 is the reference phase (the mesh geometry), i.e. phase time
t_ref = ref_index T / n_phases. An optional pre-roll of --preroll seconds is prepended, during
which the wall is held at the reference geometry (blended smoothly from the mesh when the two
differ, e.g. a phase subset without the reference phase); cardiac time starts at t = preroll.
A few extra samples (--tail) beyond the last step keep the final step from wrapping.

With --flow-in, the inlet flow waveform is written on the same clock (shifted by t_ref, half-
cosine ramp from 0 during the pre-roll, sampled every dt over the run + 50 steps, NEGATIVE =
inflow, see CONVENTIONS.md section 1).

Usage:
  python3 chain_image/write_wall_displacement.py --in wall_positions_taper.npz \
      --mesh mesh-complete.mesh.vtu --period 0.9 --dt 0.001 --cycles 3 \
      --out displacement_wall.txt [--preroll 0.15] [--ref-index 10] \
      [--flow-in inlet.csv --flow-col 1 --flow-out flow_rate.txt]
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wallfield as wf

MM2CM = 0.1


def read_waveform(path, col, time_unit):
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or not (s[0].isdigit() or s[0] in "-+."):
            continue
        parts = s.replace(";", " ").replace(",", " ").split()
        rows.append((float(parts[0]), float(parts[col])))
    t, q = np.array(rows).T
    if time_unit == "ms" or (time_unit == "auto" and t.max() > 10.0):
        t = t / 1000.0
    return t - t[0], q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="wall-field .npz")
    ap.add_argument("--mesh", required=True, help="mesh-complete.mesh.vtu (mm), for the pre-roll blend")
    ap.add_argument("--period", type=float, required=True, help="cardiac cycle T (s)")
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--preroll", type=float, default=0.0, help="pre-roll duration (s)")
    ap.add_argument("--tail", type=int, default=5, help="extra samples beyond the last step")
    ap.add_argument("--ref-index", type=int, help="reference phase (default: stored in the .npz)")
    ap.add_argument("--wrap", choices=["append", "pad"], default="append",
                    help="periodic closure of the PCHIP (see wallfield.periodic_pchip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flow-in", help="inlet waveform, 2+ columns (t, ..., Q), antegrade positive")
    ap.add_argument("--flow-col", type=int, default=1, help="column of Q in --flow-in (0 = time)")
    ap.add_argument("--flow-time-unit", choices=["auto", "s", "ms"], default="auto")
    ap.add_argument("--flow-out")
    ap.add_argument("--flow-modes", type=int, default=128)
    a = ap.parse_args()

    import pyvista as pv
    pos, gid, ref_stored = wf.load_field(a.inp)
    ref = a.ref_index if a.ref_index is not None else ref_stored
    if ref is None:
        sys.exit("reference phase unknown: pass --ref-index")
    nph, n = pos.shape[:2]
    T, dt = a.period, a.dt
    t_ref = ref / nph * T
    base = pos[ref]
    f = wf.periodic_pchip(wf.phase_times(nph, T), pos - base, T, wrap=a.wrap)
    mesh_xyz = np.asarray(pv.read(a.mesh).points)[gid - 1]
    D0 = base - mesh_xyz
    print(f"[write] {nph} phases x {n} wall nodes, reference phase {ref} (t_ref = {t_ref:.4f} s); "
          f"|reference - mesh| max = {np.linalg.norm(D0, axis=1).max():.3f} mm")

    ramp_steps = int(round(a.preroll / dt))
    ramp = ramp_steps * dt
    nsteps = ramp_steps + int(round(a.cycles * T / dt))
    nfile = nsteps + a.tail
    t_all = np.arange(nfile) * dt
    tc = t_all - ramp                                       # cardiac clock (< 0 in the pre-roll)
    u = (np.maximum(tc, 0.0) + t_ref) % T
    dmax = 0.0
    with open(a.out, "w") as fo:
        fo.write(f"{n}  {nfile}\n")
        for k0 in range(0, nfile, 200):
            ks = np.arange(k0, min(k0 + 200, nfile))
            dv = f(u[ks])
            pre = tc[ks] < 0.0
            if pre.any():                                   # mesh -> reference geometry, smoothstep
                x = np.clip((tc[ks][pre] + ramp) / max(ramp, 1e-12), 0.0, 1.0)
                dv[pre] = (x * x * (3 - 2 * x) - 1.0)[:, None, None] * D0[None]
            dmax = max(dmax, float(np.linalg.norm(base + dv - mesh_xyz, axis=2).max()))
            for j, k in enumerate(ks):
                fo.write(f"{t_all[k]:.6f}\n")
                np.savetxt(fo, np.column_stack([gid, (base + dv[j]) * MM2CM]),
                           fmt="%d  %.6f  %.6f  %.6f")
    print(f"[write] {a.out}: {nfile} samples (run = {nsteps} steps of {dt} s, pre-roll {ramp_steps} "
          f"steps, cycles start at t = {ramp:.3f} s); max |x - mesh| = {dmax:.2f} mm")

    if a.flow_in:
        if not a.flow_out:
            sys.exit("--flow-in needs --flow-out")
        from scipy.interpolate import PchipInterpolator
        tq, q = read_waveform(a.flow_in, a.flow_col, a.flow_time_unit)
        if tq[-1] < T - 1e-9:                               # close the cycle at t = T
            tq, q = np.append(tq, T), np.append(q, q[0])
        cs = PchipInterpolator(tq, q)
        tt = np.arange(nsteps + 50) * dt
        qs = np.empty_like(tt)
        m = tt < ramp
        q0 = float(cs(t_ref % T))
        qs[m] = q0 * 0.5 * (1.0 - np.cos(np.pi * tt[m] / max(ramp, 1e-12)))
        qs[~m] = cs((tt[~m] - ramp + t_ref) % T)
        qs = -qs                                            # solver: positive Q = outflow
        with open(a.flow_out, "w") as fo:
            fo.write(f"{len(tt)} {a.flow_modes}\n")
            for x, y in zip(tt, qs):
                fo.write(f"{x:.6f} {y:.6f}\n")
        print(f"[write] {a.flow_out}: {len(tt)} samples, {a.flow_modes} modes, mean {qs.mean():.2f}, "
              f"min (peak inflow) {qs.min():.2f} -- must be NEGATIVE")


if __name__ == "__main__":
    main()
