#!/usr/bin/env python3
"""Report luminal volume against cycle phase, from a moving-boundary run's output.

The companion diagnostic to tools/build_fb_at_phase.py: before choosing the phase at which a
rigid-wall run freezes the lumen, it helps to know what the cycle actually does -- where the
volume is smallest and largest, how large the excursion is, and where a given rigid mesh falls
inside it. Reading the volumes back from the solver's own output also checks the prescribed
field independently of how it was built.

With --reference-vtu, the constant volume of a rigid mesh is located within the cycle: the tool
reports the fraction of the volume span it sits at, and the instants at which the moving lumen
passes through the same volume. A rigid reference near one extreme of the span behaves very
differently from one near the middle.

Usage:
  python3 tools/volume_cycle.py --vtu-glob 'MB/4-procs/results_*.vtu' \
      --dt 1e-3 --period 0.9 --cycle 3 --reference-vtu FB/4-procs/results_2850.vtu --out cycle.csv
"""
import argparse
import csv
import glob
import re

import numpy as np
import pyvista as pv


def lumen_volume(path):
    mesh = pv.read(path)
    sizes = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    return abs(float(np.asarray(sizes.cell_data["Volume"]).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vtu-glob", required=True, help="quoted glob of the run's result files")
    p.add_argument("--dt", type=float, required=True, help="solver time step")
    p.add_argument("--period", type=float, required=True, help="cycle period")
    p.add_argument("--cycle", type=int, default=1, help="which cycle to report (1-based)")
    p.add_argument("--ramp", type=int, default=0, help="pre-roll steps before the first cycle")
    p.add_argument("--offset", type=float, default=0.0,
                   help="phase of the cycle the run starts at, if its clock is shifted")
    p.add_argument("--reference-vtu", help="a rigid mesh to locate inside the cycle")
    p.add_argument("--out", help="write the curve to this CSV")
    a = p.parse_args()

    steps = sorted(int(re.search(r"(\d+)\.vtu$", f).group(1)) for f in glob.glob(a.vtu_glob))
    if not steps:
        raise SystemExit(f"no result files matched {a.vtu_glob!r}")
    per_cycle = a.period / a.dt
    first = a.ramp + round((a.cycle - 1) * per_cycle)
    last = a.ramp + round(a.cycle * per_cycle)
    steps = [s for s in steps if first < s <= last]
    if not steps:
        raise SystemExit(f"cycle {a.cycle} spans steps {first}-{last}, which hold no result file")

    stem = a.vtu_glob.rsplit("/", 1)[0] if "/" in a.vtu_glob else "."
    rows = []
    for step in steps:
        path = next(f for f in glob.glob(a.vtu_glob) if f.endswith(f"_{step}.vtu"))
        phase = ((step - a.ramp - (a.cycle - 1) * per_cycle) * a.dt + a.offset) % a.period
        rows.append((step, phase, lumen_volume(path)))
    rows.sort(key=lambda r: r[1])
    t = np.array([r[1] for r in rows])
    v = np.array([r[2] for r in rows])

    print(f"{len(rows)} steps over cycle {a.cycle} of {stem}")
    print(f"  smallest {v.min():.1f} at phase {t[v.argmin()]:.4f}")
    print(f"  largest  {v.max():.1f} at phase {t[v.argmax()]:.4f}")
    print(f"  span     {100 * (v.max() - v.min()) / v.min():.1f}% of the smallest")
    mean = float(np.trapz(np.r_[v, v[0]], np.r_[t, t[0] + a.period]) / a.period)
    print(f"  cycle mean {mean:.1f}")

    if a.reference_vtu:
        ref = lumen_volume(a.reference_vtu)
        tc = np.r_[t, t[0] + a.period]
        vc = np.r_[v, v[0]]
        crossings = sorted(
            float(tc[i] + (ref - vc[i]) * (tc[i + 1] - tc[i]) / (vc[i + 1] - vc[i])) % a.period
            for i in range(len(vc) - 1) if (vc[i] - ref) * (vc[i + 1] - ref) < 0)
        print(f"\nreference mesh: volume {ref:.1f}, "
              f"{100 * (ref - v.min()) / (v.max() - v.min()):.1f}% of the span, "
              f"{100 * (ref - mean) / mean:+.1f}% against the cycle mean")
        if crossings:
            print("  the moving lumen passes through that volume at phase "
                  + ", ".join(f"{c:.4f}" for c in crossings))

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "phase", "volume"])
            w.writerows(rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
