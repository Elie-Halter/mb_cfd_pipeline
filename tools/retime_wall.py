#!/usr/bin/env python3
"""Shift a prescribed wall-displacement file in time (re-time the wall against the inflow).

The file is the svMultiPhysics `Prescribed_displacement` format: a header "n_nodes n_samples", then for
each sample a time line followed by n_nodes lines "global_id x y z" (positions). The last sample must
repeat the first (periodic field). New sample k = old sample (k + K) mod (n_samples - 1); times are kept.

Because the solver imposes d = x_file - x_initial_mesh, the initial mesh must have the shape of the new
sample 0, i.e. of old sample K (build it with tools/build_fb_at_phase.py).

Usage: python3 retime_wall.py displacement_in.txt displacement_out.txt --shift K
"""
import argparse
from itertools import islice
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--shift", type=int, required=True)
a = ap.parse_args()
with open(a.src) as f:
    n, nt = map(int, f.readline().split())
    T, B = [], []
    for _ in range(nt):
        T.append(f.readline().strip()); B.append("".join(islice(f, n)))
first = np.fromstring(B[0], sep=" ").reshape(n, 4); last = np.fromstring(B[-1], sep=" ").reshape(n, 4)
assert np.abs(first - last).max() < 1e-9, "the field is not periodic (last sample differs from the first)"
P = nt - 1
with open(a.dst, "w") as g:
    g.write(f"{n} {nt}\n")
    for k in range(nt):
        g.write(T[k] + "\n"); g.write(B[(k + a.shift) % P])
print(f"wrote {a.dst}: sample 0 <- old sample {a.shift % P} (t = {T[a.shift % P]})")
