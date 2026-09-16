#!/usr/bin/env python3
"""
Instantaneous closed-boundary mass check and SIGNED cap fluxes (the direct probe).

For each saved VTU (deformed geometry), integrates the P1 flux v.n over every boundary face
with OUTWARD normals ORIENTED GEOMETRICALLY (cap centroid -> nearby wall points), so it does
not rely on the .vtp winding (gmsh/SimVascular caps can be wound inconsistently). For an
incompressible fluid the closed sum  sum_caps Q_out + Q_wall  must vanish at every instant
(divergence theorem, exact for P1 on tets) -- no volume time-derivative involved, unlike
mass_balance*.py whose centred dV/dt is noisy on a fast wall. Also prints the signed cap
fluxes: the INLET must be NEGATIVE (inflow) at systole -- if it is positive the run is backwards
(see tools/make_inlet_flow.py).

Usage:
  python3 tools/closed_flux.py <run/4-procs> <mesh-surfaces dir> <s0> <s1> [--every 10] [--wall wall] \
        [--caps asc,desc,btca,lcca,lsa] [--inlet asc]
"""
import argparse, glob, re, numpy as np, pyvista as pv
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("surf"); ap.add_argument("s0", type=int); ap.add_argument("s1", type=int)
ap.add_argument("--every", type=int, default=10); ap.add_argument("--wall", default="wall")
ap.add_argument("--caps", default="asc,desc,btca,lcca,lsa"); ap.add_argument("--inlet", default="asc")
a = ap.parse_args()
caps = [c for c in a.caps.split(",") if c]
def surf(n):
    s = pv.read(f"{a.surf}/{n}.vtp"); return np.asarray(s.point_data["GlobalNodeID"]).astype(int) - 1, s.faces.reshape(-1, 4)[:, 1:]
faces = {n: surf(n) for n in [a.wall] + caps}
first = sorted(glob.glob(f"{a.run}/results_*.vtu"), key=lambda f: int(re.search(r"results_(\d+)", f).group(1)))[0]
X0 = np.asarray(pv.read(first).points); Wp = X0[faces[a.wall][0]]
def area_vec(X, g, t): p = X[g][t]; return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]) / 2.0
sgn = {a.wall: 1.0}
for n in caps:                                     # orient cap normals OUTWARD geometrically
    g, t = faces[n]; nraw = area_vec(X0, g, t).sum(0); nraw /= np.linalg.norm(nraw)
    c = X0[g].mean(0); d = np.linalg.norm(Wp - c, axis=1); near = Wp[(d > 3) & (d < 20)]
    into = near.mean(0) - c; into /= np.linalg.norm(into); sgn[n] = -1.0 if np.dot(nraw, into) > 0 else 1.0
w_g, w_t = faces[a.wall]; V0 = (X0[w_g][w_t].mean(1) * area_vec(X0, w_g, w_t)).sum() / 3.0
if V0 < 0: sgn[a.wall] = -1.0
print("outward-normal sign per face (from geometry):", sgn)
def flux(X, V, n): g, t = faces[n]; return sgn[n] * (V[g][t].mean(1) * area_vec(X, g, t)).sum() / 100.0   # mL/s
hdr = "%6s " % "step" + " ".join("%8s" % n for n in caps) + " %9s %9s %12s" % ("caps_tot", "wall", "closed_sum")
print(hdr); worst = 0.0; qin_peak = 0.0
for s in range(a.s0, a.s1 + 1, a.every):
    f = f"{a.run}/results_{s}.vtu"
    try: m = pv.read(f)
    except Exception: continue
    X = np.asarray(m.points); V = np.asarray(m.point_data["Velocity"])
    q = {n: flux(X, V, n) for n in caps}; qw = flux(X, V, a.wall); tot = sum(q.values()) + qw
    worst = max(worst, abs(tot)); qin_peak = max(qin_peak, abs(q[a.inlet]))
    print("%6d " % s + " ".join("%+8.1f" % q[n] for n in caps) + " %+9.1f %+9.1f %+12.2f" % (sum(q.values()), qw, tot))
print(f"\nmax |closed sum| = {worst:.3f} mL/s = {100*worst/max(qin_peak,1e-9):.3f} % of the peak inlet flow ({qin_peak:.0f} mL/s)")
print("Direction check: the inlet flux above must be NEGATIVE at systole (inflow). Positive = the run is backwards.")
