#!/usr/bin/env python3
"""ALE-corrected mass-conservation V&V for the moving-boundary aortic run.

The plain mass_balance.py integrates LAB-frame cap fluxes (v_fluid . area_vec) and
assumes the inlet/outlet caps are STATIONARY (Q_in - sumQ_out = dV/dt). That holds
for the GCL dilation tube (fixed ends) but NOT for the real-aorta morph, where the
caps translate ~1 cm/cycle (asc inlet ~1.8 cm). With moving caps the correct test
uses the ALE (mesh-relative) flux through each cap:

        Q_cap_ALE = sum_tri ( (v_fluid - v_cap) . area_vec ) / 100

and the incompressible balance, with the no-slip moving wall folded into dV/dt, is

        Q_in_ALE(t) - sum_k Q_out_k_ALE(t) - dV/dt(t)  ==  0   (if mass conserved)

v_cap is the cap-node velocity from a centred finite difference of the deformed
node positions (taken from the volume VTU via GlobalNodeID = row+1). dV/dt is a
centred difference of the lumen volume V(t) computed by the divergence theorem
over the full closed deformed boundary (wall + caps):  V = (1/3) sum_tri c . area_vec.

Units: mesh mm, Velocity cm/s. area_vec [mm^2]; v.area /100 -> mL/s; V [mm^3]/1000 -> mL.
"""
import argparse, glob, re, sys
import numpy as np
import pyvista as pv


def surf_tris(vtp):
    """Return (gidx, tris) : gid-1 node indices and triangle connectivity (local)."""
    s = pv.read(vtp)
    gid = np.asarray(s.point_data["GlobalNodeID"]).astype(int) - 1
    f = s.faces.reshape(-1, 4)
    assert np.all(f[:, 0] == 3), "non-triangular cap face"
    return gid, f[:, 1:]


def cap_flux_ale(tris, P, V, Vcap):
    """ALE flux through a cap. P,V,Vcap = deformed positions[mm], fluid vel[cm/s],
    cap vel[cm/s] for that cap's nodes (local order). Returns mL/s (vtp winding)."""
    Q = 0.0
    for (i, j, k) in tris:
        area_vec = 0.5 * np.cross(P[j] - P[i], P[k] - P[i])          # mm^2
        v_tri = (V[i] + V[j] + V[k]) / 3.0 - (Vcap[i] + Vcap[j] + Vcap[k]) / 3.0
        Q += np.dot(v_tri, area_vec)
    return Q / 100.0


def lumen_volume(tets, pos):
    """V[mm^3] = sum of |tet volume| over the volume mesh (unambiguous, no winding)."""
    a = pos[tets[:, 0]]
    b = pos[tets[:, 1]] - a
    c = pos[tets[:, 2]] - a
    d = pos[tets[:, 3]] - a
    vol = np.abs(np.einsum("ij,ij->i", np.cross(b, c), d)) / 6.0
    return vol.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu-glob", required=True)
    ap.add_argument("--surf-dir", required=True)
    ap.add_argument("--dt", type=float, required=True, help="solver dt [s]")
    ap.add_argument("--inlet", default="asc")
    ap.add_argument("--outlets", default="desc,btca,lcca,lsa")
    ap.add_argument("--cycle-range", default=None, help="s0,s1 step range")
    args = ap.parse_args()

    outlets = args.outlets.split(",")
    inlet_t = surf_tris(f"{args.surf_dir}/{args.inlet}.vtp")
    outlet_t = {o: surf_tris(f"{args.surf_dir}/{o}.vtp") for o in outlets}
    files = sorted(glob.glob(args.vtu_glob),
                   key=lambda p: int(re.search(r"results_(\d+)", p).group(1)))
    steps = [int(re.search(r"results_(\d+)", p).group(1)) for p in files]
    # frame spacing (steps) -> dt_frame
    dstep = steps[1] - steps[0]
    dt_frame = dstep * args.dt

    # restrict to the cycle window (+/- 2 frames for centred differences) BEFORE loading,
    # so corrupt frames outside the analysis range (e.g. a half-written final save) are skipped
    if args.cycle_range:
        c0, c1 = [int(x) for x in args.cycle_range.split(",")]
        keep = [(p, st) for p, st in zip(files, steps)
                if c0 - 2 * dstep <= st <= c1 + 2 * dstep]
        files = [p for p, _ in keep]
        steps = [st for _, st in keep]

    # tet connectivity (constant across frames) read once
    g0 = pv.read(files[0])
    cells = g0.cells.reshape(-1, 5)
    assert np.all(cells[:, 0] == 4), "non-tet volume mesh"
    tets = cells[:, 1:]

    # load all frames' positions + velocity
    pos = {}
    vel = {}
    good = []
    for p, st in zip(files, steps):
        try:
            g = pv.read(p)
            v = np.asarray(g.point_data["Velocity"])        # cm/s
        except Exception as e:
            print(f"  skip corrupt frame {st}: {e}", file=sys.stderr)
            continue
        pos[st] = np.asarray(g.points)                      # mm
        vel[st] = v
        good.append(st)
    steps = good
    print(f"{len(files)} frames, steps {steps[0]}..{steps[-1]}, dt_frame={dt_frame:.4g} s")

    s0, s1 = (steps[0], steps[-1])
    if args.cycle_range:
        s0, s1 = [int(x) for x in args.cycle_range.split(",")]

    rows = []
    for n, st in enumerate(steps):
        if st < s0 or st > s1:
            continue
        if n == 0 or n == len(steps) - 1:
            continue                                        # need neighbours for centred diff
        stp, stn = steps[n - 1], steps[n + 1]
        P = pos[st]
        # cap velocities by centred difference of positions (cm/s : mm/s *0.1)
        def vcap(gidx):
            return (pos[stn][gidx] - pos[stp][gidx]) / (2 * dt_frame) * 0.1
        gi, ti = inlet_t
        Qin = cap_flux_ale(ti, P[gi], vel[st][gi], vcap(gi))
        Qout = 0.0
        for o in outlets:
            go, to = outlet_t[o]
            Qout += cap_flux_ale(to, P[go], vel[st][go], vcap(go))
        Vn = lumen_volume(tets, pos[stn]) / 1000.0          # mL
        Vp = lumen_volume(tets, pos[stp]) / 1000.0
        dVdt = (Vn - Vp) / (2 * dt_frame)
        rows.append((st, Qin, Qout, dVdt))

    rows = np.array(rows)
    st, Qin, Qout, dVdt = rows.T
    # sign so mean inflow > 0 and outflow sum > 0 (match physical convention)
    if Qin.mean() < 0: Qin = -Qin
    if Qout.mean() < 0: Qout = -Qout
    # 2026-09-16 -- REVERTED the 2026-07-05 'sign fix' (r = Qin - Qout + dVdt). That formula only
    # 'passed' because the production runs imposed the inlet flow with the WRONG SIGN (positive Q is
    # an OUTFLOW in svMultiPhysics): blood entered through the outlets, so 'Qin' was really an
    # outflow and the physical residual looked like -2*dV/dt. With a correctly signed inlet the
    # physical balance is  Qin - Qout - dV/dt = 0  (a shrinking lumen EXPELS fluid). Prefer
    # tools/closed_flux.py (instantaneous closed-surface sum, no dV/dt) as the primary probe.
    r = Qin - Qout - dVdt
    print(f"\n{'step':>6} {'Qin_ALE':>9} {'Qout_ALE':>9} {'dV/dt':>9} {'r':>9} {'r/Qin%':>8}")
    print("-" * 60)
    for i in range(len(st)):
        print(f"{int(st[i]):>6} {Qin[i]:9.3f} {Qout[i]:9.3f} {dVdt[i]:9.3f} "
              f"{r[i]:9.3f} {100*r[i]/max(abs(Qin).max(),1e-9):8.2f}")
    acc_inst = abs(r).max() / max(abs(Qin).max(), 1e-9)
    acc_cycle = abs(np.trapz(r, st * args.dt)) / max(np.trapz(abs(Qin), st * args.dt), 1e-9)
    print("\n=== acceptance (ALE) ===")
    print(f"  max|r|/max|Qin|        = {100*acc_inst:6.2f}%   (target < 2.0%)   "
          f"{'PASS' if acc_inst < 0.02 else 'FAIL'}")
    print(f"  |int r dt|/int|Qin| dt = {100*acc_cycle:6.2f}%   (target < 0.5%)   "
          f"{'PASS' if acc_cycle < 0.005 else 'FAIL'}")
    print(f"  cycle-mean Qin = {Qin.mean():.3f} mL/s, sum Qout = {Qout.mean():.3f} mL/s")


if __name__ == "__main__":
    main()
