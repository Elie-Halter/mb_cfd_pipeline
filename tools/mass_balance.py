#!/usr/bin/env python3
"""
Rigorous mass-conservation V&V for the aortic ALE (moving-boundary) run.

For EACH saved timestep VTU (a snapshot of the DEFORMED mesh) it:
  * integrates Q_in through the INLET cap `asc` -- MEASURED from the velocity
    field, never reconstructed as the sum of the outlets;
  * integrates Q_out through every outlet cap (desc, btca/bcca, lcca, lsa);
  * computes the lumen volume V(t) of the deformed mesh and dV/dt by a centered
    finite difference;
  * forms the instantaneous mass residual.

Why measure the inlet directly?  `tools/extract_flowsplit_FB.py` reports the
"inflow" as the sum of the outlet fluxes (its `total = sum(q_mean.values())`).
That is *circular* for a conservation check: by construction it can never expose
a leak. Here Q_in is its own field integral over `asc`, so Q_in - sum(Q_out)
- dV/dt is a genuine, independent test of mass conservation.

------------------------------------------------------------------------------
Sign / balance convention  (verified on the GCL dilation case, see below)
------------------------------------------------------------------------------
We use PHYSICAL through-cap flows: Q>0 means fluid crosses the cap in the
physiological direction (in at `asc`, out at each outlet). For the REAL aorta the
wall moves and the no-slip fluid follows it (fluid velocity at the wall = wall
velocity != 0), so the wall's fluid flux equals the rate of volume change. The
lab-frame incompressible (div u = 0) mass balance over the deforming lumen then
reads, with the wall flux folded in as dV/dt:

        Q_in(t) - sum_k Q_out_k(t) = dV/dt(t)            (1)

i.e. during systole the wall expands (dV/dt>0) so more flows in than out. The
instantaneous residual is therefore

        r(t) = Q_in(t) - sum_k Q_out_k(t) - dV/dt(t)     (2)

(NB: this differs from the GCL *tube* test in gcl_test/measure_gcl.py, whose
wall has ZERO fluid velocity, so there the caps balance to 0 and dV/dt is a
purely geometric mesh term. The aorta wall carries real fluid, hence dV/dt
appears in Eq.(1). Verified by measuring the wall flux directly on both cases.)

Each cap flux is integrated with the area-vector / deformed-node formula and a
GlobalNodeID(=row+1) mapping -- identical to extract_flowsplit_FB.tri_flux:

        Q_raw [mL/s] = sum_tri (v_tri . area_vec) / 100

with v_tri [cm/s] the per-triangle mean velocity and area_vec [mm^2] the vector
area built from the DEFORMED cap-node positions taken out of the volume (the
.vtp stores reference positions, so we never use the .vtp coordinates and never
do a positional KDTree -- both are wrong under boundary motion). Q_raw carries
the .vtp winding; we map it to the PHYSICAL sign by each cap's cycle mean: a true
inlet/outlet has a definite mean direction, so we flip so that mean(Q_in)>0
(inflow positive) and each mean(Q_out)>0 (outflow positive). This is robust
because all caps share a consistent .vtp winding and the net flow is pulsatile
with a definite direction.

Acceptance numbers printed at the end:
  * max|r| / max|Q_in|              (target < 2%)   -- worst instantaneous leak
  * |int r dt| / int|Q_in| dt       (target < 0.5%) -- net cycle leak

Usage:
  python3 mass_balance.py \
      --vtu-glob '/path/to/MB_run/*-procs/results_*.vtu' \
      --surf-dir /path/to/mesh/mesh-surfaces \
      --dt 0.001 \
      [--inlet asc] [--outlets desc,btca,lcca,lsa] \
      [--cycle-range LO,HI]      # restrict to step numbers in [LO,HI]

Notes / assumptions:
  * Inlet defaults to `asc`; outlets default to desc,btca,lcca,lsa. The
    brachiocephalic cap is `bcca` on M2/FB meshes and `btca` on MB meshes --
    handled by ALIASES (same as extract_flowsplit_FB).
  * --dt is the SOLVER time step (s). The spacing between *saved* VTUs is
    inferred from the step numbers in the file names (Increment_in_saving_VTK),
    so dV/dt is correct even when only every Nth step is written.
  * Units: mesh coords mm, Velocity cm/s, Q in mL/s, V in mm^3 -> dV/dt scaled
    to mL/s (mm^3/s = 1e-3 mL/s ... no: 1 mL = 1 cm^3 = 1000 mm^3, so
    mm^3/s -> mL/s is /1000). See _volume_mm3 and dVdt handling below.
"""
import os
import glob
import argparse
import numpy as np
import pyvista as pv

# Logical cap names. Inlet measured separately; never inferred from outlets.
DEFAULT_INLET = "asc"
DEFAULT_OUTLETS = ["desc", "btca", "lcca", "lsa"]
# Same vessel named bcca (M2/FB meshes) or btca (MB pipeline meshes).
ALIASES = {"bcca": ["bcca", "btca"], "btca": ["btca", "bcca"]}


def find_vtp(surf_dir, name):
    """Resolve a logical cap name to an existing .vtp, honouring bcca<->btca."""
    for cand in ALIASES.get(name, [name]):
        p = os.path.join(surf_dir, f"{cand}.vtp")
        if os.path.exists(p):
            return p
    raise SystemExit(
        f"no .vtp for cap '{name}' (tried {ALIASES.get(name, [name])}) in {surf_dir}"
    )


def volume_row_for_gid(vol, surf_gid):
    """Row in `vol` for each surface GlobalNodeID -- deformation-proof.

    Replicated from tools/extract_flowsplit_FB.volume_row_for_gid: the svMP
    results VTU carries no GlobalNodeID but keeps the reference-mesh order with
    GlobalNodeID == row+1, so row = gid-1. NEVER a positional KDTree (wrong
    exactly where the boundary moves)."""
    vg = vol.point_data.get("GlobalNodeID")
    if vg is not None:
        vg = np.asarray(vg).astype(np.int64)
        m = np.full(int(vg.max()) + 1, -1, dtype=np.int64)
        m[vg] = np.arange(len(vg))
        idx = m[surf_gid]
    else:
        idx = surf_gid - 1
    if idx.min() < 0 or idx.max() >= vol.n_points:
        return None
    return idx


def tri_flux_raw(surf, vol):
    """Q [mL/s] through a cap in the .vtp's RAW winding.

    Identical formula to extract_flowsplit_FB.tri_flux: deformed cap positions
    from the volume (gid-1) give the time-varying area; per-triangle mean
    velocity dotted with the vector area; /100 converts cm/s*mm^2 -> mL/s."""
    surf = surf.triangulate()
    sg = np.asarray(surf.point_data["GlobalNodeID"]).astype(np.int64)
    idx = volume_row_for_gid(vol, sg)
    if idx is None:
        raise SystemExit("cap GlobalNodeID out of volume -- mesh/result mismatch")
    P = vol.points[idx]                                   # DEFORMED cap positions (mm)
    vel = np.asarray(vol.point_data["Velocity"])[idx]     # cm/s at the same nodes
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    Q = 0.0
    for i, j, k in faces:
        area_vec = 0.5 * np.cross(P[j] - P[i], P[k] - P[i])   # mm^2
        v_tri = (vel[i] + vel[j] + vel[k]) / 3.0              # cm/s
        Q += np.dot(v_tri, area_vec)
    return Q / 100.0                                          # -> mL/s


def cap_meandisp_area(surf, vol):
    """Cap mean mesh displacement (from the Displacement field) + raw vector area.

    For a RIGID cap the mesh velocity is uniform, so the relative flux correction is
    v_mesh . A_vec (a single noise-free term), with v_mesh = d(meanDisp)/dt."""
    surf = surf.triangulate()
    sg = np.asarray(surf.point_data["GlobalNodeID"]).astype(np.int64)
    idx = volume_row_for_gid(vol, sg)
    P = vol.points[idx]
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    Araw = np.zeros(3)
    for i, j, k in faces:
        Araw += 0.5 * np.cross(P[j] - P[i], P[k] - P[i])      # vector area, raw winding
    disp = vol.point_data.get("Displacement")
    md = np.asarray(disp)[idx].mean(0) if disp is not None else None
    return md, Araw


def lumen_volume_mm3(vol):
    """Signed-magnitude volume of an all-tet deformed mesh [mm^3].

    Same tet-volume sum as gcl_test/measure_gcl.mesh_volume."""
    c = np.asarray(vol.cells).reshape(-1, 5)[:, 1:]
    p = np.asarray(vol.points)
    a, b, cc, d = p[c[:, 0]], p[c[:, 1]], p[c[:, 2]], p[c[:, 3]]
    return np.abs((np.cross(b - a, cc - a) * (d - a)).sum(1)).sum() / 6.0


def step_of(path):
    return int("".join(filter(str.isdigit, os.path.basename(path))) or -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtu-glob", required=True,
                    help="glob for the deformed-mesh VTUs, e.g. '.../*-procs/results_*.vtu'")
    ap.add_argument("--surf-dir", required=True, help="dir with cap .vtp files")
    ap.add_argument("--dt", type=float, required=True, help="solver time step (s)")
    ap.add_argument("--inlet", default=DEFAULT_INLET)
    ap.add_argument("--outlets", default=",".join(DEFAULT_OUTLETS))
    ap.add_argument("--rigid-cap", action="append", default=[],
                    help="cap(s) prescribed as a RIGID translation (e.g. asc): use relative flux "
                         "Q_abs - v_mesh.A_vec with uniform mesh velocity from the Displacement field")
    ap.add_argument("--cycle-range", default=None,
                    help="LO,HI step numbers to restrict to one cycle (inclusive)")
    args = ap.parse_args()

    outlets = [o.strip() for o in args.outlets.split(",") if o.strip()]

    vtus = sorted(glob.glob(args.vtu_glob), key=step_of)
    if not vtus:
        raise SystemExit(f"no VTU matched {args.vtu_glob}")
    if args.cycle_range:
        lo, hi = (int(x) for x in args.cycle_range.split(","))
        vtus = [f for f in vtus if lo <= step_of(f) <= hi]
        if not vtus:
            raise SystemExit(f"no VTU with step in [{lo},{hi}]")
    steps = np.array([step_of(f) for f in vtus])
    # spacing between saved frames, in solver steps (e.g. 10) -> dt_frame in s
    if len(steps) > 1:
        dstep = int(np.round(np.median(np.diff(steps))))
    else:
        dstep = 1
    dt_frame = dstep * args.dt
    print(f"{len(vtus)} VTU snapshots: {os.path.basename(vtus[0])} .. "
          f"{os.path.basename(vtus[-1])}  (save every {dstep} steps, "
          f"dt_frame={dt_frame:g} s)")

    inlet_surf = pv.read(find_vtp(args.surf_dir, args.inlet))
    outlet_surfs = {o: pv.read(find_vtp(args.surf_dir, o)) for o in outlets}

    rigidset = set(c.strip() for c in args.rigid_cap if c.strip())
    allcaps = {args.inlet: inlet_surf, **outlet_surfs}
    md_stack = {c: [] for c in rigidset if c in allcaps}
    ar_stack = {c: [] for c in rigidset if c in allcaps}

    # pass 1: raw cap fluxes (vtp winding) + lumen volume per frame
    qin_raw = np.zeros(len(vtus))
    qout_raw = {o: np.zeros(len(vtus)) for o in outlets}
    Vmm3 = np.zeros(len(vtus))
    for t, f in enumerate(vtus):
        vol = pv.read(f)
        qin_raw[t] = tri_flux_raw(inlet_surf, vol)
        for o in outlets:
            qout_raw[o][t] = tri_flux_raw(outlet_surfs[o], vol)
        Vmm3[t] = lumen_volume_mm3(vol)
        for c in md_stack:
            md, ar = cap_meandisp_area(allcaps[c], vol)
            md_stack[c].append(md); ar_stack[c].append(ar)

    # RIGID-cap correction: subtract the mesh-velocity flux v_mesh.A_vec (raw winding,
    # uniform v_mesh = d(mean cap displacement)/dt) so the residual sees the RELATIVE flux.
    for c in list(md_stack):
        if any(m is None for m in md_stack[c]):
            print(f"[rigid-cap {c}] no Displacement field in VTUs -- correction skipped")
            continue
        md = np.array(md_stack[c]); ar = np.array(ar_stack[c])      # (T,3),(T,3)
        vmesh = np.gradient(md, dt_frame, axis=0)
        corr = np.einsum('ti,ti->t', vmesh, ar) / 100.0            # mL/s, raw winding
        if c == args.inlet:
            qin_raw = qin_raw - corr
        elif c in qout_raw:
            qout_raw[c] = qout_raw[c] - corr
        print(f"[rigid-cap {c}] relative-flux correction applied "
              f"(|v_mesh.A| max={np.max(np.abs(corr)):.3f} mL/s)")

    # Orient each cap to its PHYSICAL sign from its cycle mean:
    #   inlet  physical-positive = INFLOW   -> sign so that mean(Q_in)  > 0
    #   outlet physical-positive = OUTFLOW  -> sign so that mean(Q_out) > 0
    # (each cap has a definite mean direction; this maps raw .vtp winding to the
    #  physiological convention without needing parent-tet apex info.)
    s_in = 1.0 if qin_raw.mean() >= 0 else -1.0
    Qin = s_in * qin_raw
    Qout = {}
    for o in outlets:
        s = 1.0 if qout_raw[o].mean() >= 0 else -1.0
        Qout[o] = s * qout_raw[o]
    Qout_sum = sum(Qout.values())

    # dV/dt by centered difference (mm^3/s) -> mL/s  (1 mL = 1000 mm^3)
    dVdt = np.gradient(Vmm3, dt_frame) / 1000.0

    # instantaneous residual, Eq.(2): Q_in - sum Q_out - dV/dt
    r = Qin - Qout_sum - dVdt

    # ---- per-step table ----
    hdr = (f"{'step':>6} {'Q_in':>9} {'sumQ_out':>9} {'dV/dt':>9} "
           f"{'r':>9} {'r/Q_in%':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for t in range(len(vtus)):
        rel = 100.0 * r[t] / Qin[t] if abs(Qin[t]) > 1e-12 else float("nan")
        print(f"{steps[t]:6d} {Qin[t]:9.3f} {Qout_sum[t]:9.3f} {dVdt[t]:9.3f} "
              f"{r[t]:9.3f} {rel:8.2f}")

    # ---- acceptance numbers ----
    # NB endpoints use one-sided np.gradient (less accurate); the integrals below
    # use the trapezoid rule consistent with that spacing.
    max_qin = np.max(np.abs(Qin))
    acc_inst = np.max(np.abs(r)) / max_qin if max_qin > 0 else float("nan")
    int_r = np.trapz(r, dx=dt_frame)
    int_qin = np.trapz(np.abs(Qin), dx=dt_frame)
    acc_cycle = abs(int_r) / int_qin if int_qin > 0 else float("nan")

    print("\n=== acceptance ===")
    print(f"  max|r|/max|Q_in|        = {100*acc_inst:6.2f}%   (target < 2.0%)   "
          f"{'PASS' if acc_inst < 0.02 else 'FAIL'}")
    print(f"  |int r dt|/int|Q_in| dt = {100*acc_cycle:6.2f}%   (target < 0.5%)   "
          f"{'PASS' if acc_cycle < 0.005 else 'FAIL'}")
    print(f"  V range [mL] = [{Vmm3.min()/1000:.3f}, {Vmm3.max()/1000:.3f}]  "
          f"(swing {100*(Vmm3.max()/Vmm3.min()-1):.1f}%)")
    print(f"  cycle-mean Q_in = {Qin.mean():.3f} mL/s, "
          f"sum Q_out = {Qout_sum.mean():.3f} mL/s")


if __name__ == "__main__":
    main()
