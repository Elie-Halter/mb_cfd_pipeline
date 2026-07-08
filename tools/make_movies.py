#!/usr/bin/env python3
"""
Automatic post-run visualization: MOTION / VELOCITY / HELICITY movies (+ optional
FB-vs-MB side-by-side), from a directory of svMultiPhysics `results_*.vtu`.

No ParaView required (not installed on pc1): headless rendering with PyVista
(off_screen Plotter) + ffmpeg for mp4/gif assembly (no ImageMagick needed).

Fields expected in the VTU point data: Velocity [cm/s], Pressure, WSS, Vorticity,
Displacement [mm]. There is NO Helicity field -- it is computed here:
    H       = (Velocity . Vorticity)                    (density, per node)
    LNH     = H / (|Velocity| * |Vorticity| + eps)       (normalised, in [-1,1])
The VTU points are already the DEFORMED (moving-mesh/ALE) coordinates, so simply
reading the `results_*.vtu` series in order animates the wall motion -- no "Warp
By Vector" needed. Wall-only context (excluding caps) is built by mapping a
wall.vtp (with a GlobalNodeID point array) onto each frame's volume via the
GID-1 index -- the *same* deformation-proof technique used by hemo_indices.py /
compare_FB_MB.py in this tools/ directory (positional KDTree would be WRONG
here because the boundary moves).

Units: mesh coordinates in mm, Velocity in cm/s (see tools/README.md).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  python3 make_movies.py <run_dir> --wall <wall.vtp> \\
      [--stl-dir <dir with *_capped.stl>] \\
      [--cycle-start 974 --cycle-end 1948] [--dt 0.001 --period 0.974] \\
      [--fps 12] [--out <run_dir>/movies] \\
      [--types motion,velocity,helicity,fbmb] [--fb-dir <FB run_dir>] \\
      [--width 1280 --height 800] [--view arch|iso|front|side|top] [--rotate] \\
      [--slice-origin x,y,z --slice-normal nx,ny,nz] \\
      [--seed-origin x,y,z --seed-normal nx,ny,nz --seed-radius R --n-seeds N] \\
      [--max-frames N --stride S]      # quick tests: few, small frames

Quick smoke test (a handful of small frames, no ffmpeg heavy load):
  python3 make_movies.py <run_dir> --wall wall.vtp --max-frames 5 \\
      --width 480 --height 360 --types motion,velocity,helicity

End-of-run hook: see run_movies.sh in this same directory.

--------------------------------------------------------------------------
ANATOMY-DERIVED SMART DEFAULTS (computed from the geometry, not bbox guesses)
--------------------------------------------------------------------------
When --wall (and its cap dir, or --cap-dir) is given, the tool derives:
  * the SLICE PLANE = the aortic-arch plane, i.e. the smallest-variance PCA
    axis of the wall point cloud (origin = wall centroid). This cuts the full
    "candy-cane" cross-section instead of a horizontal plane that would clip
    the two limbs into disconnected ellipses.
  * the STREAMLINE SEED DISC = a disc just inside the ASCENDING INLET cap
    (asc.vtp): centre = cap centroid, normal = cap normal, radius = 0.85*r_max,
    so streamlines actually enter the lumen and trace up-and-over the arch.
  * the CAMERA (--view arch, the default) = looks face-on at that arch plane
    (oblique variant for the 3D motion movie).
Any of these is overridable: explicit --slice-origin/--slice-normal,
--seed-origin/--seed-normal/--seed-radius, --view iso|front|side|top all win
over the derived value. With no --wall, everything degrades to the old bbox /
horizontal / iso defaults (a warning is printed).

OTHER ASSUMPTIONS TO TUNE PER GEOMETRY (documented, not hidden)
  * Helicity isosurface level defaults to --helicity-iso-frac (0.7) * robust
    |H| max over the selected frames; override with --helicity-iso-value.
  * Color scales are FIXED over the whole selected frame range so they don't
    flicker; by default "robust" (per-frame 1st/99th percentile, then
    median across frames) rather than exact min/max, since raw CFD extrema
    at cap edges are often outlier spikes that wash out the colormap.
    Use --exact-clim for literal global min/max instead.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess

import numpy as np
import pyvista as pv

FRAME_RE = re.compile(r"results_(\d+)\.vtu$")


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def step_of(path):
    m = FRAME_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_vtus(run_dir, start=None, end=None, stride=1, max_frames=None):
    files = sorted(glob.glob(os.path.join(run_dir, "results_*.vtu")), key=step_of)
    if start is not None:
        files = [f for f in files if step_of(f) >= start]
    if end is not None:
        files = [f for f in files if step_of(f) <= end]
    files = files[:: max(stride, 1)]
    if max_frames:
        files = files[:max_frames]
    return files


def parse_vec3(s):
    if not s:
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f'expected "x,y,z", got {s!r}')
    return tuple(parts)


def bbox_center(b):
    return ((b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0)


# --------------------------------------------------------------------------
# ANATOMY: derive a *meaningful* slice plane, streamline seed disc and camera
# from the geometry itself (wall PCA + ascending-inlet cap), instead of the
# non-anatomical bbox-center / horizontal defaults. Everything degrades
# gracefully to None so callers keep working when inputs are missing.
# --------------------------------------------------------------------------
def principal_axes(points):
    """SVD principal axes of a point cloud. Returns (centroid, singvals, axes)
    with axes rows sorted by decreasing variance; axes[2] = smallest-variance
    direction = normal of the best-fit plane (the aortic-arch plane for a wall)."""
    P = np.asarray(points, dtype=float)
    c = P.mean(axis=0)
    _u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, s, vt


def cap_info(path):
    """(centroid, unit normal, r_max) of a cap surface .vtp, or None on failure.
    Normal = smallest-variance axis of the ~planar cap disc; radius = max in-plane
    distance to the centroid (so the seed disc sits just inside the real lumen)."""
    try:
        s = pv.read(path)
    except Exception:
        return None
    if s.n_points < 3:
        return None
    c, _sv, vt = principal_axes(s.points)
    n = vt[2] / (np.linalg.norm(vt[2]) + 1e-12)
    r = float(np.sqrt(((s.points - c) ** 2).sum(axis=1)).max())
    return c, n, r


def derive_anatomy(wall_topo, cap_dir, asc_name="asc.vtp"):
    """Smart geometry-derived defaults:
      * slice_origin / slice_normal -> the aortic-arch plane (wall PCA), so the
        velocity/helicity slice cuts the full candy-cane, not two stray ellipses;
      * view_dir       -> look face-on at that arch plane (best 2D-field view);
      * view_oblique   -> a rotated-about-the-long-axis view for the 3D motion movie;
      * seed_origin/normal/radius -> a disc just inside the ASCENDING INLET, so
        streamlines actually enter the lumen and trace up-and-over the arch.
    Any missing input simply leaves that entry None (caller keeps its own default)."""
    A = dict(slice_origin=None, slice_normal=None, view_dir=None,
             view_oblique=None, view_up=(0.0, 0.0, 1.0),
             seed_origin=None, seed_normal=None, seed_radius=None, note=[])
    if wall_topo is not None and wall_topo.n_points >= 10:
        c, sv, vt = principal_axes(wall_topo.points)
        normal = vt[2] / (np.linalg.norm(vt[2]) + 1e-12)
        # deterministic sign: face the arch from the -x / anterior-ish side
        if normal[0] > 0:
            normal = -normal
        long_axis = vt[0]                      # ~ vessel long axis (mostly +z)
        in_plane = vt[1]                       # 2nd in-plane axis
        oblique = normal * 0.82 + in_plane * 0.5 + long_axis * 0.06
        oblique = oblique / (np.linalg.norm(oblique) + 1e-12)
        A["slice_origin"] = tuple(c)
        A["slice_normal"] = tuple(normal)
        A["view_dir"] = tuple(normal)
        A["view_oblique"] = tuple(oblique)
        A["note"].append(
            "arch-plane PCA normal=%s (var frac %s)"
            % (np.round(normal, 3).tolist(), np.round(sv / sv.sum(), 3).tolist()))
    asc = cap_info(os.path.join(cap_dir, asc_name)) if cap_dir else None
    if asc is not None:
        c, n, r = asc
        A["seed_origin"] = tuple(c)
        A["seed_normal"] = tuple(n)
        A["seed_radius"] = 0.85 * r
        A["note"].append("seed disc @ ascending inlet ctr=%s r=%.2f mm"
                          % (np.round(c, 2).tolist(), 0.85 * r))
    return A


def anat_get(args, cli_val, key, fallback, is_vec=True):
    """Resolve a value with priority: explicit CLI > derived anatomy > hard fallback."""
    v = parse_vec3(cli_val) if (is_vec and isinstance(cli_val, str)) else cli_val
    if v is not None:
        return v
    a = getattr(args, "_anat", None)
    if a is not None and a.get(key) is not None:
        return a[key]
    return fallback


def wall_gid_idx(wall, vol):
    """Row index in `vol` for each `wall` point -- deformation-proof (see module
    docstring). Same convention as hemo_indices.py:surface_rows_in_volume."""
    gid = wall.point_data.get("GlobalNodeID")
    if gid is not None:
        idx = np.asarray(gid).astype(np.int64) - 1
        if idx.min() >= 0 and idx.max() < vol.n_points:
            return idx, "gid-1"
    from scipy.spatial import cKDTree
    tree = cKDTree(vol.points)
    _, idx = tree.query(wall.points, k=1)
    print("  [WARN] wall->volume mapping via KDTree (no GlobalNodeID on the wall) "
          "-- WRONG if the boundary moves.")
    return idx, "kdtree"


def wall_surface_from_volume(vol, wall_topo, idx, fields=()):
    """Wall topology (faces) of `wall_topo`, points/fields pulled from `vol` at
    the current (possibly deformed) frame -- gives the moving wall surface."""
    surf = wall_topo.copy()
    surf.points = vol.points[idx]
    for f in fields:
        if f in vol.point_data:
            surf.point_data[f] = vol.point_data[f][idx]
    return surf


def compute_helicity(mesh, eps=1e-8):
    """Adds 'Helicity' (density, v.omega) and 'Helicity_LNH' (normalised) point
    arrays. No-op if Velocity/Vorticity are missing."""
    if "Velocity" not in mesh.point_data or "Vorticity" not in mesh.point_data:
        return mesh
    V = np.asarray(mesh.point_data["Velocity"])
    W = np.asarray(mesh.point_data["Vorticity"])
    H = np.einsum("ij,ij->i", V, W)
    mesh.point_data["Helicity"] = H
    vmag = np.linalg.norm(V, axis=1)
    wmag = np.linalg.norm(W, axis=1)
    mesh.point_data["Helicity_LNH"] = H / (vmag * wmag + eps)
    return mesh


def scan_stats(files, need, robust=True):
    """One pass over `files`, returns {key: (vmin, vmax)} for each requested key
    in {'velmag','dispmag','helicity','helicity_abs','wssmag'}. Missing fields
    are silently skipped (key -> None) so callers degrade gracefully."""
    exact = {k: [np.inf, -np.inf] for k in need}
    pct = {k: {"lo": [], "hi": []} for k in need}
    n_seen = {k: 0 for k in need}

    def _update(key, arr):
        if arr is None or len(arr) == 0:
            return
        exact[key][0] = min(exact[key][0], float(np.min(arr)))
        exact[key][1] = max(exact[key][1], float(np.max(arr)))
        lo, hi = np.percentile(arr, [1, 99])
        pct[key]["lo"].append(lo)
        pct[key]["hi"].append(hi)
        n_seen[key] += 1

    for f in files:
        m = pv.read(f)
        if "velmag" in need and "Velocity" in m.point_data:
            _update("velmag", np.linalg.norm(m.point_data["Velocity"], axis=1))
        if "dispmag" in need and "Displacement" in m.point_data:
            _update("dispmag", np.linalg.norm(m.point_data["Displacement"], axis=1))
        if ("helicity" in need or "helicity_abs" in need) and \
                "Velocity" in m.point_data and "Vorticity" in m.point_data:
            compute_helicity(m)
            if "helicity" in need:
                _update("helicity", m.point_data["Helicity"])
            if "helicity_abs" in need:
                _update("helicity_abs", np.abs(m.point_data["Helicity"]))
        if "wssmag" in need and "WSS" in m.point_data:
            _update("wssmag", np.linalg.norm(m.point_data["WSS"], axis=1))

    out = {}
    for k in need:
        if n_seen[k] == 0:
            out[k] = None
        elif robust:
            out[k] = (float(np.median(pct[k]["lo"])), float(np.median(pct[k]["hi"])))
        else:
            out[k] = (exact[k][0], exact[k][1])
    return out


def default_camera(bounds, view="iso", zoom=1.0, explicit_dir=None, up=None):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    dirs = {
        "iso": np.array([1.0, -1.3, 0.55]),
        "front": np.array([0.0, -1.0, 0.05]),
        "side": np.array([1.0, 0.0, 0.05]),
        "top": np.array([0.001, 0.001, 1.0]),
    }
    if explicit_dir is not None:
        d = np.asarray(explicit_dir, dtype=float)
    else:
        d = dirs.get(view, dirs["iso"])
    d = d / (np.linalg.norm(d) + 1e-12)
    dist = diag * 1.5 / max(zoom, 1e-3)
    pos = center + d * dist
    if up is not None:
        viewup = tuple(up)
    else:
        viewup = (0.0, 0.0, 1.0) if view != "top" else (0.0, 1.0, 0.0)
    return [tuple(pos), tuple(center), viewup]


def scene_camera(args, bounds, oblique=False):
    """Resolve the camera for a scene, honouring the anatomy-derived arch views.
      view == 'arch'  -> look face-on at the wall PCA arch plane;
      oblique=True    -> rotated-about-long-axis variant (motion / 3D shots)."""
    a = getattr(args, "_anat", None)
    up = a.get("view_up") if a else None
    if a is not None:
        key = "view_oblique" if oblique else "view_dir"
        if args.view == "arch" and a.get(key) is not None:
            return default_camera(bounds, explicit_dir=a[key], up=up, zoom=args.zoom)
        if args.view == "arch" and a.get("view_dir") is not None:
            return default_camera(bounds, explicit_dir=a["view_dir"], up=up, zoom=args.zoom)
    return default_camera(bounds, args.view if args.view != "arch" else "iso", args.zoom)


def scalar_bar(title, fmt="%.0f", n_labels=4):
    """Consistent, tidy scalar bars: horizontal, bottom-centred, no italics."""
    return dict(title=title, color="white", n_labels=n_labels, fmt=fmt,
                italic=False, bold=False, font_family="arial",
                title_font_size=20, label_font_size=16, shadow=True,
                width=0.42, height=0.055, position_x=0.29, position_y=0.035)


def add_scene_lighting(pl):
    """Soft three-point-ish lighting so vessels/streamlines read with depth
    instead of the flat default headlight. No-op safe."""
    try:
        pl.remove_all_lights()
        for pos, inten in (((1, -1, 1), 0.85), ((-1, -0.5, 0.5), 0.45),
                           ((0, 1, -0.3), 0.35)):
            lt = pv.Light(position=pos, focal_point=(0, 0, 0),
                          intensity=inten, light_type="scene light")
            lt.positional = False
            pl.add_light(lt)
    except Exception:
        pass


def make_plotter(shape=(1, 1), size=(1280, 800), dark=True, lighting=True):
    pl = pv.Plotter(off_screen=True, shape=shape, window_size=size, border=False,
                    lighting="none" if lighting else "light kit")
    pl.set_background("#0a0d14" if dark else "white")
    try:
        pl.enable_anti_aliasing("msaa", multi_samples=8)
    except Exception:
        pass
    if lighting:
        add_scene_lighting(pl)
    return pl


def annotate(pl, args, step, i, n, extra=""):
    """Elegant two-corner overlay: title + phase clock upper-left, field tag
    upper-right. Keeps the frame uncluttered (scalar bar owns the bottom)."""
    if args.dt and args.period:
        t = step * args.dt
        frac = (t % args.period) / args.period
        left = f"{args.title}\nt = {t:.3f} s   t/T = {frac:.2f}"
    elif args.dt:
        t = step * args.dt
        left = f"{args.title}\nt = {t:.3f} s   step {step}"
    else:
        left = f"{args.title}\nstep {step}  ({i + 1}/{n})"
    pl.add_text(left, position="upper_left", font_size=args.font_size,
                color="white", shadow=True, font="arial")
    if extra:
        pl.add_text(extra, position="upper_right",
                    font_size=max(args.font_size + 2, 14),
                    color="#cfe3ff", shadow=True, font="arial")


def assemble_video(frame_dir, out_prefix, fps, scale=None):
    """frame_dir/frame_%04d.png -> out_prefix.mp4 + out_prefix.gif (ffmpeg only,
    no ImageMagick). Returns (mp4_path or None, gif_path or None)."""
    pattern = os.path.join(frame_dir, "frame_%04d.png")
    mp4 = out_prefix + ".mp4"
    gif = out_prefix + ".gif"
    chain = [f"fps={fps}"]
    if scale:
        chain.append(f"scale={scale}:-1:flags=lanczos")
    chain_str = ",".join(chain)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4],
            check=True, capture_output=True,
        )
    except Exception as e:
        print(f"  [warn] ffmpeg mp4 failed: {e}")
        mp4 = None

    palette = os.path.join(frame_dir, "_palette.png")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
             "-vf", f"{chain_str},palettegen", palette],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-i", palette,
             "-lavfi", f"[0:v]{chain_str}[x];[x][1:v]paletteuse", gif],
            check=True, capture_output=True,
        )
    except Exception as e:
        print(f"  [warn] ffmpeg gif failed: {e}")
        gif = None
    return mp4, gif


def save_hero(frame_dir, out_dir, name, n, hero_frac):
    idx = int(round(hero_frac * (n - 1))) if n > 1 else 0
    src = os.path.join(frame_dir, f"frame_{idx:04d}.png")
    dst = os.path.join(out_dir, f"{name}_hero.png")
    if os.path.exists(src):
        shutil.copy(src, dst)
        return dst
    return None


# --------------------------------------------------------------------------
# movie #1: MOTION (deforming wall, + optional phase-STL ghosts)
# --------------------------------------------------------------------------
def render_motion(files, wall_topo, idx, stl_paths, out_dir, args):
    print("[motion] scanning displacement range...")
    stats = scan_stats(files, need={"dispmag"}, robust=not args.exact_clim)
    if stats["dispmag"] is None:
        print("  [skip] no Displacement field in these VTU")
        return None
    dmin, dmax = stats["dispmag"]
    dmin = max(dmin, 0.0)
    print(f"  |Displacement| clim: [{dmin:.3f}, {dmax:.3f}] mm")

    vol0 = pv.read(files[0])
    bounds = list(wall_topo.bounds) if wall_topo is not None else list(vol0.bounds)
    ghosts = []
    for p in stl_paths:
        try:
            g = pv.read(p)
            ghosts.append((os.path.basename(p), g))
            b = g.bounds
            bounds = [min(bounds[0], b[0]), max(bounds[1], b[1]),
                      min(bounds[2], b[2]), max(bounds[3], b[3]),
                      min(bounds[4], b[4]), max(bounds[5], b[5])]
        except Exception as e:
            print(f"  [warn] could not read STL ghost {p}: {e}")

    cam = scene_camera(args, bounds, oblique=True)
    frame_dir = os.path.join(out_dir, "frames_motion")
    os.makedirs(frame_dir, exist_ok=True)
    n = len(files)
    for i, f in enumerate(files):
        step = step_of(f)
        try:
            vol = pv.read(f)
            if wall_topo is not None:
                surf = wall_surface_from_volume(vol, wall_topo, idx, fields=("Displacement",))
            else:
                surf = vol.extract_surface()
            if "Displacement" in surf.point_data:
                surf["DispMag"] = np.linalg.norm(surf.point_data["Displacement"], axis=1)
            else:
                surf["DispMag"] = np.zeros(surf.n_points)
            try:
                surf = surf.compute_normals(auto_orient_normals=True)
            except Exception:
                pass

            pl = make_plotter(size=(args.width, args.height))
            for _, g in ghosts:
                pl.add_mesh(g, color="#7f93b5", opacity=0.06, style="wireframe", line_width=1)
            pl.add_mesh(surf, scalars="DispMag", cmap=args.cmap_motion, clim=(dmin, dmax),
                        smooth_shading=True, specular=0.5, specular_power=25,
                        ambient=0.28, diffuse=0.85,
                        scalar_bar_args=scalar_bar("|Displacement|  [mm]", fmt="%.2f"))
            pl.camera_position = cam
            if args.rotate:
                pl.camera.azimuth(360.0 * i / max(n - 1, 1))
            annotate(pl, args, step, i, n, extra="MOTION")
            png = os.path.join(frame_dir, f"frame_{i:04d}.png")
            pl.screenshot(png)
            pl.close()
            print(f"  frame {i + 1}/{n}: step {step} -> {png}")
        except Exception as e:
            print(f"  [warn] motion frame failed at step {step}: {e}")

    mp4, gif = assemble_video(frame_dir, os.path.join(out_dir, "motion"), args.fps)
    hero = save_hero(frame_dir, out_dir, "motion", n, args.hero_frac)
    print(f"[motion] -> {mp4}, {gif}, {hero}")
    return dict(mp4=mp4, gif=gif, hero=hero)


# --------------------------------------------------------------------------
# movie #2: VELOCITY (slice + streamlines, |v|)
# --------------------------------------------------------------------------
def render_velocity(files, wall_topo, idx, out_dir, args):
    print("[velocity] scanning |v| range...")
    stats = scan_stats(files, need={"velmag"}, robust=not args.exact_clim)
    if stats["velmag"] is None:
        print("  [skip] no Velocity field in these VTU")
        return None
    vmin, vmax = stats["velmag"]
    vmin = max(vmin, 0.0)
    print(f"  |v| clim: [{vmin:.1f}, {vmax:.1f}] cm/s")

    vol0 = pv.read(files[0])
    bounds = list(wall_topo.bounds) if wall_topo is not None else list(vol0.bounds)
    origin = anat_get(args, args.slice_origin, "slice_origin", bbox_center(bounds))
    normal = anat_get(args, args.slice_normal, "slice_normal", (0.0, 0.0, 1.0))
    seed_origin = anat_get(args, args.seed_origin, "seed_origin", origin)
    seed_normal = anat_get(args, args.seed_normal, "seed_normal", normal)
    seed_radius = anat_get(args, args.seed_radius, "seed_radius",
                           0.3 * np.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2]),
                           is_vec=False)
    cam = scene_camera(args, bounds)

    frame_dir = os.path.join(out_dir, "frames_velocity")
    os.makedirs(frame_dir, exist_ok=True)
    n = len(files)
    for i, f in enumerate(files):
        step = step_of(f)
        try:
            vol = pv.read(f)
            if "Velocity" not in vol.point_data:
                print(f"  [skip] no Velocity at step {step}")
                continue
            vol.point_data["VelMag"] = np.linalg.norm(vol.point_data["Velocity"], axis=1)

            pl = make_plotter(size=(args.width, args.height))
            if wall_topo is not None:
                wsurf = wall_surface_from_volume(vol, wall_topo, idx, fields=())
                pl.add_mesh(wsurf, color="#afc0d8", opacity=args.wall_opacity,
                            smooth_shading=True, specular=0.2)

            sl = vol.slice(origin=origin, normal=normal)
            if sl.n_points:
                pl.add_mesh(sl, scalars="VelMag", cmap=args.cmap_velocity, clim=(vmin, vmax),
                            opacity=0.92,
                            scalar_bar_args=scalar_bar("|v|  [cm/s]"))
            else:
                print(f"  [warn] empty slice at step {step} (adjust --slice-origin/--slice-normal)")

            disc = pv.Disc(center=seed_origin, inner=0, outer=seed_radius,
                            normal=seed_normal, r_res=8, c_res=args.n_seeds)
            try:
                streams = vol.streamlines_from_source(
                    disc, vectors="Velocity", integration_direction="both",
                    max_length=args.stream_length, step_unit="l",
                    initial_step_length=0.5, terminal_speed=1e-3,
                )
                if streams.n_points:
                    tubes = streams.tube(radius=args.stream_radius)
                    pl.add_mesh(tubes, scalars="VelMag", cmap=args.cmap_velocity,
                                clim=(vmin, vmax), show_scalar_bar=False,
                                specular=0.6, specular_power=20, ambient=0.2)
            except Exception as e:
                print(f"  [warn] streamlines failed at step {step}: {e}")

            pl.camera_position = cam
            annotate(pl, args, step, i, n, extra="VELOCITY")
            png = os.path.join(frame_dir, f"frame_{i:04d}.png")
            pl.screenshot(png)
            pl.close()
            print(f"  frame {i + 1}/{n}: step {step} -> {png}")
        except Exception as e:
            print(f"  [warn] velocity frame failed at step {step}: {e}")

    mp4, gif = assemble_video(frame_dir, os.path.join(out_dir, "velocity"), args.fps)
    hero = save_hero(frame_dir, out_dir, "velocity", n, args.hero_frac)
    print(f"[velocity] -> {mp4}, {gif}, {hero}")
    return dict(mp4=mp4, gif=gif, hero=hero)


# --------------------------------------------------------------------------
# movie #3: HELICITY (streamlines colored by H, + isosurfaces of +-H)
# --------------------------------------------------------------------------
def render_helicity(files, wall_topo, idx, out_dir, args):
    print("[helicity] scanning H = v.omega range...")
    stats = scan_stats(files, need={"helicity", "helicity_abs"}, robust=not args.exact_clim)
    if stats["helicity_abs"] is None:
        print("  [skip] Velocity/Vorticity missing -- cannot compute helicity")
        return None
    hlo, hhi = stats["helicity"]
    M = max(abs(hlo), abs(hhi), stats["helicity_abs"][1])
    print(f"  symmetric clim +/- {M:.4g} (helicity density, cm/s^2)")

    vol0 = pv.read(files[0])
    bounds = list(wall_topo.bounds) if wall_topo is not None else list(vol0.bounds)
    seed_origin = anat_get(args, args.seed_origin, "seed_origin", bbox_center(bounds))
    seed_normal = anat_get(args, args.seed_normal, "seed_normal", (0.0, 0.0, 1.0))
    seed_radius = anat_get(args, args.seed_radius, "seed_radius",
                           0.3 * np.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2]),
                           is_vec=False)
    iso_val = args.helicity_iso_value if args.helicity_iso_value else args.helicity_iso_frac * M
    cam = scene_camera(args, bounds)

    frame_dir = os.path.join(out_dir, "frames_helicity")
    os.makedirs(frame_dir, exist_ok=True)
    n = len(files)
    for i, f in enumerate(files):
        step = step_of(f)
        try:
            vol = pv.read(f)
            if "Velocity" not in vol.point_data or "Vorticity" not in vol.point_data:
                print(f"  [skip] missing fields at step {step}")
                continue
            compute_helicity(vol)

            pl = make_plotter(size=(args.width, args.height))
            if wall_topo is not None:
                wsurf = wall_surface_from_volume(vol, wall_topo, idx, fields=())
                pl.add_mesh(wsurf, color="#afc0d8", opacity=args.wall_opacity,
                            smooth_shading=True, specular=0.2)

            if not args.no_iso:
                try:
                    iso_pos = vol.contour(isosurfaces=[iso_val], scalars="Helicity")
                    iso_neg = vol.contour(isosurfaces=[-iso_val], scalars="Helicity")
                    if iso_pos.n_points:
                        pl.add_mesh(iso_pos, color="#e8503a", opacity=0.38,
                                    smooth_shading=True, specular=0.3, diffuse=0.9)
                    if iso_neg.n_points:
                        pl.add_mesh(iso_neg, color="#3a78e8", opacity=0.38,
                                    smooth_shading=True, specular=0.3, diffuse=0.9)
                except Exception as e:
                    print(f"  [warn] isosurface failed at step {step}: {e}")

            if not args.no_streamlines:
                disc = pv.Disc(center=seed_origin, inner=0, outer=seed_radius,
                                normal=seed_normal, r_res=8, c_res=args.n_seeds)
                try:
                    streams = vol.streamlines_from_source(
                        disc, vectors="Velocity", integration_direction="both",
                        max_length=args.stream_length, step_unit="l",
                        initial_step_length=0.5, terminal_speed=1e-3,
                    )
                    if streams.n_points:
                        tubes = streams.tube(radius=args.stream_radius)
                        pl.add_mesh(tubes, scalars="Helicity", cmap=args.cmap_helicity,
                                    clim=(-M, M), specular=0.5, specular_power=20,
                                    scalar_bar_args=scalar_bar("H = v . ω  [cm/s²]",
                                                               fmt="%.1e", n_labels=5))
                except Exception as e:
                    print(f"  [warn] streamlines failed at step {step}: {e}")

            pl.camera_position = cam
            annotate(pl, args, step, i, n, extra="HELICITY")
            png = os.path.join(frame_dir, f"frame_{i:04d}.png")
            pl.screenshot(png)
            pl.close()
            print(f"  frame {i + 1}/{n}: step {step} -> {png}")
        except Exception as e:
            print(f"  [warn] helicity frame failed at step {step}: {e}")

    mp4, gif = assemble_video(frame_dir, os.path.join(out_dir, "helicity"), args.fps)
    hero = save_hero(frame_dir, out_dir, "helicity", n, args.hero_frac)
    print(f"[helicity] -> {mp4}, {gif}, {hero}")
    return dict(mp4=mp4, gif=gif, hero=hero)


# --------------------------------------------------------------------------
# bonus: FB vs MB side-by-side (velocity slice, matched by step number)
# --------------------------------------------------------------------------
def render_fbmb(mb_files, fb_dir, wall_topo, idx, out_dir, args):
    fb_files_all = sorted(glob.glob(os.path.join(fb_dir, "results_*.vtu")), key=step_of)
    fb_by_step = {step_of(f): f for f in fb_files_all}
    pairs = [(f, fb_by_step[step_of(f)]) for f in mb_files if step_of(f) in fb_by_step]
    if not pairs:
        print("[fbmb] no matching step numbers between FB and MB dirs -- skip")
        return None
    print(f"[fbmb] {len(pairs)} matched steps")

    mb_stats = scan_stats([p[0] for p in pairs], need={"velmag"}, robust=not args.exact_clim)
    fb_stats = scan_stats([p[1] for p in pairs], need={"velmag"}, robust=not args.exact_clim)
    if mb_stats["velmag"] is None or fb_stats["velmag"] is None:
        print("  [skip] Velocity missing on one side")
        return None
    vmin = min(mb_stats["velmag"][0], fb_stats["velmag"][0])
    vmax = max(mb_stats["velmag"][1], fb_stats["velmag"][1])
    print(f"  common |v| clim: [{vmin:.1f}, {vmax:.1f}] cm/s")

    vol0 = pv.read(pairs[0][0])
    bounds = list(wall_topo.bounds) if wall_topo is not None else list(vol0.bounds)
    origin = anat_get(args, args.slice_origin, "slice_origin", bbox_center(bounds))
    normal = anat_get(args, args.slice_normal, "slice_normal", (0.0, 0.0, 1.0))
    seed_origin = anat_get(args, args.seed_origin, "seed_origin", origin)
    seed_normal = anat_get(args, args.seed_normal, "seed_normal", normal)
    seed_radius = anat_get(args, args.seed_radius, "seed_radius",
                           0.3 * np.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2]),
                           is_vec=False)
    cam = scene_camera(args, bounds)

    frame_dir = os.path.join(out_dir, "frames_fbmb")
    os.makedirs(frame_dir, exist_ok=True)
    n = len(pairs)
    for i, (mbf, fbf) in enumerate(pairs):
        step = step_of(mbf)
        try:
            pl = pv.Plotter(off_screen=True, shape=(1, 2),
                             window_size=(args.width * 2, args.height), border=False,
                             lighting="none")
            pl.set_background("#0a0d14")
            try:
                pl.enable_anti_aliasing("msaa", multi_samples=8)
            except Exception:
                pass
            for col, (label, f) in enumerate([("FB  (rigid wall)", fbf),
                                              ("MB  (moving wall)", mbf)]):
                pl.subplot(0, col)
                add_scene_lighting(pl)
                vol = pv.read(f)
                vol.point_data["VelMag"] = np.linalg.norm(vol.point_data["Velocity"], axis=1)
                if wall_topo is not None:
                    try:
                        wsurf = wall_surface_from_volume(vol, wall_topo, idx, fields=())
                        pl.add_mesh(wsurf, color="#afc0d8", opacity=args.wall_opacity,
                                    smooth_shading=True, specular=0.2)
                    except Exception:
                        pass
                sl = vol.slice(origin=origin, normal=normal)
                if sl.n_points:
                    pl.add_mesh(sl, scalars="VelMag", cmap=args.cmap_velocity, clim=(vmin, vmax),
                                opacity=0.92, show_scalar_bar=(col == 1),
                                scalar_bar_args=scalar_bar("|v|  [cm/s]"))
                if not args.fbmb_no_streamlines:
                    disc = pv.Disc(center=seed_origin, inner=0, outer=seed_radius,
                                   normal=seed_normal, r_res=8, c_res=args.n_seeds)
                    try:
                        streams = vol.streamlines_from_source(
                            disc, vectors="Velocity", integration_direction="both",
                            max_length=args.stream_length, step_unit="l",
                            initial_step_length=0.5, terminal_speed=1e-3)
                        if streams.n_points:
                            pl.add_mesh(streams.tube(radius=args.stream_radius),
                                        scalars="VelMag", cmap=args.cmap_velocity,
                                        clim=(vmin, vmax), show_scalar_bar=False,
                                        specular=0.6, specular_power=20)
                    except Exception:
                        pass
                pl.camera_position = cam
                pl.add_text(label, position="upper_edge", font_size=args.font_size + 2,
                            color="white", shadow=True, font="arial")
                if col == 0:
                    # footer on the left panel only (no scalar bar there -> no overlap)
                    foot = f"{args.title} | step {step} ({i + 1}/{n})"
                    if args.dt:
                        foot = f"{args.title} | t = {step * args.dt:.3f} s"
                    pl.add_text(foot, position="lower_edge",
                                font_size=max(args.font_size - 2, 8),
                                color="#cfe3ff", shadow=True, font="arial")
            png = os.path.join(frame_dir, f"frame_{i:04d}.png")
            pl.screenshot(png)
            pl.close()
            print(f"  frame {i + 1}/{n}: step {step} -> {png}")
        except Exception as e:
            print(f"  [warn] fbmb frame failed at step {step}: {e}")

    mp4, gif = assemble_video(frame_dir, os.path.join(out_dir, "fbmb"), args.fps)
    hero = save_hero(frame_dir, out_dir, "fbmb", n, args.hero_frac)
    print(f"[fbmb] -> {mp4}, {gif}, {hero}")
    return dict(mp4=mp4, gif=gif, hero=hero)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="dir with results_*.vtu (svMultiPhysics output)")
    ap.add_argument("--wall", default=None,
                     help="wall.vtp with GlobalNodeID (GID-1 mapping; deformation-proof)")
    ap.add_argument("--cap-dir", default=None,
                     help="dir with cap surfaces (asc.vtp,...); default = dir of --wall. "
                          "Used to seed streamlines at the ascending inlet.")
    ap.add_argument("--stl-dir", default=None,
                     help="dir with phase-segmented *_capped.stl for ghost overlay (motion movie)")
    ap.add_argument("--cycle-start", type=int, default=None)
    ap.add_argument("--cycle-end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None, help="cap #frames (quick tests)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=None, help="default: <run_dir>/movies")
    ap.add_argument("--types", default="motion,velocity,helicity",
                     help="comma list: motion,velocity,helicity,fbmb")
    ap.add_argument("--fb-dir", default=None, help="FB run dir for the fbmb side-by-side panel")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--dt", type=float, default=None, help="timestep [s] for the t= annotation")
    ap.add_argument("--period", type=float, default=None, help="cardiac period T [s] (t/T annotation)")
    ap.add_argument("--title", default=None, help="default: basename(run_dir)")
    ap.add_argument("--view", default="arch",
                     choices=["arch", "iso", "front", "side", "top"],
                     help="'arch' (default) = face-on the wall-PCA aortic-arch plane; "
                          "falls back to 'iso' if no wall/anatomy is available.")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--rotate", action="store_true", help="slow azimuth sweep across the motion movie")
    ap.add_argument("--slice-origin", default=None, help='"x,y,z" mm; default = bbox center')
    ap.add_argument("--slice-normal", default=None,
                     help='"nx,ny,nz"; default 0,0,1 -- ASSUMPTION, tune to the true arch plane')
    ap.add_argument("--seed-origin", default=None, help='streamline seed disc center "x,y,z"')
    ap.add_argument("--seed-normal", default=None, help='streamline seed disc normal "nx,ny,nz"')
    ap.add_argument("--seed-radius", type=float, default=None, help="streamline seed disc radius [mm]")
    ap.add_argument("--n-seeds", type=int, default=14, help="angular resolution of the seeding disc")
    ap.add_argument("--stream-length", type=float, default=250.0,
                     help="max streamline path length [mm]")
    ap.add_argument("--stream-radius", type=float, default=0.35, help="streamline tube radius [mm]")
    ap.add_argument("--helicity-iso-value", type=float, default=None)
    ap.add_argument("--helicity-iso-frac", type=float, default=0.7,
                     help="helicity isosurface level as a fraction of the robust |H| max")
    ap.add_argument("--no-iso", action="store_true", help="disable helicity isosurfaces")
    ap.add_argument("--no-streamlines", action="store_true", help="disable streamlines")
    ap.add_argument("--fbmb-no-streamlines", action="store_true",
                     help="FB-vs-MB panel: show slice only, no streamlines")
    ap.add_argument("--wall-opacity", type=float, default=0.10,
                     help="translucent wall context opacity (see the flow inside)")
    ap.add_argument("--exact-clim", action="store_true",
                     help="use true global min/max instead of robust 1-99pct color scales")
    ap.add_argument("--hero-frac", type=float, default=0.32,
                     help="fraction into the sequence used for the *_hero.png")
    ap.add_argument("--font-size", type=int, default=14)
    ap.add_argument("--cmap-velocity", default="turbo")
    ap.add_argument("--cmap-helicity", default="coolwarm")
    ap.add_argument("--cmap-motion", default="plasma")
    return ap


def main():
    args = build_argparser().parse_args()
    pv.OFF_SCREEN = True
    if args.title is None:
        args.title = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = args.out or os.path.join(args.run_dir, "movies")
    os.makedirs(out_dir, exist_ok=True)
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    files = list_vtus(args.run_dir, args.cycle_start, args.cycle_end, args.stride, args.max_frames)
    if not files:
        raise SystemExit(f"no results_*.vtu found in {args.run_dir} for the given range")
    print(f"{len(files)} frames selected: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}")

    wall_topo = idx = None
    if args.wall:
        wall_topo = pv.read(args.wall)
        vol0 = pv.read(files[0])
        idx, mode = wall_gid_idx(wall_topo, vol0)
        print(f"wall->volume mapping: {mode} ({wall_topo.n_points} wall nodes)")
    else:
        print("[WARN] no --wall given: wall-context overlays disabled; motion movie will use "
              "extract_surface() per frame (includes caps).")

    # anatomy-derived smart defaults (arch slice plane, ascending-inlet seed disc,
    # face-on arch camera). Explicit CLI --slice-*/--seed-*/--view still win.
    cap_dir = args.cap_dir or (os.path.dirname(args.wall) if args.wall else None)
    args._anat = derive_anatomy(wall_topo, cap_dir)
    if args._anat.get("note"):
        print("[anatomy] " + " | ".join(args._anat["note"]))
    elif args.view == "arch":
        print("[anatomy] no wall/caps -> --view arch falls back to iso, "
              "slice/seed use bbox defaults")

    stl_paths = []
    if args.stl_dir:
        stl_paths = sorted(glob.glob(os.path.join(args.stl_dir, "*_capped.stl")))
        print(f"{len(stl_paths)} phase STL ghosts found in {args.stl_dir}")

    results = {}
    if "motion" in types:
        results["motion"] = render_motion(files, wall_topo, idx, stl_paths, out_dir, args)
    if "velocity" in types:
        results["velocity"] = render_velocity(files, wall_topo, idx, out_dir, args)
    if "helicity" in types:
        results["helicity"] = render_helicity(files, wall_topo, idx, out_dir, args)
    if "fbmb" in types:
        if not args.fb_dir:
            print("[fbmb] --fb-dir not given -- skip")
        else:
            results["fbmb"] = render_fbmb(files, args.fb_dir, wall_topo, idx, out_dir, args)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"\n-> {out_dir}/")


if __name__ == "__main__":
    main()
