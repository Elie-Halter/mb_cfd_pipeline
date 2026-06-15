#!/usr/bin/env python3
"""
Figure scaffold for the paper (FB vs MB, and segmentation sensitivity).

Main input: the comparison VTP produced by compare_FB_MB.py
(fields FB_TAWSS, MB_TAWSS, dTAWSS, dTAWSS_pct, FB_OSI, MB_OSI, dOSI).

Outputs (in --out-dir):
  - fig_tawss_hist.png      dTAWSS histogram (matplotlib, headless OK)
  - fig_tawss_scatter.png   FB vs MB scatter (+ y=x, correlation)
  - fig_osi_hist.png        dOSI histogram
  - fig_split_bar.png       simulated vs MRI flow split (if --split given)
  - fig_wall_*.png          3D wall maps (FB/MB/delta) -- best-effort (needs OpenGL/xvfb)

The matplotlib figures work WITHOUT a GPU (Agg backend). The 3D pyvista maps need a GL
context: on a headless machine, run via `xvfb-run -a python3 ...` (the script tries
start_xvfb automatically and continues without crashing if it fails).

Usage:
  python3 make_figures.py --wall-vtp cmp_P0001_FB_vs_MB_wall.vtp --out-dir figs/ \
      [--split split.json] [--label P0001]

  # compare TWO segmentations (same machinery: pass the VTP from
  #   compare_FB_MB.py --fb segA ... --mb segB ...  -> "MB_"=segB, "FB_"=segA)
  python3 make_figures.py --wall-vtp cmp_segA_vs_segB_wall.vtp --out-dir figs_seg/ \
      --label "seg A vs seg B" --names "segA,segB"
"""
import argparse, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless, no GPU required
import matplotlib.pyplot as plt
import pyvista as pv


def mpl_figures(w, out, label, names):
    a, b = names
    fb_t = np.asarray(w["FB_TAWSS"]); mb_t = np.asarray(w["MB_TAWSS"]); d_t = np.asarray(w["dTAWSS"])
    fb_o = np.asarray(w["FB_OSI"]);   mb_o = np.asarray(w["MB_OSI"]);   d_o = np.asarray(w["dOSI"])

    # 1) dTAWSS histogram
    plt.figure(figsize=(5, 3.5))
    plt.hist(d_t, bins=60, color="#4477aa")
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(f"dTAWSS = {b} - {a}  [dyne/cm^2]"); plt.ylabel("wall nodes")
    plt.title(f"{label}: dTAWSS (med {np.median(d_t):+.2f})")
    plt.tight_layout(); plt.savefig(f"{out}/fig_tawss_hist.png", dpi=160); plt.close()

    # 2) FB vs MB scatter (TAWSS)
    corr = float(np.corrcoef(fb_t, mb_t)[0, 1])
    lim = float(max(fb_t.max(), mb_t.max()))
    plt.figure(figsize=(4.2, 4.2))
    plt.scatter(fb_t, mb_t, s=2, alpha=0.25, color="#4477aa")
    plt.plot([0, lim], [0, lim], "k--", lw=0.8)
    plt.xlabel(f"TAWSS {a}"); plt.ylabel(f"TAWSS {b}")
    plt.title(f"corr {corr:+.3f}"); plt.axis("square")
    plt.tight_layout(); plt.savefig(f"{out}/fig_tawss_scatter.png", dpi=160); plt.close()

    # 3) dOSI histogram
    plt.figure(figsize=(5, 3.5))
    plt.hist(d_o, bins=60, color="#aa6644")
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(f"dOSI = {b} - {a}  [-]"); plt.ylabel("wall nodes")
    plt.title(f"{label}: dOSI (med {np.median(d_o):+.4f})")
    plt.tight_layout(); plt.savefig(f"{out}/fig_osi_hist.png", dpi=160); plt.close()
    print(f"  matplotlib: 3 figures (TAWSS/OSI hist, scatter) -> {out}")


def split_bar(split_json, out, label):
    with open(split_json) as f:
        d = json.load(f)                    # {"sim": {"desc":.., ...}, "mri": {...}}
    sim, mri = d["sim"], d.get("mri", {})
    keys = list(sim.keys())
    x = np.arange(len(keys)); wdt = 0.38
    plt.figure(figsize=(5, 3.5))
    plt.bar(x - wdt/2, [sim[k] for k in keys], wdt, label="simulated", color="#4477aa")
    if mri:
        plt.bar(x + wdt/2, [mri.get(k, 0) for k in keys], wdt, label="MRI", color="#ccaa44")
    plt.xticks(x, keys); plt.ylabel("flow split [%]"); plt.legend()
    plt.title(f"{label}: split vs MRI")
    plt.tight_layout(); plt.savefig(f"{out}/fig_split_bar.png", dpi=160); plt.close()
    print(f"  split bar -> {out}/fig_split_bar.png")


def wall_maps(w, out, names):
    """3D wall maps -- best-effort (OpenGL/xvfb required)."""
    try:
        try:
            pv.start_xvfb()
        except Exception:
            pass
        pv.OFF_SCREEN = True
        fields = [("FB_TAWSS", "TAWSS " + names[0], "hot"),
                  ("MB_TAWSS", "TAWSS " + names[1], "hot"),
                  ("dTAWSS", "dTAWSS", "coolwarm"),
                  ("FB_OSI", "OSI " + names[0], "viridis"),
                  ("MB_OSI", "OSI " + names[1], "viridis"),
                  ("dOSI", "dOSI", "coolwarm")]
        n = 0
        for fld, title, cmap in fields:
            if fld not in w.point_data:
                continue
            pl = pv.Plotter(off_screen=True, window_size=(900, 700))
            sym = fld.startswith("d")
            pl.add_mesh(w, scalars=fld, cmap=cmap, scalar_bar_args={"title": title},
                        clim=(-np.percentile(np.abs(w[fld]), 95), np.percentile(np.abs(w[fld]), 95)) if sym else None)
            pl.view_xz(); pl.screenshot(f"{out}/fig_wall_{fld}.png"); pl.close(); n += 1
        print(f"  pyvista: {n} 3D wall maps -> {out}")
    except Exception as e:
        print(f"  [skip 3D maps] GL rendering unavailable ({type(e).__name__}: {e}).")
        print(f"        -> rerun via: xvfb-run -a python3 tools/make_figures.py ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-vtp", required=True)
    ap.add_argument("--out-dir", default="figs")
    ap.add_argument("--split", default=None, help="JSON {sim:{...}, mri:{...}} for the bar chart")
    ap.add_argument("--label", default="FB vs MB")
    ap.add_argument("--names", default="FB,MB", help="names of the two cases (e.g. 'segA,segB')")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = [s.strip() for s in a.names.split(",")][:2]
    w = pv.read(a.wall_vtp)
    need = {"FB_TAWSS", "MB_TAWSS", "dTAWSS", "FB_OSI", "MB_OSI", "dOSI"}
    miss = need - set(w.point_data.keys())
    if miss:
        raise SystemExit(f"missing fields in {a.wall_vtp}: {miss} "
                         f"(produce the VTP with compare_FB_MB.py)")
    mpl_figures(w, a.out_dir, a.label, names)
    if a.split:
        split_bar(a.split, a.out_dir, a.label)
    wall_maps(w, a.out_dir, names)
    print(f"\n-> figures in {a.out_dir}/")


if __name__ == "__main__":
    main()
