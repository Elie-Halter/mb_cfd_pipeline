#!/usr/bin/env python3
"""
Derive a case's FB and MB svMultiPhysics XML files from the template XMLs (FB_example.xml,
MB_example.xml, the production configuration), repointing mesh / surfaces / wall displacement /
scale / number of steps. Substitution via ElementTree (robust -- not sed); the template comments
are preserved.

Assumption: the case has the SAME face names as the template (asc/desc/btca/lcca/lsa/wall).
The RCR R/C values stay as in the template -> fill them in from calibrate_rcr.py (the
calibration is per case; the MRI split is not guessed here). The inflow file keeps the
template's relative name (inflow.txt, staged into the run directory) unless --inflow is given.

The MB output is checked against the production configuration (CONVENTIONS.md sections 3-4):
  * fluid wall BC = Prescribed_displacement with Impose_on_state_variable_integral true;
  * mesh equation: wall = Prescribed_displacement, every other face (the caps) = Dir 0;
  * the displacement file is a STANDARD (wall-only) file: an EXTENDED all-node file from
    morph/ is refused -- convert it with tools/extended_to_standard.py --face wall.vtp.
--no-production-check skips these checks (for non-production experiments only).

Usage:
  python3 make_patient_xml.py --fb-template FB_example.xml --mb-template MB_example.xml \
      --mesh .../mesh-complete.mesh.vtu --surf .../mesh-surfaces --disp .../wall_displacement.txt \
      --scale 0.1 --nsteps 3896 --fb-out X_FB.xml --mb-out X_MB.xml [--inflow inflow.txt]
"""
import argparse, os
import xml.etree.ElementTree as ET


def _parser():
    # preserve comments
    return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))


def _txt(el, tag):
    c = el.find(tag)
    return (c.text or "").strip() if c is not None else None


def check_production_mb(root):
    """Return the list of deviations of an MB tree from the production configuration."""
    errs = []
    faces = [f.get("name") for f in root.iter("Add_face")]
    eqs = {e.get("type"): e for e in root.findall("Add_equation")}
    fluid, mesh = eqs.get("fluid"), eqs.get("mesh")
    if fluid is None or mesh is None:
        return ["MB XML needs both a fluid and a mesh equation"]
    fbc = {b.get("name"): b for b in fluid.findall("Add_BC")}
    w = fbc.get("wall")
    if w is None or _txt(w, "Type") != "Prescribed_displacement" \
            or _txt(w, "Impose_on_state_variable_integral") != "true":
        errs.append("fluid wall BC must be Prescribed_displacement with "
                    "Impose_on_state_variable_integral true (not Dirichlet 0)")
    mbc = {b.get("name"): b for b in mesh.findall("Add_BC")}
    w = mbc.get("wall")
    if w is None or _txt(w, "Type") != "Prescribed_displacement":
        errs.append("mesh-equation wall BC must be Prescribed_displacement")
    for f in faces:
        if f == "wall":
            continue
        b = mbc.get(f)
        ok = b is not None and _txt(b, "Type") in ("Dir", "Dirichlet")
        if ok:
            try:
                ok = float(_txt(b, "Value")) == 0.0
            except (TypeError, ValueError):
                ok = False
        if not ok:
            errs.append(f"mesh-equation BC on cap '{f}' must be Dir 0 (caps pinned)")
    return errs


def check_standard_file(path):
    """Refuse an EXTENDED all-node displacement file (production prescribes the wall only)."""
    if path is None or not os.path.isfile(path):
        return None
    with open(path) as f:
        f.readline()
        if f.readline().strip() == "EXTENDED":
            return (f"{path} is an EXTENDED all-node file; production prescribes the wall only: "
                    f"python3 tools/extended_to_standard.py {path} wall_displacement.txt "
                    f"--face <mesh-surfaces>/wall.vtp")
    return None


def retarget(template, out, mesh, surf, disp, scale, nsteps, inflow=None, check=False):
    tree = ET.parse(template, parser=_parser())
    root = tree.getroot()
    n_face = 0
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str):
            continue
        if tag == "Mesh_file_path":
            el.text = f" {mesh} "
        elif tag == "Face_file_path":
            el.text = f" {os.path.join(surf, os.path.basename(el.text.strip()))} "
            n_face += 1
        elif tag == "Mesh_scale_factor":
            el.text = f" {scale} "
        elif tag == "Number_of_time_steps" and nsteps is not None:
            el.text = f" {nsteps} "
        elif tag == "Prescribed_displacement_file_path" and disp is not None:
            el.text = f" {disp} "
        elif tag == "Temporal_values_file_path" and inflow is not None:
            el.text = f" {inflow} "
    if check:
        errs = check_production_mb(root)
        e = check_standard_file(disp)
        if e:
            errs.append(e)
        if errs:
            raise SystemExit(f"{template}: not the production MB configuration:\n  - "
                             + "\n  - ".join(errs)
                             + "\n(--no-production-check to override)")
    tree.write(out, encoding="unicode", xml_declaration=False)
    print(f"  wrote {out}  (mesh, {n_face} faces, scale={scale}"
          + (f", nsteps={nsteps}" if nsteps else "")
          + (", wall displacement" if disp else "")
          + (", production checks passed" if check else "") + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fb-template", required=True)
    ap.add_argument("--mb-template", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--surf", required=True)
    ap.add_argument("--disp", required=True, help="STANDARD wall-only displacement file")
    ap.add_argument("--scale", default="0.1")
    ap.add_argument("--nsteps", default=None)
    ap.add_argument("--inflow", default=None, help="inflow file path (default: keep the template's)")
    ap.add_argument("--fb-out", required=True)
    ap.add_argument("--mb-out", required=True)
    ap.add_argument("--no-production-check", action="store_true")
    a = ap.parse_args()
    # FB: no displacement (rigid wall); MB: prescribed wall displacement, caps pinned
    retarget(a.fb_template, a.fb_out, a.mesh, a.surf, None, a.scale, a.nsteps, a.inflow)
    retarget(a.mb_template, a.mb_out, a.mesh, a.surf, a.disp, a.scale, a.nsteps, a.inflow,
             check=not a.no_production_check)
    print("  NOTE: fill in the case's RCR R/C values in the Neumann <Add_BC> blocks (see calibrate_rcr).")


if __name__ == "__main__":
    main()
