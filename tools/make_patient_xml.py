#!/usr/bin/env python3
"""
Derive a patient's FB and MB svMP XML files from template XMLs, repointing
mesh / surfaces / displacement / scale / number of steps. Substitution via
ElementTree (robust -- not sed). The template comments are preserved.

Assumption: the patient has the SAME cap names as the template (asc/desc/btca/lcca/lsa/wall).
The RCR R/C values stay as in the template -> to be filled in by hand from calibrate_rcr
(the calibration is per patient; we do not guess the MRI split here).

Usage:
  python3 make_patient_xml.py --fb-template FB_example.xml --mb-template MB_example.xml \
      --mesh .../mesh-complete.mesh.vtu --surf .../mesh-surfaces --disp .../displacement_X.txt \
      --scale 0.1 --nsteps 1948 --fb-out X_FB.xml --mb-out X_MB.xml
"""
import argparse, os
import xml.etree.ElementTree as ET


def _parser():
    # preserve comments
    return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))


def retarget(template, out, mesh, surf, disp, scale, nsteps):
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
    tree.write(out, encoding="unicode", xml_declaration=False)
    print(f"  wrote {out}  (mesh, {n_face} faces, scale={scale}"
          + (f", nsteps={nsteps}" if nsteps else "")
          + (", displacement" if disp and 'MB' in os.path.basename(out).upper() else "") + ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb-template", required=True)
    ap.add_argument("--mb-template", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--surf", required=True)
    ap.add_argument("--disp", required=True)
    ap.add_argument("--scale", default="0.1")
    ap.add_argument("--nsteps", default=None)
    ap.add_argument("--fb-out", required=True)
    ap.add_argument("--mb-out", required=True)
    a = ap.parse_args()
    # FB: no displacement (rigid wall); MB: prescribed displacement
    retarget(a.fb_template, a.fb_out, a.mesh, a.surf, None, a.scale, a.nsteps)
    retarget(a.mb_template, a.mb_out, a.mesh, a.surf, a.disp, a.scale, a.nsteps)
    print("  NOTE: fill in the patient's RCR R/C values in the Neumann <Add_BC> blocks (see calibrate_rcr).")


if __name__ == "__main__":
    main()
