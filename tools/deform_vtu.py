#!/usr/bin/env python3
"""deform_vtu.py -- write a copy of the results_*.vtu of a moving-wall run in the DEFORMED configuration.

The official svMultiPhysics writes its output on the reference coordinates and stores the mesh motion in the
`Displacement` array (same length unit as the mesh). Points + Displacement give the deformed mesh, needed for any
geometric post-processing (volumes, sections, fluxes). Rigid-wall runs are unchanged.
Usage: python3 deform_vtu.py RUN_OUTPUT_DIR OUT_DIR
"""
import sys, os, glob
import numpy as np, pyvista as pv

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "results_*.vtu")))
for f in files:
    m = pv.read(f)
    if "Displacement" in m.point_data:
        m.points = np.asarray(m.points) + np.asarray(m.point_data["Displacement"])
    m.save(os.path.join(dst, os.path.basename(f)))
print(len(files), "VTU ->", dst)
