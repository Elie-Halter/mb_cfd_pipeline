# legacy/ — solver patches of versions <= 0.3 (not used)

Up to version 0.3 the pipeline ran a patched svMultiPhysics (commit `97ef512` + the fifteen patches in
`patches/svMP/`), with the moving wall imposed through a `Prescribed_displacement` boundary condition added
by the patches. Since version 1.0 it runs the **official, unmodified svMultiPhysics** (commit `86729244`):
the moving wall goes through the FSI equation with a fluid domain only plus the mesh equation, and
`tools/to_official_solver.py` writes that input. The analytic benchmarks give the same results with both
(`benchmarks/README.md`).

The patches and `benchmarks/build_nogcl.sh` (a contrast build without patch 0014) are kept here for the
record only; `install.sh`, the Docker and Apptainer recipes no longer apply them.
