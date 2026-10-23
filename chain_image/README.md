# chain_image/ — wall motion from image registration (one segmented phase)

The second of the two wall-displacement chains. The first (`morph/`) registers a reference
surface onto **several segmented phases**. This one needs a lumen segmentation of **a single
phase**: the motion of every other phase is taken from the images themselves, by deformable
registration of the time-resolved acquisition (e.g. the 4D-flow anatomy/magnitude images
reconstructed per cardiac phase). It was used for the 20-phase case of the companion paper
(Supplementary S3.2).

Output: the solver's `Prescribed_displacement` file for the wall, built on the solver mesh's
own wall nodes. As in production, the interior of the mesh is then moved by the solver's mesh
equation; nothing here touches interior nodes.

## Steps

| # | Script | What it does |
|---|---|---|
| 0 | `dicom_to_phases.py` | (optional) split a time-resolved DICOM series into one volume per trigger time |
| 1 | `register_phases.py` | elastix B-spline registration, fixed = phase k, moving = reference phase; transform parameters per phase, reference segmentation warped to every phase, optional Dice / surface distance against manual segmentations of other phases |
| 2 | `invert_propagate.py` | **inverts** each transform at the wall nodes of the solver mesh (damped fixed point, residual < 1e-3 mm; non-convergent nodes filled by inverse-distance weighting and reported) → node-corresponding wall field, all phases |
| 3 | `smooth_wall_field.py` | Taubin smoothing of the displacement + local passes on folded triangles and on edge-strain outliers (> 0.8) over the interpolated cycle; reference phase kept exactly at zero |
| 4 | `taper_wall_field.py` | caps pinned, wall displacement faded in over a geodesic distance L = 12 mm from the cap rings (smoothstep), with an optional fold-proxy sweep over L — same construction as `morph/pin_caps.py` |
| 5 | `write_wall_displacement.py` | periodic PCHIP in time, simulation clock t = 0 at the reference phase, optional pre-roll, solver displacement file sampled at every time step of the run; optionally the inlet flow file on the same clock (negative = inflow) |

`wallfield.py` holds the shared pieces: the wall-field file format, the periodic PCHIP, the
wall-surface graph (Laplacian, Taubin, k-rings), the cycle scan for folds and edge strain, and
the transform inversion. `tools/phase_subset_wall.py` rebuilds the motion from a subset of the
phases (how many segmented phases does a run need?), and `tools/build_fb_at_phase.py` freezes a
rigid mesh at any phase of the same field.

Why invert? elastix returns a transform from the fixed image (phase k) to the moving image (the
reference phase). Warping the reference segmentation with it gives a label map at phase k, but a
label map has no node correspondence. Inverting the transform at the reference wall nodes gives
the position of each of those nodes at phase k: one node set, one topology, every phase.

## Example

```bash
python3 chain_image/dicom_to_phases.py dicom/anatomy/ phases/
python3 chain_image/register_phases.py --phases 'phases/phase*.nii.gz' --ref-index 10 \
    --ref-seg seg_ref.nrrd --out reg/ --validate 4:seg_p04.nrrd --validate 8:seg_p08.nrrd
#   build the solver mesh from the reference segmentation (tools/build_iso_mesh.py) ...
python3 chain_image/invert_propagate.py --reg-dir reg/ --wall-vtp mesh/mesh-surfaces/wall.vtp \
    --out wall_raw.npz --surface-stl lumen_ref.stl
python3 chain_image/smooth_wall_field.py --in wall_raw.npz --wall-vtp mesh/mesh-surfaces/wall.vtp \
    --period 0.9 --out wall_smooth.npz
python3 chain_image/taper_wall_field.py --in wall_smooth.npz --mesh mesh/mesh-complete.mesh.vtu \
    --surfaces mesh/mesh-surfaces --caps asc desc btca lcca lsa --length 12 --sweep 0 8 12 16 \
    --out wall_taper.npz
python3 chain_image/write_wall_displacement.py --in wall_taper.npz --mesh mesh/mesh-complete.mesh.vtu \
    --period 0.9 --dt 0.001 --cycles 3 --preroll 0.15 --out displacement_wall.txt \
    --flow-in inlet_flow.csv --flow-col 1 --flow-out flow_rate.txt
python3 tools/check_inlet_sign.py flow_rate.txt --t-cycle 0.9
```

The fluid boundary condition on the wall must be `Prescribed_displacement` with
`Impose_on_state_variable_integral`, the caps `Dir 0` in the mesh equation, and the inlet flow
negative (see `CONVENTIONS.md`). `MB_example.xml` shows the solver set-up.

## Verification

Run on the inputs of the companion paper's 20-phase case, steps 2-5 and
`tools/phase_subset_wall.py` reproduce the production files exactly: the inverted wall
positions (two phases checked against the production field), the smoothed and tapered fields,
the phase-subset fields, and the first 400 samples of the displacement file and the inlet flow
file byte for byte. The elastix registration (step 1) was not re-run. `tests/test_chain_image.py`
exercises steps 2-5 on synthetic data.

## Notes and assumptions

- Images, segmentations and the mesh must share the physical frame; `register_phases.py` and
  `invert_propagate.py` assume an axis-aligned image direction (physical = origin + spacing ×
  index) and warn otherwise.
- The registration parameters are those of the production run: B-spline, final control-point
  spacing 12 mm, 600 iterations, 4,000 spatial samples, crop = reference lumen bounding box +
  25 mm. `--validate K:SEG` compares the warped reference with a manual segmentation of phase K
  and also reports the *static* baseline (the unwarped reference against the same manual
  segmentation): a registration that does not beat the static baseline on a phase has not
  captured that phase's motion. Agreement of the final wall with the images is best checked
  independently, on manual contours of a few phases.
- Phases are assumed evenly spaced over the cycle. The cycle period is the one of the flow
  measurement; the solver's displacement file is sampled over the whole run, so its own period
  never matters.
- Dependencies beyond `requirements.txt`: `itk-elastix`, `SimpleITK` (steps 1–2), `pydicom`
  (step 0). Steps 3–5 need only numpy, scipy and pyvista.
