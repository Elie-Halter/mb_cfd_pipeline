# REPRODUCE — pipeline ALE moving-boundary aortique (open-source, sans remaillage)

Guide pour repartir de zéro : construire le solveur patché, lancer un patient (rigide FB + paroi
mobile MB), post-traiter. Testé Ubuntu 22/24.

## 0. Prérequis
- Linux, ~30-60 min de build, ~8 Go RAM, MPI (OpenMPI), CMake, gfortran, Python 3.10+.
- Un patient = une géométrie segmentée (STL par phase du cycle) + un maillage volumique tétraédrique.

## 1. Cloner + installer
```bash
git clone https://github.com/Elie-Halter/mb_cfd_pipeline.git
cd mb_cfd_pipeline
bash install.sh          # clone svMultiPhysics @ 97ef512, applique patches/svMP/*.patch (git am), build
source ~/.bashrc
bash check_install.sh    # verifie le binaire + les deps Python
```
`install.sh` reconstruit le **solveur patché** (15 patches, dont GCL 0014 + anneaux 0015) depuis un
commit officiel épinglé — **le fork n'a pas besoin d'être cloné séparément**.

## 2. Préparer un patient
- Placer les STL par phase du cycle + le maillage tétraédrique tout-tét (voir `morph/README.md`
  pour la génération de maillage iso RBM sans couche limite, `build_iso_mesh.py`).
- Copier `patients/TEMPLATE.env` → `patients/<sujet>.env`, renseigner : chemins STL/maillage,
  faces (wall/asc/desc/branches), débit inlet, RCR par sortie, dt, T_cycle, n_procs.
- Le morph (registration N-ICP + rest-shape J>0) produit le fichier de déplacement pariétal :
  voir `morph/pipeline.py`.

## 3. Lancer
```bash
# pipeline complet par patient (morph -> FB -> MB) :
bash run_patient.sh patients/<sujet>.env
# ou un cas isolé (exemples fournis) :
mpirun -np 4 ~/svmp_bin/svmultiphysics FB_example.xml     # rigide (baseline)
mpirun -np 4 ~/svmp_bin/svmultiphysics MB_example.xml     # paroi mobile (wall-only)
```
⚠️ **Jamais 2 svMultiPhysics simultanés** sur une machine 4-procs (RAM). Runs sériels.

## 4. Post-traiter (dossier `tools/`)
```bash
python3 tools/extract_flowsplit_FB.py   <run>/4-procs  <mesh-surfaces>     # split debit vs cibles MRI
python3 tools/mass_balance_ale.py --vtu-glob "<run>/4-procs/results_*.vtu" \
        --surf-dir <mesh-surfaces> --dt 0.001 --cycle-range 974,1948        # bilan de masse ALE
python3 tools/hemo_indices.py <run>/4-procs --wall <wall.vtp> --cycle-start 974 --cycle-end 1948
python3 tools/compare_FB_MB.py           # Delta FB (rigide) vs MB (mobile) : pression/vitesse/helicite
```
Mapping paroi→volume par **GlobalNodeID−1** (le VTU svMP garde l'ordre du maillage), PAS de KDTree
positionnel (faux en frontière mobile).

## 5. Tests
```bash
bash tests/run_tests.sh   # mecanisme prescribed-disp, GCL, bilan de masse, deplacement hors-ligne
```
Faire passer TOUS les tests courts AVANT tout run long.

## Détails
- Solveur patché : `patches/svMP/README.md` (verdict d'utilité par patch).
- Morph & maillage : `morph/README.md`. Post-traitement : `tools/README.md`.
- Méthode : svMultiPhysics tout-tét + morph prescrit image-dérivé, **sans remaillage ni
  interpolation**, GCL-consistant (patch 0014). FB et MB sur le MÊME maillage → Δ nœud-à-nœud.
