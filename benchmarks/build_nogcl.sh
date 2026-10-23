#!/usr/bin/env bash
# =============================================================================
# build_nogcl.sh -- build the CONTRAST solver for the DGCL benchmark: the same svMultiPhysics
# commit and the same patches as install.sh, EXCEPT patch 0014 (fluid assembly on the deformed
# geometry). Nothing else differs, so B1/B2 run with both binaries isolate the effect of 0014.
#
# Usage:
#   bash benchmarks/build_nogcl.sh [BUILD_DIR]      (default BUILD_DIR: $HOME/build_mb_cfd_nogcl)
# Result:
#   $HOME/svmp_bin/svmultiphysics_nogcl   -> then: SVMP_BIN_NOGCL=... bash benchmarks/run_all.sh
# Requires the system dependencies installed by install.sh (compilers, MPI, VTK, PETSc).
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$HOME/build_mb_cfd_nogcl}"
OUT_BIN="${SVMP_BIN_NOGCL:-$HOME/svmp_bin/svmultiphysics_nogcl}"
BASE_COMMIT=$(grep -m1 '^SVMP_BASE_COMMIT=' "$REPO/install.sh" | cut -d'"' -f2)
[ -n "$BASE_COMMIT" ] || { echo "cannot read SVMP_BASE_COMMIT from install.sh" >&2; exit 1; }

mkdir -p "$BUILD_DIR"; cd "$BUILD_DIR"
[ -d svMultiPhysics ] || git clone --quiet https://github.com/SimVascular/svMultiPhysics.git
cd svMultiPhysics
git am --abort 2>/dev/null || true
git checkout --quiet --force "$BASE_COMMIT"
git config user.email "build@local"; git config user.name "nogcl builder"
for p in "$REPO"/patches/svMP/*.patch; do
  case "$(basename "$p")" in
    0014-*) echo "  skip  $(basename "$p")  (the patch under test)"; continue ;;
  esac
  echo "  apply $(basename "$p")"
  git am --quiet "$p"
done

PETSC_FLAG=""
[ -e "$HOME/petsc_svmp/lib/libpetsc.so" ] && PETSC_FLAG="-DSV_PETSC_DIR=$HOME/petsc_svmp"
mkdir -p build && cd build
cmake -DSV_USE_MPI=ON $PETSC_FLAG -DCMAKE_BUILD_TYPE=Release .. > cmake.log 2>&1
make -j"$(nproc)" svMultiPhysics > make.log 2>&1
BUILT=$(find . -name svmultiphysics -executable -type f | head -1)
[ -n "$BUILT" ] || { echo "binary not found after build (see make.log)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT_BIN")"
cp "$BUILT" "$OUT_BIN"
if [ -x "$HOME/svmp_bin/svmultiphysics" ] && cmp -s "$OUT_BIN" "$HOME/svmp_bin/svmultiphysics"; then
  echo "ERROR: the contrast binary is identical to the production binary" >&2; exit 1
fi
echo "built $OUT_BIN (base $BASE_COMMIT, all patches except 0014)"
