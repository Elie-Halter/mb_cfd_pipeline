#!/bin/bash
# ==============================================================================
# Verifies that all MB-CFD pipeline dependencies are in place
# Usage: bash check_install.sh
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

ok()   { echo -e "${GREEN}[OK]${NC}   $*";   PASS=$((PASS+1)); }
fail() { echo -e "${RED}[FAIL]${NC} $*";     FAIL=$((FAIL+1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; WARN=$((WARN+1)); }

echo "============================================================"
echo "  MB-CFD pipeline install verification"
echo "============================================================"
echo ""

# ------------------------------------------------------------------------------
# 1. System
# ------------------------------------------------------------------------------
echo "--- System ---"

if grep -qE "Ubuntu (22|24)" /etc/os-release; then
    ok "Ubuntu $(grep VERSION_ID /etc/os-release | cut -d'"' -f2)"
else
    warn "Untested OS: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2)"
fi

CORES=$(nproc)
if [ "$CORES" -ge 4 ]; then
    ok "Cores: $CORES"
else
    warn "Cores: $CORES (4+ recommended for MPI)"
fi

RAM_GB=$(free -g | awk 'NR==2 {print $2}')
if [ "$RAM_GB" -ge 8 ]; then
    ok "RAM: ${RAM_GB} GB"
else
    warn "RAM: ${RAM_GB} GB (8+ recommended)"
fi

DISK_AVAIL=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$DISK_AVAIL" -ge 5 ]; then
    ok "Free disk: ${DISK_AVAIL} GB"
else
    warn "Free disk: ${DISK_AVAIL} GB (5+ recommended)"
fi

# ------------------------------------------------------------------------------
# 2. Binaries
# ------------------------------------------------------------------------------
echo ""
echo "--- Binaries ---"

if command -v mpirun > /dev/null; then
    MPI_VER=$(mpirun --version 2>&1 | head -1)
    ok "OpenMPI: $MPI_VER"
else
    fail "OpenMPI not found (sudo apt install openmpi-bin libopenmpi-dev)"
fi

if [ -x /usr/local/bin/mmg3d_O3 ]; then
    if ldd /usr/local/bin/mmg3d_O3 2>&1 | grep -q "libElas.*not found"; then
        fail "mmg3d_O3 found but libElas.so NOT FOUND (check LD_LIBRARY_PATH=\$HOME/lib)"
    elif ldd /usr/local/bin/mmg3d_O3 2>&1 | grep -q libElas; then
        ok "MMG3D with Elas: /usr/local/bin/mmg3d_O3"
    else
        warn "MMG3D present but WITHOUT Elas (USE_ELAS=ON required for -lag)"
    fi
else
    fail "mmg3d_O3 not found (re-run install.sh)"
fi

if [ -x "$HOME/svmp_bin/svmultiphysics" ]; then
    if ldd "$HOME/svmp_bin/svmultiphysics" 2>&1 | grep -q "not found"; then
        fail "svmultiphysics: missing libs:"
        ldd "$HOME/svmp_bin/svmultiphysics" 2>&1 | grep "not found" | sed 's/^/      /'
    else
        ok "svMultiPhysics: $HOME/svmp_bin/svmultiphysics"
    fi
else
    fail "svmultiphysics not found (re-run install.sh)"
fi

# ------------------------------------------------------------------------------
# 3. ISCDtoolbox libraries
# ------------------------------------------------------------------------------
echo ""
echo "--- Libraries ---"

[ -f "$HOME/lib/libCommons.so" ] && ok "libCommons.so: $HOME/lib/" || fail "libCommons.so not found"
[ -f "$HOME/lib/libElas.so" ] && ok "libElas.so: $HOME/lib/" || fail "libElas.so not found"

if echo "$LD_LIBRARY_PATH" | grep -q "$HOME/lib"; then
    ok "LD_LIBRARY_PATH contains \$HOME/lib"
else
    warn "LD_LIBRARY_PATH does not contain \$HOME/lib (run: source ~/.bashrc)"
fi

# ------------------------------------------------------------------------------
# 4. Python
# ------------------------------------------------------------------------------
echo ""
echo "--- Python ---"

if command -v python3 > /dev/null; then
    PY_VER=$(python3 --version)
    ok "Python: $PY_VER"
else
    fail "Python3 not found"
fi

for pkg in numpy scipy meshio vtk yaml; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VER=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        ok "Python: $pkg $VER"
    else
        fail "Python: $pkg missing (pip install $pkg)"
    fi
done

# ------------------------------------------------------------------------------
# 5. Repo
# ------------------------------------------------------------------------------
echo ""
echo "--- Pipeline ---"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for f in morph/pipeline.py morph/register.py morph/morph_volume.py \
         tools/hemo_indices.py tools/compare_FB_MB.py run_patient.sh run_MB_aniso.sh; do
    [ -f "$SCRIPT_DIR/$f" ] && ok "Code: $f" || fail "Missing code: $f"
done

[ -f "$SCRIPT_DIR/patients/TEMPLATE.env" ] && ok "patients/TEMPLATE.env" || warn "patients/TEMPLATE.env missing"

# Patches
N_PATCHES=$(ls "$SCRIPT_DIR/patches/svMP/"*.patch 2>/dev/null | wc -l)
if [ "$N_PATCHES" -gt 0 ]; then
    ok "svMP patches: $N_PATCHES files"
else
    warn "svMP patches: none found in patches/svMP/"
fi

# ------------------------------------------------------------------------------
# 6. Data
# ------------------------------------------------------------------------------
echo ""
echo "--- Patient data ---"

if [ -d "$SCRIPT_DIR/data/original_mesh" ] && [ -f "$SCRIPT_DIR/data/original_mesh/mesh-complete.mesh.vtu" ]; then
    ok "Original mesh present"
else
    warn "No mesh: provide your patient data under data/"
fi

if [ -d "$SCRIPT_DIR/data/stl_phases" ]; then
    N_STL=$(ls "$SCRIPT_DIR/data/stl_phases/"*.stl 2>/dev/null | wc -l)
    if [ "$N_STL" -ge 4 ]; then
        ok "STL phases: $N_STL files"
    else
        warn "STL phases: $N_STL files (4 expected)"
    fi
else
    warn "No stl_phases/: provide your patient data under data/"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo -e "  ${GREEN}OK   : $PASS${NC}"
echo -e "  ${YELLOW}WARN : $WARN${NC}"
echo -e "  ${RED}FAIL : $FAIL${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Installation incomplete. Re-run: bash install.sh${NC}"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo -e "${YELLOW}Installation OK with warnings (usually missing patient data).${NC}"
    exit 0
else
    echo -e "${GREEN}All good, ready to go.${NC}"
    exit 0
fi
