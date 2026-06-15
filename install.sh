#!/bin/bash
# ==============================================================================
# Full installation of the MB-CFD pipeline (svMultiPhysics + MMG3D)
# Ubuntu 22.04 / 24.04
# ==============================================================================
# This script installs:
#  1. System dependencies (compilers, MPI, OpenBLAS, Python)
#  2. Commons (ISCDtoolbox)
#  3. LinearElasticity / Elas (ISCDtoolbox)
#  4. MMG3D with USE_ELAS=ON
#  5. svMultiPhysics from source with the patches
#  6. Python dependencies (numpy, scipy, meshio, vtk, pyyaml)
#  7. LD_LIBRARY_PATH configuration
# ==============================================================================

set -e  # exit on error
set -u  # exit on undefined variable

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_PREFIX="$HOME"
BUILD_DIR="$HOME/build_mb_cfd"
LOG_FILE="$SCRIPT_DIR/install.log"

# Official svMP commit the patches apply on
SVMP_BASE_COMMIT="97ef51223e5a079bdee018bd9c0490c861c07fe0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*" | tee -a "$LOG_FILE"
}

ok() {
    echo -e "${GREEN}[OK]${NC} $*" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

err() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
    exit 1
}

section() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}  $*${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
}

# ------------------------------------------------------------------------------
# Step 0: Pre-flight checks
# ------------------------------------------------------------------------------
section "Step 0/7: Pre-flight checks"

# Reset log
echo "Installation log $(date)" > "$LOG_FILE"

# OS check
if ! grep -qE "Ubuntu (22|24)" /etc/os-release; then
    warn "Untested OS (Ubuntu 22/24 recommended). Continue? [y/N]"
    read -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi
ok "OS: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2)"

# Disk space (at least 10 GB)
DISK_AVAIL=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$DISK_AVAIL" -lt 10 ]; then
    err "Insufficient disk space: ${DISK_AVAIL}G available (10G minimum)"
fi
ok "Disk space: ${DISK_AVAIL}G available"

# Cores
CORES=$(nproc)
ok "Available cores: $CORES"

# RAM
RAM_GB=$(free -g | awk 'NR==2 {print $2}')
ok "RAM: ${RAM_GB}G"
if [ "$RAM_GB" -lt 4 ]; then
    warn "Low RAM (<4G). Compilation may be slow."
fi

# sudo available?
if ! sudo -n true 2>/dev/null; then
    log "sudo is required to install system packages."
    sudo -v || err "sudo not available"
fi

# Working dir
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ------------------------------------------------------------------------------
# Step 1: System dependencies
# ------------------------------------------------------------------------------
section "Step 1/7: System dependencies (apt)"

log "Updating apt..."
sudo apt-get update -qq

log "Installing packages..."
sudo apt-get install -y -q \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    openmpi-bin \
    libopenmpi-dev \
    libopenblas0 \
    libopenblas-dev \
    libvtk9-dev \
    libxml2-dev \
    libxslt-dev \
    libpetsc-real-dev \
    pkg-config \
    || err "apt install failed"

ok "System dependencies installed"

# ------------------------------------------------------------------------------
# Step 2: Commons (ISCDtoolbox)
# ------------------------------------------------------------------------------
section "Step 2/7: Commons (ISCDtoolbox)"

if [ -f "$INSTALL_PREFIX/lib/libCommons.so" ]; then
    ok "Commons already installed (libCommons.so found)"
else
    cd "$BUILD_DIR"
    if [ ! -d Commons ]; then
        log "Cloning Commons..."
        git clone --quiet https://github.com/ISCDtoolbox/Commons.git
    fi
    cd Commons
    mkdir -p build
    cd build
    log "CMake Commons..."
    cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" .. > /dev/null 2>&1
    log "Building Commons..."
    make -j"$CORES" > /dev/null 2>&1
    log "Installing Commons..."
    make install > /dev/null 2>&1

    [ -f "$INSTALL_PREFIX/lib/libCommons.so" ] || err "libCommons.so not found after install"
    ok "Commons installed in $INSTALL_PREFIX/lib/"
fi

# ------------------------------------------------------------------------------
# Step 3: LinearElasticity (Elas)
# ------------------------------------------------------------------------------
section "Step 3/7: LinearElasticity (Elas)"

if [ -f "$INSTALL_PREFIX/lib/libElas.so" ]; then
    ok "Elas already installed (libElas.so found)"
else
    cd "$BUILD_DIR"
    if [ ! -d LinearElasticity ]; then
        log "Cloning LinearElasticity..."
        git clone --quiet https://github.com/ISCDtoolbox/LinearElasticity.git
    fi
    cd LinearElasticity
    mkdir -p build
    cd build
    log "CMake LinearElasticity (with Commons from $INSTALL_PREFIX)..."
    cmake \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCOMMONS_DIR="$INSTALL_PREFIX" \
        -DCOMMONS_INCLUDE_DIR="$INSTALL_PREFIX/include" \
        -DCOMMONS_LIBRARY="$INSTALL_PREFIX/lib/libCommons.so" \
        -DCMAKE_C_FLAGS="-I$INSTALL_PREFIX/include" \
        .. > /dev/null 2>&1
    log "Building LinearElasticity..."
    make -j"$CORES" > /dev/null 2>&1
    make install > /dev/null 2>&1

    [ -f "$INSTALL_PREFIX/lib/libElas.so" ] || err "libElas.so not found after install"
    ok "Elas installed in $INSTALL_PREFIX/lib/"
fi

# For MMG: ELAS_DIR must point to the build directory
ELAS_BUILD_DIR="$BUILD_DIR/LinearElasticity/build"

# ------------------------------------------------------------------------------
# Step 4: MMG3D with USE_ELAS=ON
# ------------------------------------------------------------------------------
section "Step 4/7: MMG3D"

if [ -f /usr/local/bin/mmg3d_O3 ] && ldd /usr/local/bin/mmg3d_O3 2>/dev/null | grep -q libElas; then
    ok "MMG3D already installed with USE_ELAS"
else
    cd "$BUILD_DIR"
    if [ ! -d mmg ]; then
        log "Cloning MMG..."
        git clone --quiet https://github.com/MmgTools/mmg.git
    fi
    cd mmg
    mkdir -p build
    cd build
    rm -rf ./*

    log "CMake MMG (USE_ELAS=ON)..."
    cmake \
        -DUSE_ELAS=ON \
        -DELAS_DIR="$ELAS_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        .. > /dev/null 2>&1
    log "Building MMG..."
    make -j"$CORES" > /dev/null 2>&1
    log "Installing MMG (sudo)..."
    sudo make install > /dev/null 2>&1

    [ -f /usr/local/bin/mmg3d_O3 ] || err "mmg3d_O3 not found after install"
    ok "MMG3D installed: /usr/local/bin/mmg3d_O3"
fi

# ------------------------------------------------------------------------------
# Step 5: svMultiPhysics from source with patches
# ------------------------------------------------------------------------------
section "Step 5/7: svMultiPhysics"

SVMP_DIR="$BUILD_DIR/svMultiPhysics"
SVMP_BIN_DIR="$HOME/svmp_bin"

if [ -x "$SVMP_BIN_DIR/svmultiphysics" ]; then
    ok "svMultiPhysics already built: $SVMP_BIN_DIR/svmultiphysics"
else
    cd "$BUILD_DIR"
    if [ ! -d svMultiPhysics ]; then
        log "Cloning svMultiPhysics (official)..."
        git clone --quiet https://github.com/SimVascular/svMultiPhysics.git
    fi
    cd svMultiPhysics

    log "Checking out base commit ($SVMP_BASE_COMMIT)..."
    git fetch --quiet
    git checkout --quiet "$SVMP_BASE_COMMIT"

    # Configure git for git am
    git config user.email "install@local" || true
    git config user.name "MB-CFD installer" || true

    log "Applying patches..."
    PATCHES_DIR="$SCRIPT_DIR/patches/svMP"
    if [ ! -d "$PATCHES_DIR" ]; then
        err "Patches directory not found: $PATCHES_DIR"
    fi

    # Reset in case a previous am failed
    git am --abort 2>/dev/null || true

    for patch in "$PATCHES_DIR"/*.patch; do
        log "  -> $(basename $patch)"
        if ! git am --quiet "$patch"; then
            err "Patch failed: $(basename $patch). Make sure you are on the correct base commit."
        fi
    done
    ok "Patches applied"

    # PETSc/GAMG (required for the moving-boundary mesh equation): svMP links -lpetsc,
    # apt provides libpetsc_real.so -> build a local symlink dir that svMP expects.
    PETSC_LINK="$HOME/petsc_svmp"
    mkdir -p "$PETSC_LINK/lib"
    if [ -d /usr/lib/petsc/include ]; then
        ln -sf /usr/lib/petsc/include "$PETSC_LINK/include"
    else
        ln -sf "$(dirname "$(ls /usr/lib/petscdir/*/x86_64-linux-gnu-real/include/petsc.h 2>/dev/null | head -1)")" "$PETSC_LINK/include" 2>/dev/null || true
    fi
    PETSC_SO=$(ls /usr/lib/x86_64-linux-gnu/libpetsc_real.so 2>/dev/null | head -1)
    [ -z "$PETSC_SO" ] && PETSC_SO=$(ls /usr/lib/x86_64-linux-gnu/libpetsc_real.so.* 2>/dev/null | head -1)
    if [ -n "$PETSC_SO" ]; then
        ln -sf "$PETSC_SO" "$PETSC_LINK/lib/libpetsc.so"
        ok "PETSc trouvé -> $PETSC_LINK (build svMP avec GAMG)"
        SV_PETSC_FLAG="-DSV_PETSC_DIR=$PETSC_LINK"
    else
        warn "PETSc introuvable (libpetsc-real-dev) -> svMP sera construit SANS PETSc (FB ok, MB non)"
        SV_PETSC_FLAG=""
    fi

    log "CMake svMultiPhysics (MPI + PETSc/GAMG)..."
    mkdir -p build
    cd build
    cmake \
        -DSV_USE_MPI=ON \
        $SV_PETSC_FLAG \
        -DCMAKE_BUILD_TYPE=Release \
        .. > /dev/null 2>&1

    log "Building svMultiPhysics (may take 15-30 min on $CORES cores)..."
    make -j"$CORES" svmultiphysics

    # Locate the binary (path varies by version)
    SVMP_BUILT=$(find . -name svmultiphysics -executable -type f | head -1)
    [ -n "$SVMP_BUILT" ] || err "svmultiphysics binary not found after build"

    # Copy to a stable location
    mkdir -p "$SVMP_BIN_DIR"
    cp "$SVMP_BUILT" "$SVMP_BIN_DIR/svmultiphysics"
    chmod +x "$SVMP_BIN_DIR/svmultiphysics"

    ok "svMultiPhysics built: $SVMP_BIN_DIR/svmultiphysics"
fi

# ------------------------------------------------------------------------------
# Step 6: Python dependencies
# ------------------------------------------------------------------------------
section "Step 6/7: Python dependencies"

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    log "Installing Python packages..."
    pip3 install --user --break-system-packages -q -r "$SCRIPT_DIR/requirements.txt" \
        || err "pip install failed"
    ok "Python packages installed"
else
    warn "requirements.txt not found, skipping pip install"
fi

# ------------------------------------------------------------------------------
# Step 7: Environment configuration
# ------------------------------------------------------------------------------
section "Step 7/7: Environment configuration"

# LD_LIBRARY_PATH (for libElas.so and libCommons.so)
LD_LINE="export LD_LIBRARY_PATH=\$HOME/lib:\$LD_LIBRARY_PATH"
if ! grep -q "$HOME/lib:" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# MB-CFD pipeline: libElas.so and libCommons.so in \$HOME/lib" >> ~/.bashrc
    echo "$LD_LINE" >> ~/.bashrc
    ok "LD_LIBRARY_PATH added to ~/.bashrc"
else
    ok "LD_LIBRARY_PATH already configured in ~/.bashrc"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
section "Installation complete"

echo "" | tee -a "$LOG_FILE"
echo "Installed binaries:" | tee -a "$LOG_FILE"
echo "  - svMultiPhysics : $SVMP_BIN_DIR/svmultiphysics" | tee -a "$LOG_FILE"
echo "  - MMG3D          : /usr/local/bin/mmg3d_O3" | tee -a "$LOG_FILE"
echo "  - Commons        : $INSTALL_PREFIX/lib/libCommons.so" | tee -a "$LOG_FILE"
echo "  - Elas           : $INSTALL_PREFIX/lib/libElas.so" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "  1. Load the environment   : source ~/.bashrc" | tee -a "$LOG_FILE"
echo "  2. Verify the install     : bash check_install.sh" | tee -a "$LOG_FILE"
echo "  3. PETSc/GAMG (for MB)     : sudo apt install -y libpetsc-real-dev" | tee -a "$LOG_FILE"
echo "  4. Run a patient          : cp patients/TEMPLATE.env patients/P0001.env && bash run_patient.sh patients/P0001.env" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
ok "Full log: $LOG_FILE"
