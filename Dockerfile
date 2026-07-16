# ============================================================================
#  Dockerfile — image reproductible du pipeline MB-CFD (solveur patché + outils)
#  Tout ce qu'il faut pour LANCER les simulations sur n'importe quel PC.
#
#  Build :   docker build -t mbcfd .
#  Run   :   docker run --rm -it -v /chemin/donnees:/data mbcfd
#            (les données patient ne sont PAS dans l'image — montées via -v)
#  Voir docs/PORTABLE_SETUP.md.
# ============================================================================
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ARG CORES=4
ARG SVMP_COMMIT=97ef51223e5a079bdee018bd9c0490c861c07fe0

# --- 1. dépendances système (= install.sh) + PETSc (apt) + gfortran ---------
RUN apt-get update -qq && apt-get install -y -q \
      build-essential cmake git wget ca-certificates pkg-config gfortran \
      python3 python3-pip python3-venv python3-dev \
      openmpi-bin libopenmpi-dev libopenblas0 libopenblas-dev \
      libvtk9-dev libxml2-dev libxslt-dev \
      libpetsc-real-dev \
 && rm -rf /var/lib/apt/lists/*

ENV INSTALL_PREFIX=/usr/local

# --- 2. PETSc : dossier-symlink attendu par svMP (cf. docs/REPRODUCE.md) -----
#   apt fournit libpetsc_real.so ; svMP linke -lpetsc -> on crée les symlinks.
RUN D=/usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-real \
 && mkdir -p /opt/petsc_svmp/lib \
 && ln -s $D/include /opt/petsc_svmp/include \
 && ln -s $D/lib/libpetsc_real.so /opt/petsc_svmp/lib/libpetsc.so

# --- 3. ISCDtoolbox Commons + LinearElasticity + MMG (USE_ELAS) -------------
RUN git clone --quiet https://github.com/ISCDtoolbox/Commons.git /opt/Commons \
 && cd /opt/Commons && mkdir build && cd build \
 && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX .. >/dev/null \
 && make -j$CORES >/dev/null && make install >/dev/null
RUN git clone --quiet https://github.com/ISCDtoolbox/LinearElasticity.git /opt/Elas \
 && cd /opt/Elas && mkdir build && cd build \
 && cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DCOMMONS_DIR=$INSTALL_PREFIX \
      -DCOMMONS_INCLUDE_DIR=$INSTALL_PREFIX/include \
      -DCOMMONS_LIBRARY=$INSTALL_PREFIX/lib/libCommons.so \
      -DCMAKE_C_FLAGS="-I$INSTALL_PREFIX/include" .. >/dev/null \
 && make -j$CORES >/dev/null && make install >/dev/null
RUN git clone --quiet https://github.com/MmgTools/mmg.git /opt/mmg \
 && cd /opt/mmg && mkdir build && cd build \
 && cmake -DUSE_ELAS=ON -DELAS_DIR=/opt/Elas/build -DCMAKE_BUILD_TYPE=Release .. >/dev/null \
 && make -j$CORES >/dev/null && make install >/dev/null

# --- 4. svMultiPhysics @ commit figé + nos patches + PETSc ------------------
COPY patches/svMP/*.patch /opt/patches/
RUN git clone --quiet https://github.com/SimVascular/svMultiPhysics.git /opt/svMP \
 && cd /opt/svMP && git checkout --quiet $SVMP_COMMIT \
 && git config user.email build@local && git config user.name build \
 && for p in $(ls /opt/patches/*.patch | sort); do echo "git am $p"; git am "$p" || { git am --abort; exit 1; }; done \
 && mkdir build && cd build \
 && cmake -DSV_USE_MPI=ON -DSV_PETSC_DIR=/opt/petsc_svmp -DCMAKE_BUILD_TYPE=Release .. \
 && make -j$CORES svmultiphysics \
 && cp "$(find . -type f -name svmultiphysics | head -1)" /usr/local/bin/svmultiphysics

# --- 5. dépendances Python + le pipeline ------------------------------------
COPY requirements.txt /opt/repo/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/repo/requirements.txt
COPY . /opt/repo

ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/petsc_svmp/lib:${LD_LIBRARY_PATH}
ENV SVMP_BIN=/usr/local/bin/svmultiphysics
WORKDIR /opt/repo
# GAMG (éq. maillage MB) — voir run_MB_aniso.sh ; toujours exporter ceci pour un run MB :
ENV PETSC_OPTIONS="-pc_type gamg -pc_gamg_type agg"
CMD ["/bin/bash"]
