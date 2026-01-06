#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# User knobs
# ---------------------------
CFG=${1:-config/cfg_odvg.py}
DATA=${2:-config/datasets_mixed_odvg.json}
OUT=${3:-outputs}
WEIGHT=${4:-weights/groundingdino_swint_ogc.pth}

NUM_RANKS=${NUM_RANKS:-4}
PE_PER_RANK=${PE_PER_RANK:-14}                 # CPU cores per rank
NUM_WORKERS_PER_RANK=${NUM_WORKERS_PER_RANK:-14}
BATCH_PER_RANK=${BATCH_PER_RANK:-8}

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

# ---------------------------
# Hardening: CUDA / toolchain
# ---------------------------
export TOKENIZERS_PARALLELISM=false
export JT_SYNC=${JT_SYNC:-0}

CUDA_ROOT=/usr/local/cuda-12.8
export CUDA_HOME="$CUDA_ROOT"
export nvcc_path="$CUDA_ROOT/bin/nvcc"
export mpicc_path=/usr/bin/mpicc

# IMPORTANT:
# For Jittor MPI multi-GPU, keep a *set* of visible GPUs globally and let mpirun -np N map ranks onto it.
# Do NOT shrink to 1 GPU per rank (that can lead to mpi_local_rank > device_count warnings).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Remove torch backends flags if previously used
unset USE_TORCH TRANSFORMERS_NO_TORCH USE_TF USE_FLAX

# Use current conda python
PYTHON_BIN="$(command -v python)"
export PYTHON_BIN

# Optional: keep CPU thread behavior sane under core binding
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# Make CUDA 12.8 nvcc win
export PATH="$CUDA_ROOT/bin:$PATH"

# Sanitize LD_LIBRARY_PATH: prepend CUDA 12.8, and drop other /usr/local/cuda-* entries if any
ORIG_LD="${LD_LIBRARY_PATH:-}"
SANITIZED_LD="$(printf "%s" "$ORIG_LD" | tr ':' '\n' | grep -vE '^/usr/local/cuda-[0-9]+' | paste -sd ':' - 2>/dev/null || true)"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${SANITIZED_LD:+:$SANITIZED_LD}"

export MASTER_ADDR MASTER_PORT

# ---------------------------
# Locate repo dir (assume this script sits in open_groundingdino)
# ---------------------------
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

echo "[ENV] workdir=$WORKDIR"
echo "[ENV] python=$PYTHON_BIN"
echo "[ENV] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[ENV] nvcc=$(command -v nvcc || true)"
nvcc --version | head -n 3 || true
echo "[ENV] mpicc=$(command -v mpicc || true)"

# ---------------------------
# WARMUP0: single-rank JIT warmup
# ---------------------------
echo "[WARMUP0] single-rank warmup (compile core ops, transformers import)..."
CUDA_VISIBLE_DEVICES=0 JT_SYNC=1 \
"$PYTHON_BIN" - <<'PY'
import jittor as jt
import transformers
x = jt.array([1.0])
y = x*x + 1
y.sync()
print("warmup0 ok:", float(y[0]))
PY

# ---------------------------
# WARMUP1: MPI-mode warmup (compile mpi/nccl extern; verify all-reduce)
#   - DO NOT import jittor.mpi (your env may not ship that submodule)
#   - Use jt.in_mpi / Var.mpi_all_reduce instead
# ---------------------------
echo "[WARMUP1] mpi warmup (all ranks import jittor + mpi_all_reduce)..."
mpirun --tag-output -np "$NUM_RANKS" \
  --bind-to core --map-by "ppr:${NUM_RANKS}:node:PE=${PE_PER_RANK}" \
  -x TOKENIZERS_PARALLELISM \
  -x JT_SYNC \
  -x CUDA_HOME \
  -x nvcc_path \
  -x mpicc_path \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x PYTHON_BIN \
  -x MASTER_ADDR \
  -x MASTER_PORT \
  bash -lc "
    set -euo pipefail
    cd \"$WORKDIR\"
    \"$PYTHON_BIN\" - <<'PY'
import os
import jittor as jt

print(f\"rank={jt.rank} world={jt.world_size} in_mpi={jt.in_mpi} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\")
x = jt.ones((1,), dtype='float32')
# This will exercise mpi/nccl path in a synchronized way across ranks
if jt.in_mpi:
    x = x.mpi_all_reduce('add')
x.sync()
print(f\"mpi warmup ok: rank={jt.rank} x={float(x[0])}\")
PY
  "

# ---------------------------
# MAIN: distributed train/eval
# ---------------------------
echo "[DIST] launching train_dist.sh with ${NUM_RANKS} ranks"
echo "[DIST] num_workers/rank=${NUM_WORKERS_PER_RANK}, batch/rank=${BATCH_PER_RANK}"
echo "[DIST] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

mpirun --tag-output -np "$NUM_RANKS" \
  --bind-to core --map-by "ppr:${NUM_RANKS}:node:PE=${PE_PER_RANK}" \
  -x TOKENIZERS_PARALLELISM \
  -x JT_SYNC \
  -x CUDA_HOME \
  -x nvcc_path \
  -x mpicc_path \
  -x PATH \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x PYTHON_BIN \
  -x MASTER_ADDR \
  -x MASTER_PORT \
  bash -lc "
    set -euo pipefail
    cd \"$WORKDIR\"

    export LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK}
    export RANK=\${OMPI_COMM_WORLD_RANK}
    export WORLD_SIZE=\${OMPI_COMM_WORLD_SIZE}

    echo \"[RANK \$RANK] LOCAL_RANK=\$LOCAL_RANK WORLD_SIZE=\$WORLD_SIZE CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python=\$PYTHON_BIN nvcc=\$(command -v nvcc) mpicc=\$(command -v mpicc)\"

    EXTRA_OPTIONS=\"--num_workers ${NUM_WORKERS_PER_RANK} --options batch_size=${BATCH_PER_RANK}\" \
    bash train_dist.sh ${CFG} ${DATA} ${OUT} ${WEIGHT}
  "

echo "[DONE] mpirun finished."