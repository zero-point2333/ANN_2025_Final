#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-config/cfg_odvg.py}
DATA=${2:-config/datasets_mixed_odvg.json}
OUT=${3:-outputs}
WEIGHT=${4:-weights/groundingdino_swint_ogc.pth}

export TOKENIZERS_PARALLELISM=false
export JT_SYNC=${JT_SYNC:-0}

# Pin CUDA 12.8 (avoid conda nvcc 13.x)
export CUDA_HOME=/usr/local/cuda-12.8
export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

unset USE_TORCH TRANSFORMERS_NO_TORCH USE_TF USE_FLAX

PYTHON_BIN="$(command -v python)"
echo "[ENV] python=${PYTHON_BIN}"
echo "[ENV] nvcc=$(command -v nvcc || true)"
nvcc --version | head -n 3 || true

echo "[WARMUP] ..."
CUDA_VISIBLE_DEVICES=0 JT_SYNC=1 \
"${PYTHON_BIN}" -c "import jittor as jt; import transformers; x=jt.array([1.0]); y=x*x+1; y.sync(); print('warmup ok', float(y[0]))"
NUM_WORKERS_PER_RANK=${NUM_WORKERS_PER_RANK:-4}
BATCH_PER_RANK=${BATCH_PER_RANK:-2}
mpirun --tag-output -np 1 \
  --bind-to core --map-by ppr:1:node:PE=14 \
  -x TOKENIZERS_PARALLELISM \
  -x JT_SYNC \
  -x CUDA_HOME \
  -x nvcc_path \
  -x LD_LIBRARY_PATH \
  bash -lc "
    set -euo pipefail
    cd ~/ANN_2025_Final/open_groundingdino
    export LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK}
    export CUDA_VISIBLE_DEVICES=\${LOCAL_RANK}
    export PATH=/usr/local/cuda-12.8/bin:\$PATH
    export CUDA_HOME=/usr/local/cuda-12.8
    export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:-}

    echo \"[RANK \${OMPI_COMM_WORLD_RANK}] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES nvcc=\$(command -v nvcc)\"

    EXTRA_OPTIONS=\"--num_workers ${NUM_WORKERS_PER_RANK} --options batch_size=${BATCH_PER_RANK}\" \
    bash train_dist.sh ${CFG} ${DATA} ${OUT} ${WEIGHT}
  "
