cd ~/ANN_2025_Final/open_groundingdino

cat > run_4gpus.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-config/cfg_odvg.py}
DATA=${2:-config/datasets_mixed_odvg.json}
OUT=${3:-outputs}
WEIGHT=${4:-weights/groundingdino_swint_ogc.pth}

# ====== Global knobs ======
export TOKENIZERS_PARALLELISM=false
export JT_SYNC=${JT_SYNC:-0}

# Force system CUDA 12.8 (avoid conda nvcc 13.x causing missing libcudart)
export CUDA_HOME=/usr/local/cuda-12.8
export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

# If you previously used these to bypass torch backends, clear them.
unset USE_TORCH TRANSFORMERS_NO_TORCH USE_TF USE_FLAX

# Use the exact python you are running now (keeps conda env consistent under mpirun)
PYTHON_BIN="$(command -v python)"
export PYTHON_BIN

echo "[ENV] python=${PYTHON_BIN}"
echo "[ENV] nvcc=$(command -v nvcc || true)"
nvcc --version | head -n 3 || true

# ====== Warmup (single rank) ======
# Goal: serialize first-time jittor/jtorch cache generation to avoid multi-rank cache corruption.
echo "[WARMUP] single-rank warmup to precompile jittor/jtorch caches..."
CUDA_VISIBLE_DEVICES=0 JT_SYNC=1 \
"${PYTHON_BIN}" -c "import jittor as jt; import transformers; x=jt.array([1.0]); y=x*x+1; y.sync(); print('warmup ok:', float(y[0]))"

# ====== Distributed launch ======
NUM_WORKERS_PER_RANK=${NUM_WORKERS_PER_RANK:-14}   # 56 cores -> 14 per rank
BATCH_PER_RANK=${BATCH_PER_RANK:-8}                # per-rank batch; global = batch * 4
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}

export MASTER_ADDR MASTER_PORT

READY_FILE="/tmp/jt_ready_${USER}_$$"
rm -f "${READY_FILE}"

echo "[DIST] launching 4 ranks, num_workers/rank=${NUM_WORKERS_PER_RANK}, batch/rank=${BATCH_PER_RANK}"
echo "[DIST] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[DIST] READY_FILE=${READY_FILE}"

mpirun --tag-output -np 4 \
  --bind-to core --map-by ppr:4:node:PE=14 \
  -x TOKENIZERS_PARALLELISM \
  -x JT_SYNC \
  -x CUDA_HOME \
  -x nvcc_path \
  -x LD_LIBRARY_PATH \
  -x PYTHON_BIN \
  -x MASTER_ADDR \
  -x MASTER_PORT \
  -x READY_FILE \
  bash -lc "
    set -euo pipefail
    cd ~/ANN_2025_Final/open_groundingdino

    export LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK}
    export RANK=\${OMPI_COMM_WORLD_RANK}
    export WORLD_SIZE=\${OMPI_COMM_WORLD_SIZE}

    # bind one GPU per rank
    export CUDA_VISIBLE_DEVICES=\${LOCAL_RANK}

    # keep conda python at front, but ensure CUDA 12.8 nvcc wins
    export PATH=/usr/local/cuda-12.8/bin:\$(dirname \"\$PYTHON_BIN\"):\$PATH
    export CUDA_HOME=/usr/local/cuda-12.8
    export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\${LD_LIBRARY_PATH:-}

    echo \"[RANK \$RANK] LOCAL_RANK=\$LOCAL_RANK CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES python=\$PYTHON_BIN nvcc=\$(command -v nvcc)\"

    # Gate: let rank0 do a quick import first, then others proceed (extra safety)
    if [ \"\$RANK\" = \"0\" ]; then
      \"\$PYTHON_BIN\" -c \"import jittor, transformers; print('rank0 import ok')\"
      touch \"\$READY_FILE\"
    else
      for i in \$(seq 1 600); do
        [ -f \"\$READY_FILE\" ] && break
        sleep 1
      done
      [ -f \"\$READY_FILE\" ] || { echo \"[RANK \$RANK] wait READY timeout\"; exit 1; }
    fi

    EXTRA_OPTIONS=\"--num_workers ${NUM_WORKERS_PER_RANK} --options batch_size=${BATCH_PER_RANK}\" \
    bash train_dist.sh ${CFG} ${DATA} ${OUT} ${WEIGHT}
  "

echo "[DONE] mpirun finished."
BASH

chmod +x run_4gpus.sh
