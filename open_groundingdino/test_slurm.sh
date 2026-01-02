PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
CFG=$3
DATASETS=$4
OUTPUT_DIR=$5
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-/home/cslabuser/ANN_2025_Final/open_groundingdino/gdinot-1.8m-odvg.pth}
TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-/home/cslabuser/ANN_2025_Final/open_groundingdino/bert-base-uncased}
EXTRA_OPTIONS=${EXTRA_OPTIONS:-}

if [ "${GPUS_PER_NODE}" -gt 1 ]; then
    if [[ "${EXTRA_OPTIONS}" != *"--distributed"* ]]; then
        EXTRA_OPTIONS="${EXTRA_OPTIONS} --distributed True"
    fi
fi

srun -p ${PARTITION} \
    --job-name=open_G_dino \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u main.py --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
        --options text_encoder_type=${TEXT_ENCODER_PATH} \
        ${EXTRA_OPTIONS}
