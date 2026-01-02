CFG="${1:-config/cfg_odvg.py}"
DATASETS="${2:-config/datasets_mixed_odvg.json}"
OUTPUT_DIR="${3:-outputs}"
PRETRAIN_MODEL_PATH="${4:-weights/gdinot-1.8m-odvg.pth}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-bert-base-uncased}"
EXTRA_OPTIONS="${EXTRA_OPTIONS:-}"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python main.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path ./gdinot-1.8m-odvg.pth \
        --options text_encoder_type=./bert-base-uncased
