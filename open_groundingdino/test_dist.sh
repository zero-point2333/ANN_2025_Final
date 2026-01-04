CFG="${1:-config/cfg_odvg.py}"
DATASETS="${2:-config/datasets_mixed_odvg.json}"
OUTPUT_DIR="${3:-outputs}"
PRETRAIN_MODEL_PATH="${4:-weights/groundingdino_swint_ogc.pth}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-bert-base-uncased}"
EXTRA_OPTIONS="${EXTRA_OPTIONS:-}"

export TOKENIZERS_PARALLELISM=false
python main.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path ${PRETRAIN_MODEL_PATH}
