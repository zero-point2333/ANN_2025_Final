CFG="${1:-config/cfg_odvg.py}"
DATASETS="${2:-config/datasets_mixed_odvg.json}"
OUTPUT_DIR="${3:-outputs}"
PRETRAIN_MODEL_PATH="${4:-weights/groundingdino_swint_ogc.pth}"
EXTRA_OPTIONS="${EXTRA_OPTIONS:-}"
export TOKENIZERS_PARALLELISM="false"
export GROUNDINGDINO_DEBUG_NAN=1
export JT_SYNC=0
# export trace_py_var=3
python main.py --config_file ${CFG} --datasets ${DATASETS} --output_dir ${OUTPUT_DIR} --pretrain_model_path ${PRETRAIN_MODEL_PATH} ${EXTRA_OPTIONS}