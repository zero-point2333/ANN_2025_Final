CFG="config/cfg_odvg.py"
DATASETS="config/datasets_mixed_odvg.json"
OUTPUT_DIR="outputs"
PRETRAIN_MODEL_PATH="weights/groundingdino_swint_ogc.pth"
export TOKENIZERS_PARALLELISM="false"
export GROUNDINGDINO_DEBUG_NAN=1

python main.py --config_file ${CFG} --datasets ${DATASETS} --output_dir ${OUTPUT_DIR} --pretrain_model_path ${PRETRAIN_MODEL_PATH}
