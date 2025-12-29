CFG=$1
DATASETS=$2
OUTPUT_DIR=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python main.py \
        --output_dir ${OUTPUT_DIR} \
        --eval \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /path/to/groundingdino_swint_ogc.pth \
        --options text_encoder_type=/path/to/bert-base-uncased
