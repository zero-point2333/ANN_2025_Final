CUDA_VISIBLE_DEVICES=0
CFG="groundingdino/config/GroundingDINO_SwinT_OGC.py"
PRETRAIN_MODEL_PATH="weights/groundingdino_swint_ogc.pth"
ANNO_PATH="coco/annotations/instances_val2017.json"
IMAGE_DIR="coco/images"

python demo/test_ap_on_coco.py -c ${CFG} -p ${PRETRAIN_MODEL_PATH} --anno_path ${ANNO_PATH} --image_dir ${IMAGE_DIR}
