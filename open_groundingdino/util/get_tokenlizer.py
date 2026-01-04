from transformers import AutoTokenizer, BertModel, RobertaModel
import os

# 定义 ModelScope 模型映射表（Hugging Face ID -> ModelScope ID）
MODELSCOPE_MAPPING = {
    "bert-base-uncased": "google-bert/bert-base-uncased",  # ModelScope 官方镜像
    "roberta-base": "AI-ModelScope/roberta-base"  # 根据实际可用模型调整
}

def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))
    # 新增：检查是否为 ModelScope 模型ID
    model_id = MODELSCOPE_MAPPING.get(text_encoder_type, text_encoder_type)
    
    # 新增：根据环境变量/原始HOME选择缓存目录（避免 HOME 被改写后找不到模型）
    cache_root = "/modelscope"
    local_dir = os.path.join(cache_root, "hub", "models", model_id)
    if not os.path.exists(local_dir):
        orig_home = os.path.expanduser("~")
        if orig_home:
            alt_dir = os.path.join(orig_home, ".cache", "modelscope", "hub", "models", model_id)
            if os.path.exists(alt_dir):
                local_dir = alt_dir
    if not os.path.exists(local_dir):
        raise EnvironmentError(f"local model {model_id} not found!")
    
    # 修改：从本地路径加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    model_id = MODELSCOPE_MAPPING.get(text_encoder_type, text_encoder_type)
    cache_root = "/modelscope"
    local_dir = os.path.join(cache_root, "hub", "models", model_id)
    if not os.path.exists(local_dir):
        orig_home = os.path.expanduser("~")
        if orig_home:
            alt_dir = os.path.join(orig_home, ".cache", "modelscope", "hub", "models", model_id)
            if os.path.exists(alt_dir):
                local_dir = alt_dir
    if not os.path.exists(local_dir):
        raise EnvironmentError("local model not found!")
    
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return BertModel.from_pretrained(local_dir, local_files_only=True, use_safetensors=False)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(local_dir, local_files_only=True, use_safetensors=False)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
