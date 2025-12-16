import os

import jittor as jt
from jittor import nn
from transformers import AutoTokenizer, AutoConfig

# 定义 ModelScope 模型映射表（Hugging Face ID -> ModelScope ID）
MODELSCOPE_MAPPING = {
    "bert-base-uncased": "google-bert/bert-base-uncased",  # ModelScope 官方镜像
    "roberta-base": "roberta-base"  # 根据实际可用模型调整
}

def get_tokenlizer(text_encoder_type):
    """
    保持原有接口：输入 text_encoder_type，返回 HF 的 tokenizer。
    """
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
    
    # 新增：下载模型到本地缓存（如果尚未下载）
    local_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", model_id)
    if not os.path.exists(local_dir):
        raise EnvironmentError("local model not found!")
    
    # 修改：从本地路径加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    return tokenizer


class JittorTextEncoder(nn.Module):
    """
    一个极简 Jittor 文本编码器，接口尽量向 HF 的 BertModel 看齐：

        encoder = JittorTextEncoder.from_pretrained("bert-base-uncased")
        outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_state = outputs["last_hidden_state"]

    这里不加载 HuggingFace 的权重，
    只是利用 AutoConfig 获取 vocab_size / hidden_size 等配置，
    然后随机初始化一个 Jittor 模型。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        vocab_size = config.vocab_size
        hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", 768))
        max_position_embeddings = getattr(config, "max_position_embeddings", 512)
        dropout = getattr(config, "hidden_dropout_prob", 0.1)

        self.hidden_size = hidden_size
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 这里不实现完整 Transformer，只做若干层简单前馈
        num_layers = getattr(config, "num_hidden_layers", 2)
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.layers = nn.ModuleList(layers)

    @classmethod
    def from_pretrained(cls, text_encoder_type):
        # 新增：使用 ModelScope 映射
        model_id = MODELSCOPE_MAPPING.get(text_encoder_type, text_encoder_type)
        local_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", model_id)
        if not os.path.exists(local_dir):
            raise EnvironmentError("local model not found!")
        
        # 修改：从本地路径加载config
        cfg = AutoConfig.from_pretrained(local_dir)
        return cls(cfg)

    def execute(self, input_ids=None, attention_mask=None, **kwargs):
        """
        输入：
            input_ids: [B, L]
            attention_mask: [B, L]，1 表示有效，0 表示 padding

        输出：
            {"last_hidden_state": [B, L, H]}
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        if not isinstance(input_ids, jt.Var):
            input_ids = jt.array(input_ids, dtype="int32")

        batch_size, seq_len = input_ids.shape

        # 位置编码 [B, L]
        position_ids = jt.arange(seq_len, dtype="int32").unsqueeze(0).repeat(batch_size, 1)

        # [B, L, H]
        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            residual = x
            out = layer(x)
            x = residual + out

        if attention_mask is not None:
            if not isinstance(attention_mask, jt.Var):
                attention_mask = jt.array(attention_mask, dtype="float32")
            # [B, L, 1]
            mask = attention_mask.unsqueeze(-1)
            x = x * mask

        return {"last_hidden_state": x}


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased" or (
        isinstance(text_encoder_type, str)
        and os.path.isdir(text_encoder_type)
        and os.path.exists(text_encoder_type)
    ):
        return JittorTextEncoder.from_pretrained(text_encoder_type)

    if text_encoder_type == "roberta-base":
        return JittorTextEncoder.from_pretrained(text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
