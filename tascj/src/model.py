import torch
from transformers import AutoTokenizer


from modules.models.modeling_qwen3 import Qwen3ForSequenceClassification
# from modules.models.modeling_qwen2 import Qwen2ForSequenceClassification
# from modules.models.modeling_glm4 import GLM4ForSequenceClassification


class ModelManager:
    MODEL_CLASS_MAP = {
        "qwen3": Qwen3ForSequenceClassification,
    }

    @staticmethod
    def load_model(cfg):
        """
        根据 ExperimentConfig 加载模型和分词器
        """
        llm_config = cfg.llm_config

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(llm_config.dtype, torch.bfloat16)

        model_type = getattr(llm_config, "model_type", "qwen3").lower()
        model_class = ModelManager.MODEL_CLASS_MAP.get(model_type)

        if model_class is None:
            raise ValueError(
                f"不支持的模型类型: {model_type}。请检查 MODEL_CLASS_MAP 配置。"
            )

        print(
            f"正在加载模型: {llm_config.backbone} (Type: {model_type}, Dtype: {llm_config.dtype})..."
        )

        model = model_class.from_pretrained(
            llm_config.backbone,
            num_labels=llm_config.num_labels,
            torch_dtype=torch_dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=False,
        )

        # 4. 开启梯度检查点
        if llm_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(
            llm_config.backbone, local_files_only=False
        )

        if hasattr(tokenizer, "deprecation_warnings"):
            tokenizer.deprecation_warnings[
                "sequence-length-is-longer-than-the-specified-maximum"
            ] = True

        return model, tokenizer
