# ruff: noqa: E402
import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path


# 定位项目根目录 (假设当前文件在 tascj/src/inference/sq_convert.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "tascj" / "src"))

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import ExperimentConfig
from model import ModelManager
from modules.logging import get_logger


from modules.models.w8a8_kernels import per_channel_quant

# 设置 PyTorch
torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to W8A8 using SmoothQuant scales")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--alpha", type=float, default=0.75, help="SmoothQuant Alpha")
    parser.add_argument("--epoch", type=int, default=1, help="Which epoch checkpoint to convert")
    parser.add_argument("--output-root", default="../artifacts", help="Root for outputs")
    parser.add_argument("--load-from", default=None, type=str, help="Specific checkpoint path (optional)")
    return parser.parse_args()


def setup(args):
    """
    初始化配置和日志
    """
    # 1. 处理 Config 路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # 2. 加载配置
    cfg = ExperimentConfig.from_yaml(str(config_path))

    # 3. 设置工作目录
    work_dir_root = Path(args.output_root)
    if not work_dir_root.is_absolute():
        work_dir_root = PROJECT_ROOT / work_dir_root
    
    cfg.work_dir = str(work_dir_root / cfg.exp_name)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    logger = get_logger("MAP", log_file=Path(cfg.work_dir) / "sq_convert.log")
    logger.info(f"Loaded config from: {config_path}")
    
    return cfg, logger


def convert_to_w8a8(linear_layer, scale, set_input_scale=False):
    """
    原地修改 Linear 层：
    1. 使用 scale 缩放权重 (SmoothQuant 核心)
    2. 将权重量化为 int8
    3. 注册 weight_scale 和 input_scale buffer
    """
    # SmoothQuant: W' = W * diag(s)
    # scale shape: [1, in_features]
    weight = linear_layer.weight.mul_(scale.view(1, -1))
    
    # Quantize to Int8
    weight_int8, weight_scale = per_channel_quant(weight, torch.int8)
    
    # 替换权重数据
    linear_layer.weight.data = weight_int8
    
    # 注册量化参数 (用于推理时反量化)
    if hasattr(linear_layer, "weight_scale"):
         # 防止重复注册报错
        linear_layer.weight_scale = weight_scale
    else:
        linear_layer.register_buffer("weight_scale", weight_scale)
        
    if set_input_scale:
        if hasattr(linear_layer, "input_scale"):
            linear_layer.input_scale = scale
        else:
            linear_layer.register_buffer("input_scale", scale)


def get_weight_scale(linear_layers):
    """
    获取一组 Linear 层权重的最大绝对值 (Channel-wise max)
    """
    # stack weights: [N, out, in] -> abs -> max over out -> [N, in]
    weight_scales = torch.stack(
        [linear_layer.weight.abs().max(dim=0)[0] for linear_layer in linear_layers]
    )
    # max over N -> [in]
    return weight_scales.max(dim=0)[0]


@torch.no_grad()
def main():
    args = parse_args()
    cfg, logger = setup(args)
    ALPHA = args.alpha

    work_dir = Path(cfg.work_dir)
    
    # ==========================================
    # 1. 确定 Checkpoint 路径
    # ==========================================
    if args.load_from:
        checkpoint_path = args.load_from
    else:
        # 默认寻找训练脚本保存的 checkpoint_epoch_X
        checkpoint_path = work_dir / f"checkpoint_epoch_{args.epoch}"
    
    logger.info(f"Target Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ==========================================
    # 2. 加载模型 (FP16/BF16)
    # ==========================================
    # 覆盖 backbone 为训练好的 checkpoint
    cfg.llm_config.backbone = str(checkpoint_path)
    # 强制不使用 gradient checkpointing 以方便修改
    cfg.llm_config.gradient_checkpointing = False 
    
    logger.info("Loading base model for conversion...")
    model, tokenizer = ModelManager.load_model(cfg)
    model.requires_grad_(False)
    
    # ==========================================
    # 3. 加载 SmoothQuant 统计值
    # ==========================================
    scales_path = work_dir / "act_scales.pth"
    if not scales_path.exists():
        raise FileNotFoundError(f"Activation scales not found at {scales_path}. Run sq_collect.py first.")
        
    logger.info(f"Loading activation scales from {scales_path}")
    act_scales = torch.load(scales_path, map_location="cuda", weights_only=True)

    # ==========================================
    # 4. 执行 SmoothQuant & W8A8 转换
    # ==========================================
    # 这里的 model.model 是指 transformers 模型内部的 base model (例如 Qwen2Model)
    # 根据 ModelManager 封装，model 可能是 XXXForSequenceClassification
    # 通常结构是 model.model.layers (Qwen2)
    
    base_model = getattr(model, "model", model) 
    # 如果是 Llama/Qwen 等结构，通常 layers 在 base_model.layers
    layers = getattr(base_model, "layers", None)
    
    if layers is None:
        raise ValueError(f"Could not find .layers in model. Available keys: {base_model.__dict__.keys()}")

    logger.info(f"Starting conversion with Alpha={ALPHA}...")
    
    for layer_idx, layer in enumerate(layers):
        logger.info(f"Processing layer {layer_idx}")
        
        # ------------------------------------------------
        # Self Attention: Q, K, V Projections
        # ------------------------------------------------
        # SmoothQuant 公式: s = max(|X|)^alpha / max(|W|)^(1-alpha)
        # 这里的 key 需要与 sq_collect.py 中 hook 的 name 匹配
        # HuggingFace Qwen2 命名: self_attn.q_proj, etc.
        
        # 注意：sq_collect 里的 name 是全名，例如 "model.layers.0.self_attn.q_proj"
        # 我们需要确保 key 匹配。
        
        prefix = f"model.layers.{layer_idx}" 
        
        # 1. QKV Projections
        act_scale = act_scales[f"{prefix}.self_attn.q_proj"]
        
        # Qwen2 是 q_proj, k_proj, v_proj 独立的
        weight_scale = get_weight_scale(
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        )
        
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        
        # 应用转换
        convert_to_w8a8(layer.self_attn.q_proj, scale)
        convert_to_w8a8(layer.self_attn.k_proj, scale)
        convert_to_w8a8(layer.self_attn.v_proj, scale)
        
        # Input LayerNorm 补偿 (除以 scale)
        layer.input_layernorm.weight.div_(scale)

        # ------------------------------------------------
        # Self Attention: Output Projection
        # ------------------------------------------------
        act_scale = act_scales[f"{prefix}.self_attn.o_proj"]
        weight_scale = get_weight_scale([layer.self_attn.o_proj])
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        
        # o_proj 需要记录 input_scale (用于反量化上一层的输出？或者用于下一层的输入？)
        # 根据原代码逻辑，这里 set_input_scale=True
        convert_to_w8a8(layer.self_attn.o_proj, scale, set_input_scale=True)

        # ------------------------------------------------
        # MLP: Gate & Up Projections
        # ------------------------------------------------
        # 检查结构：Qwen2 使用 gate_proj 和 up_proj
        # 原代码做了兼容性检查
        
        if hasattr(layer.mlp, "gate_up_proj"):
            # GLM4 / Falcon 风格
            act_scale = act_scales[f"{prefix}.mlp.gate_up_proj"]
            gate_up_projs = [layer.mlp.gate_up_proj]
        else:
            # Llama / Qwen 风格
            # 通常 gate_proj 的激活值被视为输入的代表？
            # 注意：act_scales 的 key 必须存在。
            # 如果 sq_collect 收集了 gate_proj，就用它。
            key_check = f"{prefix}.mlp.gate_proj"
            if key_check in act_scales:
                 act_scale = act_scales[key_check]
            else:
                # 尝试 up_proj 或者报错
                act_scale = act_scales[f"{prefix}.mlp.up_proj"]

            gate_up_projs = [layer.mlp.gate_proj, layer.mlp.up_proj]

        weight_scale = get_weight_scale(gate_up_projs)
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        
        for proj in gate_up_projs:
            convert_to_w8a8(proj, scale)
            
        # Post Attention LayerNorm 补偿
        layer.post_attention_layernorm.weight.div_(scale)

        # ------------------------------------------------
        # MLP: Down Projection
        # ------------------------------------------------
        act_scale = act_scales[f"{prefix}.mlp.down_proj"]
        weight_scale = get_weight_scale([layer.mlp.down_proj])
        scale = (act_scale.pow(ALPHA) / weight_scale.pow(1 - ALPHA)).clamp(min=1e-5)
        
        convert_to_w8a8(layer.mlp.down_proj, scale, set_input_scale=True)

    # ==========================================
    # 5. 保存转换后的模型
    # ==========================================
    output_dir = work_dir / "checkpoint_w8a8"
    logger.info(f"Saving W8A8 quantized model to {output_dir} ...")
    
    # 保存权重 (state_dict 会包含 .weight (int8), .weight_scale, .input_scale)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("✅ Conversion and save completed.")


if __name__ == "__main__":
    main()