# ruff: noqa: E402
import os
import sys
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ==========================================
# Path Setup & Custom Imports
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "tascj" / "src"))

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import ExperimentConfig
from model import ModelManager
from dataloader import DataLoaderManager
from modules.logging import get_logger
from modules.utils import to_gpu

# 设置 PyTorch
torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Collect activation scales for SmoothQuant")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--load-from", default=None, type=str, help="Checkpoint path")
    parser.add_argument("--output-root", default="../artifacts", help="Root for outputs")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-log-file", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def do_test(cfg, model, tokenizer, logger):
    """
    执行推理以触发 Hook 收集统计信息
    """
    logger.info("Evaluation start for scale collection")

    # 使用 validation 模式加载数据
    val_loader = DataLoaderManager.get_dataloader(cfg, tokenizer, mode="val")

    model.eval()
    
    prog_bar = tqdm(val_loader, desc="Collecting Scales")
    probs = []
    
    for batch in prog_bar:
        batch = to_gpu(batch)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 适配新的模型 forward 签名
            logits = model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                suffix_ids=batch["suffix_ids"],
                doc_ids=batch["doc_ids"],
                last_tokens=batch["last_tokens"],
            )

        # 简单的 Loss 计算用于显示进度
        logits = logits.float().flatten()
        losses = []
        for _logits, _label in zip(
            logits.split(batch["num_candidates"]), batch["label"]
        ):
            losses.append(F.cross_entropy(_logits, _label))
            probs.append(_logits.float().softmax(dim=-1).data.cpu())
        
        loss = torch.stack(losses).mean()
        prog_bar.set_description(f"Loss: {loss.item():.4f}")

    result = [prob.numpy() for prob in probs]

    logger.info("Collection prediction done")
    
    if hasattr(val_loader.dataset, "evaluate"):
        eval_result = val_loader.dataset.evaluate(result)
    else:
        eval_result = {"info": "Dataset evaluate not implemented"}
        
    logger.info("Evaluation end")
    return result, eval_result


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
    
    # 将 work_dir 设置为 artifacts/exp_name (与 train.py 保持一致)
    cfg.work_dir = str(work_dir_root / cfg.exp_name)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    # 4. 初始化日志
    logger = get_logger("MAP", log_file=None if args.no_log_file else Path(cfg.work_dir) / "sq_collect.log")
    logger.info(f"Loaded config from: {config_path}")
    logger.info(f"Work Dir: {cfg.work_dir}")

    return cfg, logger


def main():
    args = parse_args()
    cfg, logger = setup(args)

    logger.info("ℹ️  Forcing test_fold = 0 for scale collection (mimicking original logic)")
    cfg.test_fold = 0

    if args.load_from:
        logger.info(f"Overriding model path with: {args.load_from}")
        cfg.llm_config.backbone = args.load_from
        cfg.llm_config.resume_from_checkpoint = args.load_from

    logger.info("Loading model and tokenizer...")
    model, tokenizer = ModelManager.load_model(cfg)


    logger.info("Registering hooks for activation statistics...")
    act_scales = {}

    def stat_tensor(name, tensor):
        """统计 Tensor 的最大绝对值"""
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        """Forward Hook"""
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # 使用 functools.partial 传递 layer name
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )


    result, eval_result = do_test(cfg, model, tokenizer, logger)
    logger.info(f"Collection Run Result: {eval_result}")


    for h in hooks:
        h.remove()

    save_path = Path(cfg.work_dir) / "act_scales.pth"
    torch.save(act_scales, save_path)
    logger.info(f"✅ Activation scales saved to: {save_path}")


if __name__ == "__main__":
    main()