# tascj/src/train.py

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# === 路径设置：确保能导入 src 下的模块 ===
# 获取当前脚本所在目录 (tascj/src) 的上上级目录作为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tascj" / "src"))

# === 环境变量设置 (参考原代码) ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# torch.multiprocessing.set_sharing_strategy("file_system")
# torch.backends.cudnn.deterministic = True

# === 导入自定义模块 ===
import swanlab  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from dataloader import DataLoaderManager  # noqa: E402
from model import ModelManager  # noqa: E402
from modules.logging import get_logger  # noqa: E402
from modules.utils import to_gpu  # noqa: E402
from optimizer import OptimizerManager  # noqa: E402

# ==============================================================================
# 辅助函数
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAP Task with SwanLab")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts",
        help="Root directory for outputs",
    )
    parser.add_argument(
        "--load-from", type=str, default=None, help="Path to checkpoint to load"
    )
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument(
        "--no-log-file", action="store_true", help="Do not save log to file"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Force specific random seed"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Path to save evaluation results"
    )
    return parser.parse_args()


def seed_all(seed):
    """设置所有随机种子以保证可复现性"""
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_env(args, cfg):
    """设置工作目录、日志和备份配置"""
    # 1. 确定工作目录
    # 如果 output-root 是相对路径，则相对于 PROJECT_ROOT
    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    # 最终工作目录: artifacts/exp_name
    work_dir = out_root / cfg.exp_name
    cfg.work_dir = str(work_dir)  # 将路径回写到 config 中方便后续使用
    work_dir.mkdir(parents=True, exist_ok=True)

    # 2. 生成时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # 3. 备份配置文件 (仅在训练模式)
    if not args.eval_only:
        shutil.copy(args.config, work_dir / f"config_{timestamp}.yaml")

    # 4. 初始化 Logger
    log_file = work_dir / f"{timestamp}.log" if not args.no_log_file else None
    logger = get_logger("MAP", log_file=log_file)

    logger.info(f"🚀 Project Root: {PROJECT_ROOT}")
    logger.info(f"📂 Work Dir: {work_dir}")

    # 5. 设置种子
    real_seed = args.seed if args.seed >= 0 else cfg.seed
    seed_all(real_seed)
    logger.info(f"🎲 Random Seed: {real_seed}")

    return logger, timestamp


# ==============================================================================
# 核心逻辑: 训练与测试
# ==============================================================================


@torch.no_grad()
def do_test(cfg, model, tokenizer, logger):
    """执行验证/测试循环"""
    logger.info("Evaluation start...")

    val_loader = DataLoaderManager.get_dataloader(cfg, tokenizer, mode="val")
    model.eval()

    probs = []
    # 使用 tqdm 显示进度
    prog_bar = tqdm(val_loader, desc="Evaluating", leave=False)

    for batch in prog_bar:
        batch = to_gpu(batch)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 注意：这里的参数传递方式参考了原始 train.py，使用了解构参数而非 **batch
            # 确保你的 Model forward 函数接受这些参数
            logits = model(
                batch["input_ids"],
                batch["position_ids"],
                batch["suffix_ids"],
                batch["doc_ids"],
                batch["last_tokens"],
            )

        # 处理输出：flatten 后按 num_candidates 切分
        logits = logits.float().flatten()
        batch_probs = []
        batch_losses = []

        # 针对每个样本（不同数量的 candidates）切分 logits
        for _logits, _label in zip(
            logits.split(batch["num_candidates"]), batch["label"]
        ):
            # 计算 loss 仅用于进度条显示
            batch_losses.append(F.cross_entropy(_logits, _label))
            # 保存概率用于后续指标计算
            batch_probs.append(_logits.float().softmax(dim=-1).data.cpu())

        loss = torch.stack(batch_losses).mean()
        prog_bar.set_description(f"Eval Loss: {loss.item():.4f}")
        probs.extend(batch_probs)

    # 转换结果格式
    result = [prob.numpy() for prob in probs]

    # 调用 Dataset 的评估方法 (计算 MAP@3 等)
    if hasattr(val_loader.dataset, "evaluate"):
        eval_result = val_loader.dataset.evaluate(result)
    else:
        eval_result = {"info": "Dataset does not support evaluation"}

    logger.info(f"Evaluation done. Metrics: {eval_result}")
    return result, eval_result


def do_train(cfg, model, tokenizer, logger):
    """执行训练循环"""

    # 1. 准备数据
    train_loader = DataLoaderManager.get_dataloader(cfg, tokenizer, mode="train")

    # 2. 准备优化器
    optimizer = OptimizerManager.get_optimizer(model, cfg)

    # 3. 准备 Scheduler
    # 原代码逻辑：total_steps = epochs * len(loader)
    total_steps = cfg.max_epochs * len(train_loader)
    # 如果使用了梯度累积，step 数会变少，但在 reference 代码中似乎并未除以 accumulation_steps
    # 我们这里保持与 reference 一致，基于 iter 数量
    lr_scheduler = OptimizerManager.get_scheduler(optimizer, cfg, total_steps)

    # 确定用于记录日志的参数组 ID (通常取第一个或者学习率最大的那个)
    best_param_group_id = 0

    logger.info("Training start...")
    total_updates = 0
    max_epochs = cfg.max_epochs

    for curr_epoch in range(max_epochs):
        model.train()

        # 创建进度条
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {curr_epoch + 1}/{max_epochs}")

        for curr_iter, batch in enumerate(epoch_iterator):
            batch = to_gpu(batch)

            # --- Forward ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch["input_ids"],
                    batch["position_ids"],
                    batch["suffix_ids"],
                    batch["doc_ids"],
                    batch["last_tokens"],
                )

            # --- Loss Calculation ---
            logits = logits.float().flatten()
            losses = []
            # 参考原代码：根据 num_candidates 切分 logits 和 labels
            for _logits, _label in zip(
                logits.split(batch["num_candidates"]), batch["label"]
            ):
                losses.append(F.cross_entropy(_logits, _label))

            loss = torch.stack(losses).mean()

            # --- Backward ---
            # 特殊处理：OffloadAdam 需要这个标志 (参考原代码)
            if hasattr(optimizer, "ready_for_optimizer_step"):
                optimizer.ready_for_optimizer_step = True

            loss.backward()

            # --- Optimizer Step ---
            # 如果有梯度累积，需要在此处添加逻辑。参考代码中 accumulation 似乎为 1 或未显式处理累积
            # 这里按照原代码逻辑：每个 iter 都 step
            if cfg.gradient_accumulation_steps > 1:
                # 简单的累积逻辑补充（如果 config 设置了）
                if (curr_iter + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    total_updates += 1
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                total_updates += 1

            # --- Logging ---
            if total_updates % cfg.log_interval == 0:
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                loss_val = loss.item()
                # 显存监控
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                # 更新进度条描述
                epoch_iterator.set_postfix(
                    loss=f"{loss_val:.4f}", lr=f"{lr:.2e}", mem=f"{max_mem_mb:.0f}M"
                )

                logger.info(
                    f"Epoch [{curr_epoch + 1}/{max_epochs}] Iter [{curr_iter + 1}/{len(train_loader)}] "
                    f"lr: {lr:.4e}, loss: {loss_val:.4f}, max_mem: {max_mem_mb:.0f}M"
                )

                # SwanLab Log
                swanlab.log(
                    {
                        "train/loss": loss_val,
                        "train/lr": lr,
                        "train/memory_mb": max_mem_mb,
                        "train/global_step": total_updates,
                        "train/epoch": curr_epoch + 1,
                    }
                )

        # === End of Epoch ===

        # 1. 保存 Checkpoint
        ckpt_dir = Path(cfg.work_dir) / f"checkpoint_epoch_{curr_epoch + 1}"
        logger.info(f"Saving checkpoint to: {ckpt_dir}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        # 2. 验证 (Evaluation)
        if (curr_epoch + 1) % cfg.eval_interval == 0:
            result, eval_result = do_test(cfg, model, tokenizer, logger)

            # SwanLab Log Metrics
            # 将 eval_result 中的数值项记录到 SwanLab
            swan_metrics = {
                f"val/{k}": v
                for k, v in eval_result.items()
                if isinstance(v, (int, float))
            }
            swan_metrics["val/epoch"] = curr_epoch + 1
            swanlab.log(swan_metrics)

            # 保存预测结果
            res_path = Path(cfg.work_dir) / f"result_epoch_{curr_epoch + 1}.pth"
            torch.save(result, res_path)
            logger.info(f"Saved evaluation results to {res_path}")


# ==============================================================================
# Main
# ==============================================================================


def main():
    args = parse_args()

    # 1. 解析配置文件路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    print(f"✅ Loading config from: {config_path}")
    cfg = ExperimentConfig.from_yaml(str(config_path))

    # 2. 环境与日志初始化
    logger, timestamp = setup_env(args, cfg)

    # 3. SwanLab 初始化 (仅在非评估模式下)
    if not args.eval_only:
        swanlab_dir = Path(cfg.work_dir) / "swanlab"
        logger.info(f"Initializing SwanLab (logdir: {swanlab_dir})")

        swanlab.init(
            project="MAP-Math-Misconceptions",  # 你可以修改项目名
            name=f"{cfg.exp_name}_{timestamp}",
            config=cfg.model_dump(),  # 传入所有配置
            logdir=str(swanlab_dir),
            mode="disabled"
            if args.no_log_file
            else "cloud",  # 如果不想上传云端，可改为 "local"
        )

    # 4. 加载模型与分词器
    model, tokenizer = ModelManager.load_model(cfg)

    # 处理 Resume from checkpoint (优先使用 config 中的配置，其次使用命令行参数)
    load_path = (
        args.load_from if args.load_from else cfg.llm_config.resume_from_checkpoint
    )

    if load_path:
        logger.info(f"🔄 Loading pretrained weights from: {load_path}")
        # 这里假设 ModelManager.load_model 已经加载了基础结构，
        # 如果 load_path 是完整的 HF 目录，可以直接用 from_pretrained 覆盖，
        # 或者加载 state_dict。鉴于 Qwen/GLM 代码，这里简单地假设 load_path 是模型目录
        # 为了稳健，我们修改 cfg 中的 backbone 再次调用（或者手动加载权重）
        # 简单起见，如果提供了 load_path，我们重新加载一次模型
        cfg.llm_config.backbone = load_path
        # 清理旧模型显存（可选）
        del model
        torch.cuda.empty_cache()
        model, tokenizer = ModelManager.load_model(cfg)

    # 处理 BF16 转换 (参考原代码)
    if cfg.cast_to_bf16:
        logger.info("🔧 Casting model parameters to BF16 manually.")
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)

    # 5. 开始任务
    if args.eval_only:
        result, eval_result = do_test(cfg, model, tokenizer, logger)
        if args.out:
            torch.save(result, args.out)
            logger.info(f"Saved specific output to {args.out}")
    else:
        do_train(cfg, model, tokenizer, logger)

    logger.info("✨ All finished.")


if __name__ == "__main__":
    main()
