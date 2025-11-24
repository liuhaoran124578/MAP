import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler

from modules.optim.offload_adam_gr import OffloadAdam


class OptimizerManager:
    @staticmethod
    def get_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
        """
        构建优化器
        """
        opt_config = cfg.optimizer_config
        opt_name = opt_config.name
        opt_params = opt_config.params.copy()

        # 确定优化器类
        if opt_name == "OffloadAdam":
            if OffloadAdam is None:
                raise ImportError(
                    "无法导入 OffloadAdam，请检查 modules/optim/offload_adam_gr.py"
                )
            OptimizerClass = OffloadAdam
        elif opt_name == "AdamW":
            OptimizerClass = torch.optim.AdamW
        else:
            OptimizerClass = getattr(torch.optim, opt_name, torch.optim.AdamW)

        if hasattr(OptimizerClass, "issue_h2d_transfer"):
            print(f"检测到 {opt_name} 支持 issue_h2d_transfer，传入 model 实例。")
            opt_params["model"] = model
        else:
            print(f"初始化标准优化器 {opt_name}，传入参数列表。")
            opt_params["params"] = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = OptimizerClass(**opt_params)
        return optimizer

    @staticmethod
    def get_scheduler(optimizer, cfg, num_training_steps: int):
        """
        构建调度器
        原逻辑:
          - Warmup(10%): 0.001 -> 1.0
          - Decay(90%):  1.0 -> 0.001
        """
        sched_config = cfg.lr_scheduler_config
        params = sched_config.params

        # 获取配置参数
        warmup_ratio = params.get("warmup_ratio", 0.0)
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        # 获取起始和结束因子 (对应原代码的 start_value=0.001, end_value=0.001)
        # 注意：原代码的 LinearParamScheduler 是相对 LR 的乘数
        start_factor = params.get("start_factor", 0.0)
        end_factor = params.get("end_factor", 0.0)

        scheduler_name = sched_config.name

        print(f"初始化 Scheduler: {scheduler_name}")
        print(f"  - Total Steps: {num_training_steps}")
        print(f"  - Warmup Steps: {num_warmup_steps} (Ratio: {warmup_ratio})")
        print(f"  - Start Factor: {start_factor}, End Factor: {end_factor}")

        # 如果是 "linear_warmup_decay" 且指定了非零因子，我们手动实现以保证精确一致
        if "linear" in scheduler_name and (start_factor > 0 or end_factor > 0):

            def lr_lambda(current_step: int):
                # 1. Warmup 阶段
                if current_step < num_warmup_steps:
                    # 进度从 0.0 到 1.0
                    alpha = float(current_step) / float(max(1, num_warmup_steps))
                    # 线性插值: start_factor -> 1.0
                    return start_factor + (1.0 - start_factor) * alpha

                # 2. Decay 阶段
                else:
                    # 剩余步数的进度，从 0.0 到 1.0
                    progress = float(current_step - num_warmup_steps) / float(
                        max(1, num_training_steps - num_warmup_steps)
                    )
                    # 线性插值: 1.0 -> end_factor
                    return 1.0 - (1.0 - end_factor) * progress

            return LambdaLR(optimizer, lr_lambda)

        else:
            hf_name = "linear"
            if "cosine" in scheduler_name:
                hf_name = "cosine"
            elif "constant" in scheduler_name:
                hf_name = "constant"

            return get_scheduler(
                name=hf_name,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
