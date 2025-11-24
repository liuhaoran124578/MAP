import yaml
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union


class DatasetConfig(BaseModel):
    csv_file: str
    prompt_name: str
    block_size: int 


class LLMConfig(BaseModel):
    backbone: str
    model_type: str = "qwen3"
    num_labels: int = 1
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    use_lora: bool = False
    bitsandbytes: bool = False

    resume_from_checkpoint: Optional[str] = None
    lora_params: Optional[Dict[str, Any]] = None


class LRSchedulerConfig(BaseModel):
    name: str
    params: Dict[str, Union[float, int]]


class OptimizerConfig(BaseModel):
    name: str
    params: Dict[str, Any]


class ExperimentConfig(BaseModel):
    # General
    exp_type: str
    exp_name: str
    seed: int

    # Train Phase
    n_folds: int
    test_fold: int
    max_epochs: int

    # Logging & Optimization Control
    log_interval: int
    eval_interval: int
    clip_grad: bool = False
    cast_to_bf16: bool = False
    # System
    num_workers: int

    # Batching
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int = 1

    work_dir: Optional[str] = None
    # Components
    dataset_config: DatasetConfig
    llm_config: LLMConfig
    lr_scheduler_config: LRSchedulerConfig
    optimizer_config: OptimizerConfig

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


if __name__ == "__main__":
    try:
        # 假设上面的 yaml 内容保存为 config.yaml
        cfg = ExperimentConfig.from_yaml("/root/autodl-tmp/MAP/config/tascj/x_0.yaml")

        print("✅ 配置加载成功")
        print("-" * 30)
        print(f"Exp Name:   {cfg.exp_name}")
        print(f"Seed:       {cfg.seed}")
        print(
            f"Optimizer:  {cfg.optimizer_config.name} (LR={cfg.optimizer_config.params['lr']})"
        )
        print(f"Dataset:    {cfg.dataset_config.csv_file}")
        print(f"Test Fold:  {cfg.test_fold}")
        print(f"Cast BF16:  {cfg.cast_to_bf16}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
