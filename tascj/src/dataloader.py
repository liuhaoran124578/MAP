import torch
from torch.utils.data import DataLoader
from dataset import DatasetManager
from functools import partial 

class DataLoaderManager:
    @staticmethod
    def get_dataloader(cfg, tokenizer, mode="train"):
        """
        构建 DataLoader
        
        Args:
            cfg: ExperimentConfig 配置对象
            tokenizer: 分词器
            mode: "train", "val", "test"
        """
        

        dataset = DatasetManager.load_dataset(cfg, tokenizer, mode=mode)
        
        # 2. 根据模式设置参数
        if mode == "train":
            batch_size = cfg.train_batch_size
            shuffle = True
            drop_last = True  
        else:
            # val 或 test
            batch_size = cfg.eval_batch_size
            shuffle = False
            drop_last = False

        # 3. 打印信息
        print(f"Initializing DataLoader for [{mode}]")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Shuffle: {shuffle}")
        print(f"  - Num Workers: {cfg.num_workers}")
        print(f"  - Dataset Len: {len(dataset)}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=partial(dataset.collate_fn, block_size=cfg.dataset_config.block_size)
        )
        
        return loader