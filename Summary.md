# 1st-place-solution
## 概述
*   **核心策略**：后缀分类建模 + 32B大模型全量微调 + 极端的工程优化（显存/计算）+ 多种子集成验证。

## 数据预处理 (Data Processing)
*   **去重**：删除了重复数据，剩余 35,960 条样本。
*   **分层**：基于 `Category` 标签进行 Stratified 5-Fold 交叉验证。

## 建模方法 (Modeling): Suffix Classification
将问题建模为**后缀分类任务 (Suffix Classification Task)**。
给定相同的上下文（Context/Prefix），训练模型从一组候选后缀中预测正确的那个。

### 1. 输入格式
Context (Prefix) 格式如下：
```text
<|im_start|>user
**Question:** {QuestionText}
**Choices:** {MC_Choices}
**Correct Answer:** {Answer}
**Common Misconceptions:** {MisconceptionCandidates}
**Student Answer:** {MC_Answer}
**Student Explanation:** {StudentExplanation}
<|im_end|>
<|im_start|>assistant
```
Suffixes (预测目标) 格式如下（提交格式）：
```text
False_Correct:NA<|im_end|>
```

### 2. 候选集构建
每个 `QuestionId` 根据其可能的误解（Misconception）有 8、10 或 12 个可能的后缀候选。

### 3. 模型结构与注意力机制
*   **特征提取**：提取 `[prefix ++ suffix0, prefix ++ suffix1, ...]` 的最后一个 token 的特征。
*   **分类头**：输入到 `nn.Linear(hidden_size, 1)` 获取 logits，计算交叉熵损失 (Cross-Entropy Loss)。
*   **Prefix-Shared Attention (FlexAttention)**：
    *   为了提高效率，实际上将输入组织为 `prefix ++ suffix0 ++ suffix1 ++ ...` 的形式。
    *   使用自定义的 Attention Mask，确保不同后缀之间互不可见，但都能看到 Prefix。

```python
def custom_mask(b, h, q_idx, kv_idx):
    """
    q_idx: query index
    kv_idx: key/value index
    suffix_ids: 标识当前token属于哪个后缀，-1表示prefix
    doc_ids: 标识不同的样本
    """
    causal = q_idx >= kv_idx
    # 允许看到prefix (suffix_ids == -1)
    is_prefix = suffix_ids[kv_idx] == -1
    # 允许看到同一个后缀内部的前序token
    same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx])
    # 必须是同一个样本
    same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
    
    return causal & (same_suffix | is_prefix) & same_doc
```

## 模型选择
*   **Qwen/Qwen3-32B**
*   **zai-org/GLM-Z1-32B-0414**
*   实验结论：模型越大效果越好；GLM 和 Qwen 的 32B 版本表现最佳。

## 训练策略 (Training)
### 1. 硬件优化：Offload Adam
*   **背景**：为了在单张 A100 (80G) 上全参数微调 32B 模型。
*   **方法**：使用 **Offload Adam** 优化器。将优化器状态（占显存大头）卸载到 CPU 内存，GPU 只保留模型参数和梯度进行计算。
*   **超参**：epoch=1, batch_size=32, learning_rate=1e-5。

### 2. 验证策略：多种子集成 (Multi-Seed Ensemble)
这是本方案最关键的提分点之一。
*   **问题**：由于标签噪声（主要是 `Neither` 类），单次训练（Single Seed）的验证分数非常不稳定且不可靠。
*   **发现**：
    *   不能相信单种子的验证分数。
    *   多折（Multi-fold）集成不如多种子（Multi-seed）集成效果好。
    *   **Loss 比 MAP@3 指标更值得信赖**。
*   **解决方案**：
    *   在 Qwen3-8B 上测试发现，**3个不同 Seed 的集成** 结果趋于稳定。
    *   最终方案：使用全量数据（Full Dataset），跑 **3个不同的 Seed**（每个 Seed 数据格式略有不同），然后集成。

### 3. 辅助任务：Auxiliary SFT Loss
*   **生成理由**：利用超大模型 `Qwen/Qwen3-235B-A22B-Thinking-2507-FP8` 生成每个标签的简短“理由/解释”（Justification）。
*   **训练**：在训练时加入辅助 SFT Loss，让模型学习生成这些理由。
*   **结论**：对 7B 级别的小模型有提升，对 32B 大模型提升微乎其微。最终虽然加进了集成，但贡献很小。

## 推理优化 (Inference)
面临算力（Compute）和显存（Memory）的双重挑战，特别是在 Kaggle 的 T4 显卡上。

### 1. 计算优化 (Compute): W8A8 INT8 量化
*   **瓶颈**：T4 显卡在 FP16 下实际算力只有 20 TFLOPS 且不稳定。
*   **方案**：使用 **LMDeploy** 框架配合 **SmoothQuant (alpha=0.75)**。
*   **效果**：将模型量化为 W8A8 INT8 格式，算力提升至稳定的 40+ TFLOPS。验证集分数量化前后几乎无损。

### 2. 显存优化 (Memory): 逐层推理 (Layer-wise Inference)
*   **挑战**：32B 模型无法塞进 T4 (16G) 显存。
*   **方案**：
    *   显存中只初始化并保留 **2层 Transformer Layer**。
    *   **流水线重叠 (Overlapping)**：在计算当前层（Layer N）的同时，从磁盘读取下一层（Layer N+1）的权重覆盖显存。
    *   **Batch Size**：每次 Forward 处理 640 个样本（40 micro-batches x 16 samples）以保持 GPU 满载。
*   **速度**：双 T4 显卡推理 16,000 条样本耗时约 65 分钟。

## Kaggle 环境踩坑
*   **存储问题**：
    *   `/kaggle/input`：读取极慢。
    *   `/tmp/`：本地写时复制（CoW）存储，有容量限制且文件无法真正删除（导致爆盘）。
    *   `/kaggle/working`：常规本地存储，可删除文件释放空间。
*   **影响**：最初将模型权重层存在 `/tmp` 导致崩溃，最终只来得及提交 4 个模型的集成（原计划 6 个），但依然拿到了第一。


# 3rd-place-solution
## 核心策略概述
*   **双流派融合**：Monsaraida 使用 **SequenceClassification** (判别式)，Masaya 使用 **CausalLM** (生成式)。
*   **Prompt 关键技巧**：在输入中加入所有选项（Choices），而不仅仅是学生选的那个。
*   **推理加速**：针对 T4 显卡优化精度（float16）和 Padding 策略，实现 4 倍加速。

## Monsaraida's Part (Sequence Classification)
### Prompt 工程
*   **加入选项上下文 (Choices)**：
    *   **改动**：将原本的 `{Answer}` 字段改为 `{Choices / Selected}`。
    *   **逻辑**：不仅告诉模型学生选了什么，还告诉模型**有哪些选项可选**。这能帮助模型通过对比选项推断误解（CV/LB +0.001）。
*   **先验提示 (Hints)**：
    *   针对每个 `QuestionId`，在 Prompt 中列出该问题**可能出现的误解类型**（按出现概率排序）。(CV +0.0003)。

### 建模方法
*   **基座模型**：`Qwen/Qwen3-14B` (在测试了 Gemma-2, DeepSeek-Math, Mistral 等后选定)。
*   **任务类型**：`AutoModelForSequenceClassification`。
*   **多任务学习 (Multi-task Learning)**：
    *   主任务：65分类 (Category: Misconception)。
    *   辅助任务1：2分类 (True / False)。
    *   辅助任务2：3分类 (Correct / Misconception / Neither)。
    *   辅助任务3：36分类 (仅 Misconception 的细分)。

### 训练策略 (泛化性优化)
*   **R-Drop (Regularized Dropout)**：通过一致性约束减少过拟合 (CV +0.001)。
*   **AWP (Adversarial Weight Perturbation)**：在 R-Drop 基础上进一步微调。
*   **EMA (Exponential Moving Average)**：提高模型稳定性。

### 推理优化 (Speedup)
*   **多阶段推理**：
    1.  全量测试集推理。
    2.  按置信度排序，对置信度最低的 50% 样本进行重推理（Ensemble）。
*   **T4 显卡加速技巧 (4x Speedup)**：
    1.  **精度转换**：Kaggle T4 不支持 `bfloat16`，转为 `float16` (2x 加速)。
    2.  **去 Padding**：设置 `padding=False` 而非 `max_length`，减少无效计算 (2x 加速)。

## Masaya's Part (CausalLM)
### 核心逻辑 (Question-wise Label Pruning)
*   **假设**：每个 `QuestionId` 对应的误解标签是固定的（2~5个）。
*   **策略**：
    *   限制生成空间。Prompt 中**只列出该问题可能出现的误解选项**，而不是让模型从 65 个类别里大海捞针。
    *   **去前缀**：移除了 `True_/False_` 前缀（这部分通过规则判断 `Is Correct?` 补全），进一步减少预测 Token 数。
    *   **效果**：LB/CV +0.002~0.003，且推理速度提升 3 倍。

### Prompt 示例
```text
Question Text: ...
Choices: (1)... (2)...
Student Answer: ...
Is Student Answer Correct: No
Student Explanation: ...

You must choose the your answer from the following options:
A. Correct:NA   
B. Neither:NA   
C. Misconception:FlipChange   
...
```

### 级联推理 (Cascade Inference)
为了在有限时间内使用 72B 大模型：
1.  **Easy Samples**：先用 `Qwen2.5-14B/32B` 推理所有样本。
2.  **Hard Samples**：筛选出 Top1 概率较低的样本。
3.  **Refine**：仅对这些难样本使用 `Qwen2.5-72B-Instruct-GPTQ` 进行重推理（CPU Offload + 量化）。

### 训练配置
*   **模型**：Qwen2.5 系列 (14B, 32B, 72B)。
*   **LoRA**：r=16, target_modules=全线性层。
*   **超参**：Epoch=2, LR=1e-4 (高学习率有效，但易崩溃)。
*   **EMA**：用于 72B 单一最佳模型。

## 没起作用的尝试 (Not Worked)
*   除 Qwen 以外的模型 (Llama 3.3, Gemma 2)。
*   Pair-wise CausalLM (推理太慢)。
*   固定选项 Token (如强制 A=Correct, B=Neither，效果不佳)。
*   使用其他数学竞赛 (Eedi) 的预训练模型。
*   针对 QuestionId=31778 的标签噪声修正 (反而降低了 LB)。

# 4th-place-solution
## 概述
*   **核心策略**：Pointwise Reranking（点对点重排序）+ 知识蒸馏 + Prompt Engineering（Few-shot）。
*   **模型架构**：主要使用 Qwen 系列，利用 72B 模型作为 Teacher 蒸馏 8B 模型。
*   **特殊处理**：针对 Kaggle T4 环境的推理限制，采用了特殊的训练/验证数据增强策略来规避 FP16 溢出问题。

## 数据预处理 (Data Processing)
*   **分层策略 (StratifiedKFold)**：
    *   使用 5 折交叉验证。
    *   **验证集选择技巧**：特意选择了一个 CV 分数**略低于平均水平**的 fold 作为主要验证集，避免选择分数过高或过低的 fold，以保证验证的鲁棒性。
*   **特定数据修正**：
    *   针对 `QuestionId 31778` 的 12 条数据，将错误的 `Category: True`（实际答案是 6，学生选 9 却标了 True）进行了逻辑修正，该修正应用于训练和推理阶段。
*   **无合成数据**：未使用外部生成的合成数据。

## 建模方法 (Modeling): Pointwise Reranking
将任务视为**点对点分类（Pointwise Classification）**，但在 Batch 层面进行了特殊约束。

### 1. 任务简化
*   由于 Test Set 和 Train Set 共享相同的 15 个问题，判断答案是否正确（True/False）不需要预测。
*   任务被简化为预测解释的类型：`Correct`, `Misconception`, `Neither`。
*   这使得预测标签从 65 个减少到 **37 个**。

### 2. Batch 构造策略 (关键点)
*   **Group Batching**：
    *   训练时，**每个 Step 只处理一个 `row_id`（即一个问题）**。
    *   该 `row_id` 下的所有 `candidate_targets`（通常 4-6 个候选标签）必须包含在**同一个 Batch** 中。
    *   **目的**：模型可以在同一次 Forward 中看到该问题的所有潜在选项，类似于 Listwise 的输入结构，但计算 Loss 时是 Pointwise 的。

### 3. 模型选择 (Model Ranking)
作者测试了大量 Qwen 系列模型，性能排序如下：
`Qwen3-32B` >= `Qwen3-Reranker-8B` >= `QWQ-32B` >= `Qwen3-14B` >= `Qwen2.5-Math-7B-Instruct` > ...
*   *注：作者原文写为 Qwen3，推测实际指 Qwen2.5 或其微调版本。*
*   **最终选择**：
    *   **Student**：`Qwen3-Reranker-8B` (因其性价比最高，用于迭代优化)。
    *   **Teacher**：`Qwen2.5-Math-72B-Instruct` (用于知识蒸馏)。

## Prompt 工程 (Prompt Engineering)
### Prompt V1 (Base)
基础的分类 Prompt，直接询问诊断是否正确。
```python
<|im_start|>system
You are an expert in detecting grade-school level math misconceptions. The task performs 2 steps: 
1.Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither)
2.Identifies the specific misconception present, if any.<|im_end|>
<|im_start|>user
Question: {Question_Text} ...
Student's Answer: {Student_Answer}
This answer is correct.
Student's Explanation: {Student_Explanation}

Now we judge the student's explanation is Correct, Does this diagnosis is correct? (Yes/No)<|im_end|>
<|im_start|>assistant
```

### Prompt V2 (Few-Shot with Contrastive Samples)
在 V1 基础上，加入了针对该 Target 的 **Few-Shot 样例**。
*   **构造方式**：从训练集中随机采样（训练时 1-3 个，推理时 6 个）属于 `Correct`, `Neither`, `Misconception` 的真实解释作为参考。
*   **效果**：显著提升了 CV 和 LB。

```python
<|im_start|>system
... (Same System Prompt) ...
<|im_start|>user
... (Question Info) ...
Correct's Explanation Samples:
- Because there are 9 triangles... (Sample 1)
- The answer is 3/9... (Sample 2)
--
Neither's Explanation Samples:
- i counted the in-shaded parts...
--
Misconception:WNB's Explanation Samples:
- 6 is the total and 3 is blank...
--
Student's Explanation: {Student_Explanation}

Now we judge the student's explanation exhibited the misconception of WNB, Does this diagnosis is correct? (Yes/No)<|im_end|>
<|im_start|>assistant
```

## 训练策略与推理踩坑 (Training & Inference)
### 1. 知识蒸馏 (Distillation)
*   使用 72B 模型生成的 Soft Labels 指导 8B 模型训练。
*   **效果**：
    *   Prompt_V1 + 8B (1 epoch): CV 0.9473 / LB 0.944
    *   Prompt_V1 + 8B + Distill (1 epoch): CV 0.9480 / LB 0.946

### 2. 全量训练与 FP16 溢出危机
*   **最佳策略**：Prompt_V2 + 8B + Distill + **Full Data (2 epochs)**。
*   **遇到的坑**：
    *   在 Kaggle T4 显卡上，`vllm` 不支持 `bfloat16`。
    *   强制转为 `float16` 后，模型在推理时出现了大量的 **NaN (数值溢出)**，导致预测失效。
*   **临场解决方案**：
    *   放弃全量训练出的 2-epoch 模型。
    *   使用本地验证用的 2-epoch 模型（只训练了 4/5 数据）作为基座。
    *   将剩下的 1/5 验证数据加入，再训练 2 个 epoch。
    *   以此来**模拟**全量数据的训练效果，规避了重新全量训练的时间成本和潜在的 NaN 问题。

### 3. 集成 (Ensemble)
*   **Model 3**: Qwen3-Reranker-8B (Best Single, Prompt V2, Distill, Full Data Simulated) -> LB 0.951
*   **Model 4**: Qwen3-32B (Prompt V1, Distill) -> LB 0.949
*   **Result**: Weighted Ensemble -> LB 0.952 (虽然 PB 略有下降，但泛化性更好)。

## 没起作用的尝试 (Not Worked)
*   **Listwise Ranker**：直接让模型对所有选项排序。
*   **CoT (Chain-of-Thought)**：花费了最多时间，但无提升。
*   **RAG (Retrieval)**：训练了一个 Embedding 模型检索最相关的解释加入 Prompt，CV 提升但 LB/PB 下降（过拟合）。


# 10th-place-solution
## 概述
*   **核心策略**：将任务视为**直接的多分类问题** (Direct Multi-class Classification)，而非生成式任务。
*   **模型阵容**：15 个模型的超大集成 (5-fold × 3 Backbones)。
    *   `Qwen3-Reranker-8B`
    *   `Qwen3-Embedding-8B`
    *   `Qwen2.5-32B-Instruct`
*   **工程亮点**：针对 32B 模型在 T4 上的推理，设计了 **"Split-Quantization-Embed"** 模式，利用 vLLM 的 `embed` 接口加速。

## 数据处理 (Data Processing)
*   **数据清洗 (Hygiene)**：
    *   自动修复 `True/False` 标签与文本描述不一致的逻辑错误。
    *   填充缺失的 Misconception 标签为 `NA`。
*   **复合标签**：预测目标为 `Category + ":" + Misconception` 的组合。
*   **防泄露分层 (Leakage Prevention)**：
    *   **关键点**：**"keep near-duplicates in the same fold"**。
    *   将内容高度相似的样本（Near-duplicates）强制划分到同一折中，防止训练集和验证集之间发生数据泄露。

## 训练策略 (Training Strategy)
### 1. 混合损失函数 (Mixed Loss)
为了解决长尾类别（Long-tail）和难样本问题，结合了 Focal Loss 和 Cross Entropy。
*   **Focal Loss**：针对 Unsmoothed CE 计算，用于挖掘 Hard Examples。
*   **Cross Entropy**：带有 Label Smoothing (0.1)，用于校准概率。
*   **Class-Weight Warmup**：
    *   **前 33% 步数**：不使用类别权重，让模型先学习通用特征。
    *   **后 67% 步数**：应用类别权重，强化对少样本类别的关注。

```python
# 简化的 Loss 逻辑
combined_loss = focal_weight * FocalLoss(raw_logits) + ce_weight * Weighted_CE(logits)
# 权重只在 global_step > total_steps * 0.33 后生效
```

### 2. 量化感知训练 (QLoRA)
*   **8B 模型**：使用 `bitsandbytes` 4-bit NF4 量化 + 梯度检查点 (Gradient Checkpointing)。
*   **参数设置**：
    *   `r=64, alpha=128, dropout=0.05`。
    *   Target Modules: Q/K/V/O + MLP projections。
    *   **Classifier Head**：保存为全精度（不量化）。

## 推理工程优化 (Inference Engineering) - **核心亮点**
为了在 Kaggle 的双 T4 (16G x 2) 环境下跑通 32B 模型并集成，作者采用了一种独特的**拆分推理**策略。

### 1. 8B 模型推理
*   **策略**：**不合并 LoRA**。
*   保存 4-bit 量化的基座模型，推理时动态加载 LoRA Adapter。这样单张 T4 即可处理，双卡可以并行跑不同模型。

### 2. 32B 模型推理 (vLLM Embed Mode)
由于 vLLM 对自定义分类头的支持有限，作者设计了以下流水线：
1.  **Merge & Split**：先合并 LoRA，然后将模型物理拆分为 **Base Model** (Transformer Layers) 和 **Classifier Head** (Linear Layer)。
2.  **Quantize Base**：将 Base Model 进行 **GPTQ-4bit** 量化。
3.  **Keep Head FP16**：分类头保持 FP16 精度以确保分类准确性。
4.  **vLLM Feature Extraction**：
    *   使用 `vLLM` 加载 GPTQ-4bit Base Model。
    *   设置 `task="embed"`，只输出最后一层的 Embedding，不进行 token 生成。
5.  **External Head Inference**：
    *   拿到 Embedding 后，在 PyTorch 中加载 FP16 的 Classifier Head。
    *   执行简单的矩阵乘法 (`logits = head(embeddings)`) 得到结果。

#### 拆分模型代码 (Split Model)
```python
def split_model(args):
    # 加载完整模型
    full_model = AutoModelForSequenceClassification.from_pretrained(...)
    base_model = full_model.model # 提取 Transformer 主干
    classifier_head = full_model.score # 提取分类头 (Linear)

    # 分别保存
    base_model.save_pretrained(args.base_model_save_path)
    torch.save(classifier_head.state_dict(), "classifier_state_dict.bin")
    
    # 保存分类头配置 (Hidden size, num labels) 用于重建
    # ...
```

#### vLLM 推理代码 (Embed + External Head)
```python
def run_inference(args):
    # 1. 重建分类头 (PyTorch)
    classifier_head = nn.Linear(...)
    classifier_head.load_state_dict(torch.load(...))
    classifier_head.to("cuda").eval()

    # 2. 初始化 vLLM (Embed Mode)
    llm = LLM(
        model=args.quantized_model_path,
        quantization="gptq",
        dtype="half",
        task="embed", # 关键：只做 Embedding
        override_pooler_config={"pooling_type": "LAST", "normalize": False}
    )

    # 3. 获取 Embeddings
    embed_outputs = llm.embed(TEST_PROMPTS)
    embeddings = torch.from_numpy(...).to("cuda")

    # 4. 计算 Logits
    with torch.no_grad():
        logits = classifier_head(embeddings)
        probs = torch.softmax(logits, dim=-1)
```

## 集成 (Ensemble)
*   **加权 Logit 融合**：
    *   **32B 模型**：权重 **2.0**。
    *   **8B 模型**：权重 **1.0**。
    *   先对 Logits 加权平均，再取 Top-3。作者发现这比直接平均概率（Probability Averaging）效果更稳定。

## 没起作用的尝试 (Not Worked)
*   **分层多任务分类器 (Hierarchical)**：先分大类再分小类，增加了逻辑复杂性但未提分。
*   **32B 作为 Reranker**：无论是 Pointwise 还是 Listwise，作为第二阶段的重排序模型效果都不如直接做分类器。
*   **TTA (Test Time Augmentation)**：推理时增强，耗时且无收益。



