# AMR-SELOR 基线阶段总结报告

> **文档性质**：技术研究报告  
> **报告日期**：2025-12-15  
> **实验周期**：2025-12-10 至 2025-12-15

---

## 摘要

本报告总结了 AMR-SELOR 基线阶段的研究工作。我们将 SELOR（Self-Explaining LOgic Rules）框架与 AMR（Abstract Meaning Representation）语义表示相结合，实现了从"词法匹配"到"语义推理"的转变。基线系统在 Yelp 情感分类数据集上取得了 63.69% 的测试准确率和 0.7058 的 ROC-AUC，验证了 AMR 语义三元组作为可解释单元的可行性。本报告详细阐述了六阶段实现流程、关键技术决策及实验结果，并展望了下一阶段的优化方向。

---

## 1. 研究背景与动机

### 1.1 可解释机器学习的挑战

深度学习模型在自然语言处理任务中取得了显著成功，但其"黑箱"特性严重制约了在高风险领域（如医疗诊断、法律判决、金融风控）的应用。现有可解释方法主要分为两类：

1. **事后解释（Post-hoc）**：如 LIME、SHAP，通过扰动输入估计特征重要性
2. **内在可解释（Inherently Interpretable）**：如规则学习、注意力机制，在模型结构中嵌入可解释性

SELOR 属于后者，它学习形如 `IF (条件1 AND 条件2 AND ...) THEN 类别` 的逻辑规则作为解释。

### 1.2 SELOR 的局限性

原始 SELOR 使用**词汇原子（Word Atoms）**作为规则条件，例如：

```
IF ("amazing" ≥ 1) AND ("staff" ≥ 1) THEN positive
```

这种词法层面的解释存在以下问题：

| 问题             | 示例                                                          |
| ---------------- | ------------------------------------------------------------- |
| **缺乏语义理解** | 无法区分 "not amazing" 与 "amazing"                           |
| **复述敏感**     | "staff is helpful" 与 "employees are friendly" 被视为完全不同 |
| **解释碎片化**   | 孤立的单词缺乏语义关联                                        |

### 1.3 研究目标

本研究旨在将 SELOR 的原子单元从词汇提升到语义层面，使用 AMR 语义三元组作为规则条件：

```
IF (staff :domain-of amazing) AND (service :mod excellent) THEN positive
```

**核心目标**：

1. 保持 SELOR 的可解释性框架
2. 提升解释的语义完整性
3. 增强对复述变换的鲁棒性

---

## 2. 系统架构

### 2.1 Pipeline 对比

```
原 SELOR Pipeline:
base.py → extract_base_embedding.py → build_atom_pool.py →
sample_antecedents.py → pretrain_consequent_estimator.py → selor.py

AMR-SELOR Pipeline（本研究实现）:
Stage1: extract_triples.py      → 从 AMR 图提取语义三元组
Stage2: build_triple_pool.py    → 构建全局三元组池与 true_matrix
Stage3: pretrain_ce_triple.py   → 预训练 TripleConsequentEstimator
Stage4: amr_selor.py            → 训练 AMR-SELOR 主模型
Stage5: eval_amr_selor.py       → 评估与解释导出
Stage6: run_amr_selor.py        → Pipeline 调度器
```

### 2.2 关键组件变化

| 组件           | 原 SELOR               | AMR-SELOR                   | 变化说明     |
| -------------- | ---------------------- | --------------------------- | ------------ |
| **原子定义**   | 词汇布尔表达式         | AMR 语义三元组              | 根本性变化   |
| **原子来源**   | 全局词频统计           | AMR 解析 + 关系过滤         | 动态提取     |
| **原子表示**   | one-hot 查表           | true_matrix @ CLS 均值      | 嵌入方式变化 |
| **前件选择器** | GRU + filtered_softmax | GRU + mask + Gumbel-Softmax | 适配动态候选 |
| **后件估计器** | ConsequentEstimator    | TripleConsequentEstimator   | 输入接口变化 |

---

## 3. 阶段实现详解

### 3.1 Stage 1：语义三元组提取

**文件**：`selor_amr/stage1/extract_triples.py`

**目的**：从 AMR 图或预提取的三元组列中解析出结构化的语义三元组。

**核心实现**：

1. **关系过滤策略**：仅保留语义关键的关系类型

```python
KeepRelations = {
    ":ARG0", ":ARG1", ":ARG2",  # 核心语义角色
    ":mod", ":manner", ":domain",  # 修饰关系
    ":location", ":time", ":polarity"  # 情境与否定
}
```

2. **逆向关系规范化**：将 `:ARG0-of` 转换为正向 `:ARG0`

```python
def normalize_role(role: str) -> str:
    return role[:-3] if role.endswith("-of") else role
```

3. **变量概念解析**：将 AMR 变量（如 `f`）解析为其概念（如 `food`）

**输出产物**：

- `train_triples.pkl`：训练集三元组列表
- `test_triples.pkl`：测试集三元组列表
- `global_triple_vocab.pkl`：全局三元组词表（按频率排序）

---

### 3.2 Stage 2：三元组池构建

**文件**：`selor_amr/stage2/build_triple_pool.py`

**目的**：构建全局三元组索引体系和样本-三元组关系矩阵。

**关键技术决策**：

1. **频率过滤**：使用 `--min_freq` 参数过滤低频三元组

```python
vocab, counter = build_vocab_with_freq_filter(all_triples, min_freq=args.min_freq)
```

- 原始唯一三元组：约 27 万
- 过滤后（min_freq=5）：80,497 个

2. **稀疏矩阵存储**：使用 `scipy.sparse.lil_matrix` 避免内存爆炸

```python
mat = sparse.lil_matrix((num_triples, n_data), dtype=np.int8)
```

- 矩阵维度：80,497 × 484,511
- 实际存储：~50 MB（稀疏格式）

3. **Top-K 截断**：限制每个样本的最大三元组数量（默认 50）

**输出产物**：

- `global_triple_vocab.pkl`：过滤后的词表（80,497 个三元组）
- `per_sample_indices.pkl`：每样本的三元组索引列表
- `true_matrix.npz`：稀疏 CSR 矩阵

---

### 3.3 Stage 3：后件估计器预训练

**文件**：`selor_amr/stage3/pretrain_ce_triple.py`

**目的**：预训练 TripleConsequentEstimator，使其能够根据三元组组合预测经验类别分布。

**核心创新**：

1. **基于样本的组合采样**：不同于原 SELOR 的随机组合，我们从实际样本的三元组集合中采样，保证组合真实存在

```python
def sample_combinations_from_samples(train_indices, antecedent_len, num_samples, ...):
    """从实际样本的三元组集合中采样组合，而非随机拼凑"""
```

2. **三元组嵌入计算**：使用 true_matrix 与训练集 CLS 嵌入的加权平均

```python
def build_triple_embedding(true_matrix, train_embed):
    counts = np.array(true_matrix.sum(axis=1)).flatten() + 1e-8
    emb = true_matrix.dot(train_embed.numpy()) / counts[:, None]
    return torch.from_numpy(emb).float()
```

3. **MSE 回归损失**：预测经验类别分布而非分类

**训练结果**：
| Epoch | Mu_Loss | avg_mu_err | F1 |
|-------|---------|------------|-----|
| 1 | 0.0921 | 0.2218 | 0.693 |
| 16 | 0.0686 | 0.1972 | 0.737 |

---

### 3.4 Stage 4：AMR-SELOR 主模型训练

**文件**：`selor_amr/stage4/amr_selor.py`

**目的**：训练完整的 AMR-SELOR 模型，整合 BERT 编码器、GRU 选择器和冻结的 CE。

**关键组件**：

#### 3.4.1 GRU Masked Selector

适配动态三元组候选池的选择器：

```python
class GRUMaskedSelector(nn.Module):
    def forward(self, cls_emb, triple_emb, triple_mask, training):
        # 使用 mask 屏蔽无效三元组
        scores = scores.masked_fill(~triple_mask, float('-inf'))
        # 训练时使用 Gumbel-Softmax 硬采样
        if training:
            prob = F.gumbel_softmax(scores, tau=1.0, hard=True)
        else:
            # 推理时防止重复选择
            triple_mask[can_mask] = triple_mask[can_mask].scatter(...)
```

**关键修复**：推理时更新 mask 防止重复选择同一三元组。

#### 3.4.2 AMRSELOR 主模型

```python
class AMRSELOR(nn.Module):
    def forward(self, batch, triple_emb_table):
        # 1. BERT 编码获取 CLS
        cls_emb = self.bert(input_ids, attention_mask).last_hidden_state[:, 0, :]
        # 2. 查表获取三元组嵌入
        triple_emb = triple_emb_table[triple_indices]
        # 3. GRU 选择器选择 L 个三元组
        select_probs = self.selector(cls_emb, triple_emb, triple_mask, self.training)
        # 4. CE 预测类别分布
        mu, sigma, coverage = self.ce_model(selected_emb)
        # 5. Laplace 平滑计算最终概率
        class_prob = (mu + smooth) / (1 + K * smooth)
```

**训练策略**：

- 优化器：AdamW，lr=1e-4，weight_decay=1e-5
- 学习率调度：ExponentialLR，gamma=0.95
- 早停机制：patience=4

**训练结果**：
| Epoch | Train Acc | Val Acc | Val F1 | Val ROC |
|-------|-----------|---------|--------|---------|
| 1 | 69.86% | 62.68% | 0.617 | 0.697 |
| 7 | 70.23% | 63.78% | 0.633 | 0.706 |

---

### 3.5 Stage 5：评估与解释导出

**文件**：`selor_amr/stage5/eval_amr_selor.py`

**目的**：在测试集上评估模型，并导出人类可读的解释。

**核心功能**：

1. **解释生成**：将选中的三元组索引转换为文本

```python
def idx_to_triple_text(idx, vocab):
    if vocab and 0 <= idx < len(vocab):
        return vocab[idx]  # 返回 "head relation tail" 格式
```

2. **不确定性导出**：同时导出 coverage 和 sigma 用于置信度分析

**输出产物**：

- `metrics.json`：评估指标
- `predictions.csv`：包含 text, label, pred, explanation, coverage, sigma

**测试结果**：
| 指标 | 值 |
|------|-----|
| Accuracy | 63.69% |
| Macro-F1 | 0.6320 |
| ROC-AUC | 0.7058 |
| PR-AUC | 0.6625 |

---

### 3.6 Stage 6：Pipeline 调度器

**文件**：`selor_amr/stage6/run_amr_selor.py` 和 `inference_amr_selor.py`

**目的**：提供统一的命令行接口，串行调度各阶段脚本。

**功能设计**：

- 支持选择性运行各阶段（`--run_stage1` 至 `--run_stage5`）
- 参数统一管理，一处配置全局生效
- 纯推理脚本 `inference_amr_selor.py` 支持增量推理

---

## 4. 实验结果分析

### 4.1 定量结果

| 模型               | Accuracy | Macro-F1 | ROC-AUC |
| ------------------ | -------- | -------- | ------- |
| AMR-SELOR (本研究) | 63.69%   | 0.632    | 0.706   |

### 4.2 解释样例分析

| 输入文本                                             | 预测     | 解释三元组                                    |
| ---------------------------------------------------- | -------- | --------------------------------------------- |
| "The food was amazing and staff was friendly"        | Positive | `food :mod amazing` AND `staff :mod friendly` |
| "I had to wait forever and the service was terrible" | Negative | `i :mod wait` AND `service :mod terrible`     |

### 4.3 问题诊断

通过分析训练过程和解释输出，我们发现以下核心问题：

| 问题              | 现象                   | 根因分析                   |
| ----------------- | ---------------------- | -------------------------- |
| **CE 精度天花板** | 验证 F1 约 0.74        | 8 万三元组太稀疏，统计失效 |
| **解释重复**      | 同一三元组被多次选择   | 推理逻辑缺陷（已修复）     |
| **功能性三元组**  | 如 `i do thing` 被选中 | 无区分度过滤缺失           |

---

## 5. 技术收获与经验总结

### 5.1 关键技术决策

| 决策点         | 选择                   | 理由                           |
| -------------- | ---------------------- | ------------------------------ |
| 三元组嵌入方式 | true_matrix @ CLS 均值 | 简单直接，无需额外编码器       |
| 选择器架构     | GRU + mask             | 复用原 SELOR 结构，降低风险    |
| CE 冻结策略    | 完全冻结               | 避免过拟合，保持预训练知识     |
| 稀疏矩阵       | scipy CSR              | 防止 80k×480k 密集矩阵内存爆炸 |

### 5.2 调试经验

1. **NaN Loss**：通过 `torch.nan_to_num` 和 `clamp` 处理 coverage 边界值
2. **空样本**：增加 `filter_empty_samples` 过滤无三元组样本
3. **重复选择**：推理时更新 mask 防止同一三元组被多次选择
4. **梯度爆炸**：使用 `clip_grad_norm_(max_norm=1.0)` 梯度裁剪

---

## 6. 下一阶段规划

### 6.1 Phase 1：情感区分度过滤（优先级 P0+）

**目标**：将三元组词表从 8 万压缩到 ~5000，保留与标签显著相关的

| 任务          | 方法                       | 预期收益           |
| ------------- | -------------------------- | ------------------ |
| 卡方检验过滤  | 保留 p < 0.05 的三元组     | 提升信噪比         |
| TF-IDF 加权   | 降低高频无区分度三元组权重 | 过滤功能性三元组   |
| 重新预训练 CE | 基于更稠密的统计           | CE 精度提升至 85%+ |

### 6.2 Phase 2：三元组文本编码器（优先级 P1）

**目标**：让模型能处理未见三元组

| 任务                | 实现方式                           | 预期收益      |
| ------------------- | ---------------------------------- | ------------- |
| Triple Text Encoder | DistilBERT 编码 `head rel tail`    | 解决 OOV 问题 |
| 替换查表逻辑        | `embedding = encoder(triple_text)` | 语义泛化      |

### 6.3 Phase 3：可选增强

| 优先级 | 任务                     | 触发条件         |
| ------ | ------------------------ | ---------------- |
| P1     | 指针网络选择器           | 选择质量不佳     |
| P2     | 联合训练（解除 CE 冻结） | 基线稳定后       |
| P2     | 复述鲁棒性评测           | 需验证解释一致性 |

---

## 7. 结论

本研究完成了 AMR-SELOR 基线系统的设计与实现，验证了将 AMR 语义三元组作为可解释单元的技术可行性。虽然当前性能（63.69% 准确率）尚未超越原 SELOR，但已暴露出核心瓶颈（三元组稀疏性），为后续优化指明了方向。下一阶段将通过情感区分度过滤和三元组文本编码器双管齐下，突破当前性能天花板。

---

## 参考文献

1. **SELOR**: Self-Explaining LOgic Rules for Neural Networks
2. **SPRING**: Symmetric Pattern-based Neural AMR Parser
3. **Penman**: A Python library for AMR graphs

---

## 附录 A：文件结构

```
SELOR-main/selor_amr/
├── stage1/
│   └── extract_triples.py       # 三元组提取
├── stage2/
│   ├── build_triple_pool.py     # 词表与 true_matrix 构建
│   └── triple.py                # 数据结构
├── stage3/
│   ├── extract_cls_embedding.py # CLS 嵌入提取
│   └── pretrain_ce_triple.py    # CE 预训练
├── stage4/
│   └── amr_selor.py             # 主模型训练
├── stage5/
│   └── eval_amr_selor.py        # 评估与解释导出
└── stage6/
    ├── run_amr_selor.py         # Pipeline 调度
    └── inference_amr_selor.py   # 纯推理
```

---

_文档版本：v1.0_  
_作者：研究团队_
