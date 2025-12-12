#### 1. 后件估计器 _Train_ 详解（原 SELOR）

##### 1.1 核心目标

后件估计器学习：**给定规则（原子组合） α → 预测 p(y|α)**

这是一个**回归任务**，不是分类任务！

##### 1.2 训练数据构造（带数值示例）

**Step 1: 准备原子池**

```
原子词表 = ["good", "bad", "food", "service", "great", ...] (共 10,000 个)
true_matrix[i,j] = 1 表示原子i出现在样本j中
训练样本 = 56,000 个（标签 0/1）
```

**Step 2: 采样原子组合**

```python
# 采样 10,000 个长度为 k 的组合（k = 1,2,3,4）
组合1: α = {"good", "food"}           # 长度2
组合2: α = {"bad", "service", "slow"} # 长度3
组合3: α = {"great"}                  # 长度1
...
```

**Step 3: 统计经验概率（关键步骤！）**

```python
# 对于组合 α = {"good", "food"}
覆盖样本 = 找出同时包含 "good" 和 "food" 的所有样本
         = [样本5, 样本23, 样本89, 样本456, ...]  # 假设共 500 个

正面样本数 = 420 个
负面样本数 = 80 个

# 经验概率分布
μ_empirical = [80/500, 420/500] = [0.16, 0.84]
coverage = 500 / 56000 = 0.0089

# 训练目标
X = embedding(["good", "food"])  # 输入：原子嵌入序列
Y = [0.16, 0.84]                 # 目标：经验概率分布
coverage_target = 0.0089         # 覆盖率目标
```

##### 1.3 训练过程

```python
# 训练数据格式
batch = [
    (emb_α1, μ1, cov1),  # α1={good,food} → μ=[0.16,0.84], cov=0.0089
    (emb_α2, μ2, cov2),  # α2={bad,service,slow} → μ=[0.91,0.09], cov=0.0032
    ...
]

# 前向传播
μ_pred, σ_pred, cov_pred = model(emb_α)

# 损失函数（MSE，不是交叉熵！）
loss = MSE(μ_pred, μ_true) + λ * MSE(cov_pred, cov_true)

# 评估指标
mu_MAE = mean(|μ_pred - μ_true|)  # 概率预测误差
argmax_acc = (argmax(μ_pred) == argmax(μ_true)).mean()  # 类别是否一致
```

##### 1.4 关键特性

| 特性           | 说明                             |
| -------------- | -------------------------------- |
| **训练目标**   | 回归经验概率分布，不是样本标签   |
| **损失函数**   | MSE（最小化概率差距）            |
| **采样数量**   | 10,000 个组合（每个长度）        |
| **覆盖率筛选** | 只保留覆盖 ≥ min_df 个样本的组合 |

##### 1.5 为什么能泛化？

1. **Embedding 空间的连续性**：相似词汇有相似嵌入
2. **Transformer 的组合能力**：学习原子间的交互模式
3. **统计规律的一致性**：训练集中的规律在测试集中大概率成立

##### 1.6 与 AMR-SELOR 的关键差异

| 维度       | 原 SELOR（词汇原子） | AMR-SELOR（三元组）     |
| ---------- | -------------------- | ----------------------- |
| 原子数量   | ~10,000              | ~80,000                 |
| 单原子覆盖 | 高（常见词出现万次） | 低（三元组几乎唯一）    |
| 组合覆盖   | 高（数百样本）       | **极低（通常 1-2 个）** |
| 统计意义   | 强（大样本统计）     | **弱（无法统计）**      |

**→ 这就是为什么我们需要重新思考 AMR-SELOR 的训练策略**

---





#### 2. 前件选择器（Antecedent Selector）详解

##### 2.1 核心目标

前件选择器学习：**给定样本和候选项，选择哪些原子/三元组组成规则**

这是一个**序列决策任务**，逐步选择 L 个前件，形成可解释规则。

##### 2.2 原 SELOR 的 AtomSelector

**架构**：

```
输入: CLS 嵌入 h [B, H]
      满足向量 x_ [B, num_atoms]  (布尔，标记该样本包含哪些原子)
      ↓
GRU(hidden_dim) → 状态更新
      ↓
Linear(hidden, num_atoms) → 分数 [B, 5000]
      ↓
filtered_softmax(分数, x_) → 屏蔽不满足的原子
      ↓
Gumbel-Softmax(hard=True) → 选中概率 (one-hot)
      ↓
查表 atom_embedding[选中] → 下一步输入
      ↓
重复 L 次 → 输出 L 个原子的选择概率 [B, L, num_atoms]
```

**关键特点**：

| 特点     | 说明                                  |
| -------- | ------------------------------------- |
| 候选池   | 全局固定，~5000 个词汇原子            |
| 输出层   | `Linear(hidden, 5000)` 固定维度       |
| 满足条件 | `x_` 布尔向量表示"该样本包含哪些原子" |
| 屏蔽方式 | `x_[不满足] = -inf` 后再 softmax      |

**代码核心**（`net.py` AtomSelector）：

```python
def forward(self, h, x_):
    for j in range(antecedent_len):
        _, h_n = self.gru(cur_input, cur_h)
        out = self.gru_head(h_n)           # [B, num_atoms]
        out[~x_] = float('-inf')           # 屏蔽不满足的原子
        prob = F.gumbel_softmax(out, hard=True)
        cur_input = h + matmul(prob, atom_embedding)
```

##### 2.3 AMR-SELOR 的 GRUMaskedSelector

**架构**：

```
输入: CLS 嵌入 h [B, H]
      三元组索引 triple_indices [B, T]  (该样本的候选三元组)
      三元组掩码 triple_mask [B, T]     (有效位置标记)
      ↓
查表 triple_emb_table[indices] → 候选嵌入 [B, T, H]
      ↓
GRU(hidden_dim) → 状态更新
      ↓
点积注意力 bmm(triple_emb, h) → 分数 [B, T]
      ↓
masked_fill(~mask, -inf) → 屏蔽 padding 位置
      ↓
Gumbel-Softmax(hard=True) → 选中概率 (one-hot)
      ↓
bmm(prob, triple_emb) → 选中嵌入作为下一步输入
      ↓
重复 L 次 → 输出 L 个三元组的选择概率 [B, L, T]
```

**关键特点**：

| 特点     | 说明                        |
| -------- | --------------------------- |
| 候选池   | 每样本动态，~50 个三元组    |
| 分数计算 | 点积注意力（适应可变长度）  |
| 候选来源 | 该样本的 AMR 图提取的三元组 |
| 屏蔽方式 | `mask[padding] = -inf`      |

**代码核心**（`amr_selor.py` GRUMaskedSelector）：

```python
def forward(self, cls_emb, triple_emb, triple_mask, training):
    for step in range(antecedent_len):
        _, cur_h = self.gru(cur_input, cur_h)
        # 点积注意力（核心差异！）
        scores = bmm(triple_emb, cur_h.unsqueeze(-1)).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~triple_mask, float('-inf'))
        prob = F.gumbel_softmax(scores, hard=True)
        selected = bmm(prob.unsqueeze(1), triple_emb).squeeze(1)
        cur_input = (cls_emb + selected).unsqueeze(0)
```

##### 2.4 核心差异对比

| 维度             | 原 SELOR AtomSelector   | AMR-SELOR GRUMaskedSelector |
| ---------------- | ----------------------- | --------------------------- |
| **候选池大小**   | 全局固定 ~5000          | 每样本动态 ≤50              |
| **候选来源**     | 全局词汇原子表          | 该样本的 AMR 三元组         |
| **分数计算**     | `Linear(h) → [B, 5000]` | `dot(h, emb) → [B, T]`      |
| **适应动态长度** | ❌ 不需要               | ✅ 必须                     |
| **输出维度**     | 固定 num_atoms          | 可变 T (每样本不同)         |
| **查表时机**     | 选择后查表              | 选择前查表                  |

##### 2.5 为什么改用点积注意力？

**原 SELOR 可行的原因**：

```
全局原子池固定 (5000 个)
→ 可以用 Linear(hidden, 5000) 直接输出分数
→ 所有样本共享同一个输出层
```

**AMR-SELOR 不行的原因**：

```
每样本的三元组候选不同，数量不同 (20-50 个)
→ 无法用固定维度的 Linear 层
→ 改用点积注意力：score = triple_emb · h
→ 自动适应任意候选数量
```

##### 2.6 学习信号传递

```
分类损失 (NLL Loss)
    ↓ 反向传播
Laplace 平滑 → class_prob
    ↓
后件估计器 (CE, 冻结) → mu, coverage
    ↓
选中三元组嵌入 → selected_emb
    ↓
Gumbel-Softmax (可微) → select_probs
    ↓
GRU + 点积注意力 ← 梯度更新
```

**关键**：Gumbel-Softmax 的 `hard=True` + straight-through estimator 使得离散选择可微分，梯度可以从分类损失一路传回选择器。

##### 2.7 总结

| 保持一致       | 核心改变              |
| -------------- | --------------------- |
| GRU 顺序选择   | Linear → 点积注意力   |
| Gumbel-Softmax | 固定输出 → 动态输出   |
| 状态传递机制   | 全局候选 → 每样本候选 |
| 冻结 CE 监督   | 词汇原子 → 语义三元组 |

**AMR-SELOR 的前件选择器是原 SELOR 的"动态化"改造**：保留核心思想（GRU 序列决策 + Gumbel 采样），适配三元组的稀疏动态特性。
