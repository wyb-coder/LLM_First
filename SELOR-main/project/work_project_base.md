# AMR-SELOR 项目实施规划文档

> **项目目标**：将 SELOR 框架改造为 AMR-SELOR，用 AMR 语义三元组替代词汇原子，实现从"词法匹配"到"语义推理"的转变，提升模型的可解释性质量和复述鲁棒性。

> **实施原则**：新增/改版代码集中放在 `selor_amr/` 下按 stage 分层，原有文件保持不变，便于对照回滚。
>
> **代码组织规范**：
>
> - 新增脚本：`selor_amr/stage{N}/xxx.py`（如 `stage1/extract_triples.py`）
> - 新增工具类：`selor_amr/utils/xxx.py`（如 `utils/triple.py`）
> - 需修改的原库函数：在 `selor_utils/` 原位置修改（如 `selor_utils/net.py`）
> - 输出数据：`result/triples/`、`result/embeddings/` 等

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构对比](#2-架构对比)
3. [文件修改清单](#3-文件修改清单)
4. [阶段一：语义解析模块](#4-阶段一语义解析模块)
5. [阶段二：三元组嵌入模块](#5-阶段二三元组嵌入模块)
6. [阶段三：后件估计器改造](#6-阶段三后件估计器改造)
7. [阶段四：前件生成器改造](#7-阶段四前件生成器改造)
8. [阶段五：数据集与工具适配](#8-阶段五数据集与工具适配)
9. [阶段六：主流程整合](#9-阶段六主流程整合)
10. [验证方案](#10-验证方案)
11. [风险与缓解](#11-风险与缓解)
12. [待拓展（增强项与优先级）](#12-待拓展增强项与优先级)
13. [解释生成与论文模块对应](#13-解释生成与论文模块对应)

---

## 1. 项目概述

### 1.1 核心改造目标

| 维度         | 原 SELOR                | AMR-SELOR（首版基线）                  |
| ------------ | ----------------------- | -------------------------------------- |
| **原子定义** | 词汇/统计特征布尔表达式 | AMR 语义三元组                         |
| **原子示例** | `"amazing" >= 1`        | `(staff, :domain-of, amazing-01)`      |
| **原子来源** | 全局词汇表（静态）      | 每个样本的 AMR 图（动态，top-K）       |
| **前件生成** | GRU 从固定原子池选择    | 先用 GRU+mask 复用，指针网络为可选增强 |
| **可解释性** | 词法层面                | 语义层面                               |

### 1.2 Pipeline 对比

```
原SELOR Pipeline:
base.py → extract_base_embedding.py → build_atom_pool.py →
sample_antecedents.py → pretrain_consequent_estimator.py → selor.py

AMR-SELOR Pipeline（首版基线，优先复用 GRU+mask）：
【状态】已有 AMR 解析结果（train_with_amr.csv / test_with_amr.csv，yelp）。从现有 AMR 列开始：
extract_triples.py → build_triple_pool.py → pretrain_ce_triple.py → amr_selor.py
```

---

## 2. 架构对比

### 2.1 推理流程对比

**原 SELOR 推理流程**：

```
输入句子
    → BERT编码 → 嵌入向量h
    → 检查满足哪些原子 → x_ (布尔向量)
    → AtomSelector(GRU) → 选择原子组合
    → ConsequentEstimator → 类别概率
    → 输出预测 + 解释规则
```

**AMR-SELOR 推理流程**（首版基线）：

```
输入句子
    ├→ BERT编码 → 嵌入向量h
    └→ SPRING解析 → AMR图 → 三元组集合T_x
    → GRU + mask 选择三元组组合（指针网络为可选增强）
    → TripleConsequentEstimator → 类别概率
    → 输出预测 + 语义规则解释
```

### 2.2 关键组件变化

| 组件                | 原 SELOR                          | AMR-SELOR                           | 变化说明                     |
| ------------------- | --------------------------------- | ----------------------------------- | ---------------------------- |
| 原子池              | 全局固定，基于词频                | 动态，每样本独立，top-K 截断        | **根本性变化**               |
| 原子表示            | one-hot → 查表得嵌入              | 三元组文本 → 编码或 true_matrix@CLS | **表示方式变化**             |
| AtomSelector        | GRU + fixed output dim            | 先复用 GRU+mask，指针网络可选       | **架构可升级，先低风险复用** |
| ConsequentEstimator | Transformer + 固定 atom_embedding | Transformer + 动态 triple_embedding | **输入变化，冻结/复用更稳**  |

---

## 3. 文件修改清单

### 3.1 新增文件

| 文件                                           | 功能描述                             | 所属阶段 | 状态    |
| ---------------------------------------------- | ------------------------------------ | -------- | ------- |
| `selor_amr/stage1/extract_triples.py`          | 从 AMR CSV 提取三元组                | 阶段一   | ✅ 完成 |
| `selor_amr/stage2/build_triple_pool.py`        | 构造全局三元组池、true_matrix、top-K | 阶段二   | ✅ 完成 |
| `selor_amr/stage2/triple.py`                   | 三元组相关数据结构和工具函数         | 阶段二   | ✅ 完成 |
| `selor_amr/stage3/extract_cls_embedding.py`    | 提取 BERT CLS 嵌入                   | 阶段三   | ✅ 完成 |
| `selor_amr/stage3/pretrain_ce_triple.py`       | 预训练三元组后件估计器               | 阶段三   | ✅ 完成 |
| `selor_amr/stage4/amr_selor.py`                | AMR-SELOR 主训练脚本                 | 阶段四   | ✅ 完成 |
| `selor_amr/stage5/eval_amr_selor.py`           | 评估与解释导出                       | 阶段五   | ✅ 完成 |
| `selor_amr/stage6/run_amr_selor.py`            | AMR-SELOR 完整 pipeline 调度器       | 阶段六   | ✅ 完成 |
| `selor_amr/stage6/inference_amr_selor.py`      | 纯推理脚本                           | 阶段六   | ✅ 完成 |
| `selor_amr/utils/compute_triple_embeddings.py` | （可选）三元组文本编码器             | 待拓展   | —       |

### 3.2 修改文件

| 文件                        | 原功能       | 修改内容                                                          | 所属阶段  |
| --------------------------- | ------------ | ----------------------------------------------------------------- | --------- |
| `selor_utils/net.py`        | 神经网络模块 | 先复用 GRU+mask，新增 TripleConsequentEstimator；指针网络列为可选 | 阶段三/四 |
| `selor_utils/dataset.py`    | 数据集处理   | 新增三元组数据集类；输出三元组 mask/列表                          | 阶段五    |
| `selor_utils/train_eval.py` | 训练评估     | 适配三元组解释生成/评估逻辑                                       | 阶段五    |
| `selor_utils/utils.py`      | 工具函数     | 新增三元组相关参数解析                                            | 阶段五    |

### 3.3 保留文件（无需修改）

| 文件                        | 原功能             | 保留理由                            |
| --------------------------- | ------------------ | ----------------------------------- |
| `base.py`                   | 训练基础 BERT 模型 | BERT 仍作为上下文编码器（可选使用） |
| `extract_base_embedding.py` | 提取样本嵌入       | 可选，用于计算三元组嵌入的替代方案  |

### 3.4 废弃文件

| 文件                    | 原功能             | 废弃理由                                            |
| ----------------------- | ------------------ | --------------------------------------------------- |
| `build_atom_pool.py`    | 基于词频构建原子池 | 被 SPRING 三元组池构建替代                          |
| `sample_antecedents.py` | 采样词汇原子组合   | 如需三元组组合采样，改用 sample_triple_combinations |

---

## 4. 阶段一：语义解析模块

### 4.1 `spring_parse_all.py` [新增]

**功能**：批量调用 SPRING 解析器，将数据集中所有句子转换为 AMR 图

**状态**：已完成（yelp 数据已生成 `train_with_amr.csv` / `test_with_amr.csv`，含 AMR 列，可直接作为后续输入）。脚本落点：`selor_amr/stage1/extract_triples.py`（如需重跑）。

**依赖**：

- `spring_amr` 模块（已存在于 `spring-main/spring_amr/`）
- 预训练 SPRING 模型检查点

**输入**：

- 训练集/验证集/测试集的文本

**输出**：

- `./saved_models/amr_graphs/train_amr.pkl` - 训练集 AMR 图
- `./saved_models/amr_graphs/valid_amr.pkl` - 验证集 AMR 图
- `./saved_models/amr_graphs/test_amr.pkl` - 测试集 AMR 图

> 若已存在 AMR 列（如 `train_with_amr.csv`），可跳过本脚本，直接供后续三元组提取使用。

**核心逻辑**：

```python
from spring_amr import AMRParser

class AMRProcessor:
    def __init__(self, checkpoint_path):
        self.parser = AMRParser.from_pretrained(checkpoint_path)

    def parse_batch(self, sentences: List[str]) -> List[str]:
        """批量解析句子，返回PENMAN格式AMR"""
        return [self.parser.parse(sent) for sent in sentences]
```

**修改依据**：

- 原 SELOR 的原子来源是词频统计，AMR-SELOR 需要语义解析作为新的原子来源
- SPRING 是当前最先进的 AMR 解析器之一，已在项目中准备好

**阶段验收**：

- 抽样 20 条 AMR，人工检查语义合理性；解析成功率记录
- 解析耗时/吞吐统计（便于评估大规模可行性）

---

### 4.2 `extract_triples.py` [新增]

**功能**：从 PENMAN 格式 AMR 图中提取语义三元组

**依赖**：

- `penman` Python 库

**输入**：

- 上一步生成的 AMR 图 pickle 文件

**输出**：

- `./saved_models/triples/train_triples.pkl` - 训练集三元组列表
- `./saved_models/triples/valid_triples.pkl` - 验证集三元组列表
- `./saved_models/triples/test_triples.pkl` - 测试集三元组列表
- `./saved_models/triples/global_triple_vocab.pkl` - 全局三元组词表

**状态**：AMR 输入已就绪（来自 `train_with_amr.csv` / `test_with_amr.csv`），需运行本阶段脚本以生成三元组与 true_matrix。脚本落点：`selor_amr/stage2/build_triple_pool.py`。

**核心逻辑**：

```python
import penman

def extract_triples_from_amr(amr_str: str) -> List[Tuple[str, str, str]]:
    """从PENMAN字符串提取三元组"""
    graph = penman.decode(amr_str)
    triples = []

    for source, role, target in graph.triples:
        # 过滤无意义三元组
        if role in [':instance', ':polarity', ':ARG0', ':ARG1', ':ARG2',
                    ':domain', ':mod', ':manner', ':location', ':time']:
            # 获取概念名（处理变量引用）
            source_concept = get_concept(graph, source)
            target_concept = get_concept(graph, target)
            triples.append((source_concept, role, target_concept))

    return triples
```

**过滤策略**：

- 保留核心语义角色：`:ARG0`, `:ARG1`, `:ARG2`（施事、受事等）
- 保留修饰关系：`:mod`, `:manner`, `:domain`
- 保留否定：`:polarity`（用于捕捉否定语义）
- 过滤过于泛化的关系：如 `:op1`, `:op2`（通常是名字拼接）

**修改依据**：

- 三元组是 AMR-SELOR 的"原子"，需要从 AMR 图中系统提取
- 过滤策略确保提取的三元组具有语义价值

**阶段验收**：

- true_matrix 维度与样本数一致，无全零行；三元组数分布、稀疏度统计
- 抽样三元组与原句语义一致；方向规范（:\*-of）生效
- top-K 截断后，极端空样本占比可接受（若空则补 dummy）

---

### 4.3 `selor_utils/triple.py` [新增]

**功能**：定义三元组相关的数据结构和工具类

**核心类**：

#### 4.3.1 `Triple` 类

```python
@dataclass
class Triple:
    """语义三元组：AMR-SELOR的基本解释单元"""
    head: str           # 头实体/概念
    relation: str       # 关系类型
    tail: str           # 尾实体/概念
    triple_idx: int     # 在全局词表中的索引（可选）

    @property
    def display_str(self) -> str:
        """人类可读的显示字符串"""
        return f"({self.head} {self.relation} {self.tail})"

    def to_text(self) -> str:
        """转换为文本表示，用于编码"""
        return f"{self.head} {self.relation} {self.tail}"
```

#### 4.3.2 `TriplePool` 类

```python
class TriplePool:
    """三元组池：管理一个样本的所有候选三元组"""
    def __init__(self, triples: List[Triple]):
        self.triples = triples
        self.triple2idx = {t.to_text(): i for i, t in enumerate(triples)}

    def num_triples(self) -> int:
        return len(self.triples)

    def get_triple(self, idx: int) -> Triple:
        return self.triples[idx]
```

#### 4.3.3 `GlobalTripleVocab` 类

```python
class GlobalTripleVocab:
    """全局三元组词表：用于后件估计器预训练"""
    def __init__(self):
        self.triple2idx = {}
        self.idx2triple = {}
        self.triple_count = Counter()

    def add_triple(self, triple: Triple):
        text = triple.to_text()
        if text not in self.triple2idx:
            idx = len(self.triple2idx)
            self.triple2idx[text] = idx
            self.idx2triple[idx] = triple
        self.triple_count[text] += 1

    def get_frequent_triples(self, min_count: int) -> List[Triple]:
        """获取高频三元组，用于预训练采样"""
        return [self.idx2triple[self.triple2idx[t]]
                for t, c in self.triple_count.items() if c >= min_count]
```

**修改依据**：

- 类比原 SELOR 的 `Atom` 和 `AtomPool` 类
- 提供统一的三元组操作接口

---

## 5. 阶段二：三元组嵌入模块

**基线做法**：沿用 SELOR 路径，使用 `true_matrix @ train_embeddings`（CLS 均值）生成三元组 embedding；无需新增编码脚本。

**可选增强**：三元组文本编码器（BERT）已移至“待拓展”，不影响主流程正确性。

---

## 6. 阶段三：后件估计器改造

### 6.1 `pretrain_ce_triple.py` [新增]

**原文件**：`pretrain_consequent_estimator.py`

**原功能**：

- 预训练 `ConsequentEstimator` 网络
- 输入：one-hot 原子索引 → 查表得到 `atom_embedding`
- 输出：mu（类别概率）, sigma（不确定性）, coverage（覆盖率）

**新功能（对齐原 SELOR 预训练逻辑）**：

- 采样三元组组合（长度 1..antecedent_len），统计经验分布 \(\hat p(y\mid \alpha)\) 与覆盖率
- 输入：三元组嵌入序列（查表方式），不查表 atom_embedding
- 损失：MSE 回归经验分布（主），覆盖率可作辅助；评估用 MAE/RMSE/Fidelity/argmax-acc

> 基线：`true_matrix @ train_cls_embeddings` 归一化得到三元组 embedding 表，预先存储；训练时按采样组合查表
> 可选增强：实时文本编码、组合式 head/rel/tail 编码，放“待拓展”

**核心流程（基线）**：

```python
# 1) 采样三元组组合 alpha，过滤 min_coverage
# 2) 统计经验分布 mu_hat = count(y|alpha)/n, coverage = n/N
# 3) 查表获取嵌入序列 emb = triple_embedding[alpha]
# 4) MSE(mu_pred, mu_hat) [+ 0.1*MSE(coverage_pred, coverage)]
```

**阶段验收（回归视角）**：

- 采样数充足（如 1e4），有效覆盖率通过 min_coverage 过滤
- 训练/验证 mu_MAE、mu_RMSE 收敛；coverage_MAE 合理；argmax_acc 仅作辅参考
- 产物：`ce_triple_best.pt`、配置文件（hidden_dim/num_classes/antecedent_len/采样配置）

---

### 6.3 `selor_utils/net.py` 修改 - 后件估计器部分

**原组件**：`ConsequentEstimator`（第 117-174 行）

**原功能**：

```python
class ConsequentEstimator(nn.Module):
    def __init__(self, num_classes, hidden_dim, atom_embedding):
        # atom_embedding: 预计算的固定嵌入表 [n_atom, hidden_dim]
        self.atom_embedding = atom_embedding
        self.cp_te = nn.TransformerEncoder(...)  # 6层Transformer
        self.mu_head = nn.Sequential(...)
        self.sigma_head = nn.Sequential(...)
        self.coverage_head = nn.Sequential(...)

    def forward(self, x):
        # x: one-hot tensor [batch, antecedent_len, n_atom]
        emb = torch.matmul(x, self.atom_embedding)  # 查表
        out = self.cp_te(emb)
        ...
```

**新组件**：`TripleConsequentEstimator`

**新功能**：

```python
class TripleConsequentEstimator(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        # 不需要预定义atom_embedding，直接接收嵌入
        self.cp_te = nn.TransformerEncoder(...)  # 保持6层
        self.mu_head = nn.Sequential(...)        # 保持不变
        self.sigma_head = nn.Sequential(...)     # 保持不变
        self.coverage_head = nn.Sequential(...)  # 保持不变

    def forward(self, triple_embeddings):
        # triple_embeddings: [batch, antecedent_len, hidden_dim]
        # 直接输入，无需查表
        out = self.cp_te(triple_embeddings)
        out = torch.mean(out, dim=1)

        mu = F.softmax(self.mu_head(out), dim=-1)
        sigma = torch.exp(self.sigma_head(out))
        coverage = torch.sigmoid(self.coverage_head(out))

        return mu, sigma, coverage
```

**修改依据**：

- 原版依赖固定的 `atom_embedding` 查表
- 新版接收动态计算的三元组嵌入
- Transformer 结构保持不变，只改变输入方式

---

## 7. 阶段四：前件生成器改造

> **基线方案**：复用原 AtomSelector 路径（GRU + filtered_softmax + mask，固定 num_triples = 词表大小），三元组嵌入通过查表获取。
> **可选增强**：指针网络方案列为可选增强，放入“待拓展”。

### 7.1 `selor_utils/net.py` 修改 - 前件生成器部分

**原组件**：`AtomSelector`（第 176-273 行）

**原功能**：

```python
class AtomSelector(nn.Module):
    def __init__(self, num_atoms, antecedent_len, hidden_dim, atom_embedding):
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.gru_head = nn.Linear(hidden_dim, num_atoms)  # 固定输出维度
        self.atom_embedding = atom_embedding

    def filtered_softmax(self, x, x_, pos, pre_max_index):
        # x_: 布尔向量，表示哪些原子被满足
        x[torch.logical_not(x_)] = float('-inf')  # 屏蔽不满足的原子
        x = F.gumbel_softmax(x, tau=1, hard=True)
        return x

    def forward(self, cls_emb, x_):
        # cls_emb: BERT的CLS嵌入 [batch, hidden_dim]
        # x_: 满足的原子mask [batch, num_atoms]
        for j in range(antecedent_len):
            _, h_n = self.gru(cur_input, cur_h_0)
            out = self.gru_head(h_n)  # [batch, num_atoms]
            prob = self.filtered_softmax(out, x_, j, max_index)
            ...
```

**可选增强组件**：`TriplePointerSelector`（移至"待拓展"）

> ⚠️ **注意**：以下指针网络为增强方案，基线版本复用原 AtomSelector 的 GRU+mask 结构。

**增强功能**（处理动态候选长度）：

```python
class TriplePointerSelector(nn.Module):
    """基于指针网络的三元组选择器"""
    def __init__(self, hidden_dim, antecedent_len):
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.antecedent_len = antecedent_len

    def forward(self, cls_emb, triple_embeddings, triple_mask=None):
        """
        Args:
            cls_emb: BERT的CLS嵌入 [batch, hidden_dim]
            triple_embeddings: 候选三元组嵌入 [batch, num_triples, hidden_dim]
            triple_mask: 可选，标记有效三元组 [batch, num_triples]

        Returns:
            attention_weights: 选择概率 [batch, antecedent_len, num_triples]
        """
        batch_size, num_triples, hidden_dim = triple_embeddings.shape

        selected_probs = []
        cur_input = cls_emb.unsqueeze(0)  # [1, batch, hidden]
        cur_hidden = None

        for step in range(self.antecedent_len):
            # GRU更新状态
            if cur_hidden is not None:
                _, cur_hidden = self.gru(cur_input, cur_hidden)
            else:
                _, cur_hidden = self.gru(cur_input)

            query = cur_hidden.squeeze(0)  # [batch, hidden]

            # 计算对所有候选三元组的注意力分数
            # query: [batch, 1, hidden], key/value: [batch, num_triples, hidden]
            attn_output, attn_weights = self.attention(
                query.unsqueeze(1).transpose(0, 1),
                triple_embeddings.transpose(0, 1),
                triple_embeddings.transpose(0, 1),
                key_padding_mask=~triple_mask if triple_mask is not None else None
            )
            # attn_weights: [batch, 1, num_triples]

            # Gumbel-Softmax采样
            if self.training:
                prob = F.gumbel_softmax(attn_weights.squeeze(1), tau=1, hard=True)
            else:
                _, idx = torch.max(attn_weights.squeeze(1), dim=-1)
                prob = F.one_hot(idx, num_triples).float()

            selected_probs.append(prob)

            # 更新输入：选中的三元组嵌入
            selected_emb = torch.bmm(prob.unsqueeze(1), triple_embeddings)
            cur_input = (cls_emb + selected_emb.squeeze(1)).unsqueeze(0)

        return torch.stack(selected_probs, dim=1)  # [batch, antecedent_len, num_triples]
```

**关键设计差异**：

| 方面     | 原 AtomSelector                  | 新 TriplePointerSelector            |
| -------- | -------------------------------- | ----------------------------------- |
| 输出层   | `Linear(hidden, num_atoms)` 固定 | `Attention` 动态适应                |
| 候选池   | 全局固定                         | 每样本动态                          |
| 选择机制 | 分类 + filtered_softmax          | 注意力 + Gumbel-Softmax             |
| 输入     | `x_` 布尔 mask                   | `triple_embeddings` + `triple_mask` |

**修改依据**：

- 原 SELOR 的输出层维度固定，无法处理动态候选
- 指针网络通过注意力机制天然适应可变长度输入
- 保留 Gumbel-Softmax 实现可微分离散采样

---

### 7.2 `selor_utils/net.py` 修改 - 主模型

**原组件**：`AntecedentGenerator`（第 276-383 行）

**原功能**：

- 继承 `BaseModel`
- 整合 `AtomSelector` 和 `ConsequentEstimator`
- 从固定原子池中选择原子并预测

**新组件（基线）**：`AMRAntecedentGenerator`

**基线方案说明**：

- 复用原 AtomSelector 结构，输出维度 = 全局三元组词表大小 (80,497)
- 三元组嵌入通过查表获取（预计算的 triple_embedding）
- 使用 x\_（满足向量）屏蔽不属于当前样本的三元组

```python
class AMRAntecedentGenerator(nn.Module):
    """基线版AMR-SELOR前件生成器（复用GRU+mask）"""
    def __init__(self, base, hidden_dim, num_classes, antecedent_len,
                 triple_embedding, consequent_estimator, n_data):
        super().__init__()

        # BERT上下文编码器
        _, self.tf_model, _ = get_tf_model(base)

        # 预计算的三元组嵌入表 [num_triples, hidden_dim]
        # 来源：true_matrix @ train_cls_embeddings / count
        self.register_buffer('triple_embedding', triple_embedding)
        self.num_triples = triple_embedding.shape[0]

        # 原始 AtomSelector 结构（复用）
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.gru_head = nn.Linear(hidden_dim, self.num_triples)  # 输出维度=词表大小
        self.antecedent_len = antecedent_len

        # 后件估计器（冻结）
        self.consequent_estimator = consequent_estimator
        for p in self.consequent_estimator.parameters():
            p.requires_grad = False

        self.num_classes = num_classes
        self.n_data = n_data
        self.alpha = nn.Parameter(torch.ones(1))

    def filtered_softmax(self, logits, x_, temperature=1.0):
        """屏蔽不属于当前样本的三元组"""
        logits = logits.masked_fill(~x_, float('-inf'))
        if self.training:
            return F.gumbel_softmax(logits, tau=temperature, hard=True)
        else:
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, self.num_triples).float()

    def forward(self, input_ids, attention_mask, x_):
        """
        Args:
            input_ids: BERT输入 [batch, seq_len]
            attention_mask: BERT掩码 [batch, seq_len]
            x_: 满足向量 [batch, num_triples]，表示当前样本包含哪些三元组
        """
        batch_size = input_ids.shape[0]

        # 1. BERT编码句子
        bert_out = self.tf_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]  # [batch, hidden]

        # 2. GRU逐步选择三元组（复用原AtomSelector逻辑）
        selected_probs = []
        cur_input = cls_emb.unsqueeze(0)  # [1, batch, hidden]
        cur_hidden = None

        for step in range(self.antecedent_len):
            if cur_hidden is not None:
                _, cur_hidden = self.gru(cur_input, cur_hidden)
            else:
                _, cur_hidden = self.gru(cur_input)

            logits = self.gru_head(cur_hidden.squeeze(0))  # [batch, num_triples]
            prob = self.filtered_softmax(logits, x_)       # 屏蔽 + Gumbel-Softmax
            selected_probs.append(prob)

            # 下一步输入：选中的三元组嵌入
            selected_emb = torch.matmul(prob, self.triple_embedding)  # [batch, hidden]
            cur_input = (cls_emb + selected_emb).unsqueeze(0)

        select_probs = torch.stack(selected_probs, dim=1)  # [batch, antecedent_len, num_triples]

        # 3. 获取选中的三元组嵌入（查表）
        selected_embeddings = torch.matmul(select_probs, self.triple_embedding)  # [batch, antecedent_len, hidden]

        # 4. 后件估计
        mu, _, coverage = self.consequent_estimator(selected_embeddings)

        # 5. 计算最终预测概率
        n = coverage * self.n_data
        smooth = self.alpha / (n + 1e-8)
        smooth = smooth.unsqueeze(-1).expand(-1, self.num_classes)
        class_prob = (mu + smooth) / (1 + self.num_classes * smooth)

        return torch.log(class_prob + 1e-8), select_probs
```

**与原 SELOR 的关键对应**：

| 原 SELOR                      | AMR-SELOR 基线                  | 说明           |
| ----------------------------- | ------------------------------- | -------------- |
| `atom_embedding`              | `triple_embedding`              | 预计算的嵌入表 |
| `x_` (bool)                   | `x_` (bool)                     | 满足向量       |
| `gru_head(hidden, num_atoms)` | `gru_head(hidden, num_triples)` | 输出维度不同   |
| `filtered_softmax`            | `filtered_softmax`              | 逻辑相同       |

**阶段验收**：

- 训练/验证 loss 正常下降，无 NaN；梯度流检查通过
- 验证集 Macro-F1、PR-AUC 不低于 SELOR 基线过多（小幅回撤可接受）
- 抽样解释：三元组语义合理，dummy 选中比例可控

---

## 8. 阶段五：数据集与工具适配

### 8.1 `selor_utils/dataset.py` 修改

**添加新类**：`AMRDataset`

```python
class AMRDataset(Dataset):
    """AMR-SELOR专用数据集"""
    def __init__(self, df, triples_dict, tf_tokenizer, config):
        """
        Args:
            df: 原始数据DataFrame
            triples_dict: {sample_id: List[Triple]} 预解析的三元组
            tf_tokenizer: BERT tokenizer
            config: BERT config
        """
        self.df = df
        self.triples_dict = triples_dict
        self.tf_tokenizer = tf_tokenizer
        self.max_len = config.max_position_embeddings

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 获取文本和标签
        text = row['text']  # 或适配其他列名
        label = row['label']

        # BERT tokenization
        encoding = self.tf_tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 获取该样本的三元组
        triples = self.triples_dict.get(idx, [])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'triples': triples,
            'label': label
        }

def collate_fn_amr(batch):
    """自定义collate函数，处理变长三元组列表"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    triples = [item['triples'] for item in batch]  # 保持列表形式

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'triples': triples,
        'labels': labels
    }
```

**修改依据**：

- 原数据集返回 `(inputs, y)`，其中 inputs 包含 token 和原子 mask
- 新数据集需要额外返回三元组列表

**阶段验收**：

- dataloader 输出 shape 与模型输入对齐；padding/mask 正确
- 端到端小批前向无 shape error；解释生成能正常拿到三元组文本

---

### 8.2 `selor_utils/train_eval.py` 修改

**添加新函数**：`get_amr_explanation`

```python
def get_amr_explanation(model, sample_triples, select_probs, class_names):
    """生成AMR-SELOR的解释"""
    batch_size, antecedent_len, num_triples = select_probs.shape

    explanations = []
    for b in range(batch_size):
        selected_indices = torch.argmax(select_probs[b], dim=-1)
        selected_triples = [sample_triples[b][idx] for idx in selected_indices
                           if idx < len(sample_triples[b])]

        rule_str = ' AND '.join([t.display_str for t in selected_triples])
        explanations.append(rule_str)

    return explanations
```

**修改依据**：

- 解释生成逻辑需要适配新的三元组格式

---

### 8.3 `selor_utils/utils.py` 修改

**添加新参数**：

```python
def parse_arguments(return_default=False):
    parser = argparse.ArgumentParser()

    # ... 原有参数 ...

    # AMR-SELOR新增参数
    parser.add_argument('--spring_checkpoint', type=str,
                        default='./checkpoints/spring', help='SPRING模型路径')
    parser.add_argument('--triple_encoder', type=str,
                        default='bert-base-uncased', help='三元组编码器')
    parser.add_argument('--max_triples', type=int, default=50,
                        help='每个样本最大三元组数')
    parser.add_argument('--min_triple_freq', type=int, default=5,
                        help='三元组最小出现频率')
    ...
```

---

## 9. 阶段六：主流程整合

### 9.1 `amr_selor.py` [新增]

**对标文件**：`selor.py`

**核心流程**：

```python
if __name__ == "__main__":
    args = utils.parse_arguments()

    # 1. 加载预解析的三元组
    train_triples = load_triples('train')
    valid_triples = load_triples('valid')
    test_triples = load_triples('test')

    # 2. 加载三元组编码器
    triple_encoder = TripleEncoder(args.triple_encoder)

    # 3. 加载预训练的后件估计器
    ce_model = TripleConsequentEstimator(...)
    ce_model.load_state_dict(torch.load(ce_model_path))

    # 4. 创建数据集和数据加载器
    train_dataset = AMRDataset(train_df, train_triples, tf_tokenizer, config)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn_amr, ...)

    # 5. 创建AMR前件生成器
    model = AMRAntecedentGenerator(
        triple_encoder=triple_encoder,
        consequent_estimator=ce_model,
        ...
    )

    # 6. 训练
    model = train(model, train_loader, valid_loader, ...)

    # 7. 评估和生成解释
    evaluate_and_explain(model, test_loader, ...)
```

**阶段验收**：

- 全流程跑通，生成预测与解释文件；日志无异常
- 抽样对照：解释与预测标签一致性合理；性能对标基线

---

### 9.2 `run_amr_selor.py` [新增]

**对标文件**：`run_all.py`

```python
files_to_run = [
    # 阶段一：语义解析
    'spring_parse_all.py',
    'extract_triples.py',

    # 阶段二：嵌入计算
    'compute_triple_embeddings.py',

    # 阶段三：后件估计器预训练
    'sample_triple_combinations.py',
    'pretrain_ce_triple.py',

    # 阶段四：主模型训练
    'amr_selor.py'
]

for file in files_to_run:
    cmd = f'python {file} {option}'
    print(cmd)
    os.system(cmd)
```

---

## 10. 验证方案

### 10.1 单元测试

| 测试项     | 测试文件          | 验证内容                      |
| ---------- | ----------------- | ----------------------------- |
| 三元组提取 | `test_triple.py`  | PENMAN 解析、三元组提取正确性 |
| 三元组编码 | `test_encoder.py` | 编码输出维度、缓存机制        |
| 指针网络   | `test_pointer.py` | 注意力计算、Gumbel 采样       |
| 后件估计器 | `test_ce.py`      | 输入输出维度、梯度流          |

### 10.2 集成测试

```bash
# 在小规模数据子集上运行完整pipeline
python run_amr_selor.py --dataset yelp --num_samples 1000 --epochs 2
```

### 10.3 性能对比测试

| 指标          | SELOR 基线 | AMR-SELOR 目标 |
| ------------- | ---------- | -------------- |
| PR AUC (Yelp) | 97.78      | ~97.7          |
| F1 Score      | 96.26      | ~96.2          |
| 人类精度      | 46.7%      | >60%           |

### 10.4 复述鲁棒性测试（新指标）

1. 生成复述测试集
2. 计算原始/复述样本解释的 Jaccard 相似度
3. 预期 AMR-SELOR 显著高于 SELOR

### 10.5 总体指标与解释质量

- 任务性能：Macro-F1、PR-AUC；不低于基线过多为目标
- 解释质量：人类精度（好/最佳比例）、覆盖率分布；解释与标签一致性抽查
- 复述鲁棒性：解释 Jaccard 相似度；预测一致率
- 资源效率：解析/训练/推理耗时与显存占用

---

## 11. 风险与缓解

### 11.1 技术风险

| 风险            | 影响             | 缓解措施                                            |
| --------------- | ---------------- | --------------------------------------------------- |
| SPRING 解析错误 | 噪声三元组       | 过滤低置信度/罕见关系；空样本填 dummy               |
| 三元组过多      | 内存/计算开销    | top-K、max_triples 截断；稀疏 true_matrix           |
| 动态长度处理    | 批处理困难       | padding + mask；先固定上限复用 GRU+mask             |
| 后件估计器泛化  | 未见三元组效果差 | 文本编码或组合式编码；数据增强                      |
| 嵌入/索引错位   | 解释与概率不一致 | 固定三元组顺序，true_matrix 与 embedding 一致性检查 |
| 解析耗时        | 全流程时间过长   | 批量解析、缓存；必要时改用子集/并行                 |

### 11.2 实施风险

| 风险              | 影响       | 缓解措施                              |
| ----------------- | ---------- | ------------------------------------- |
| 开发周期长        | 延期       | 分阶段验收，先跑通 GRU+mask 基线      |
| 与原 SELOR 差异大 | 调试困难   | 保持模块化、充分日志，逐步替换组件    |
| 评测与解释脱节    | 结果不可信 | 统一索引/三元组文本，解释抽样人工验收 |

---

## 12. 待拓展（增强项与优先级）

| 优先级  | 增强项                                         | 价值                                      | 触发条件/时机              |
| ------- | ---------------------------------------------- | ----------------------------------------- | -------------------------- |
| P0+     | 指针网络选择器                                 | 原生支持变长候选、注意力可视化            | 基线跑通后，需提升选择质量 |
| **P0+** | **情感区分度三元组过滤**                       | **过滤功能性/无区分度三元组，提升信噪比** | **基线效果不佳时优先尝试** |
| P1      | 三元组文本编码器                               | 处理未见三元组，减少 true_matrix 依赖     | 出现大量 OOV 三元组时      |
| P1      | 复述数据增强 + 稳定性评测                      | 提升鲁棒性，解释一致性                    | 完成一次端到端训练后       |
| P1      | 组合式 head/rel/tail 嵌入                      | 参数共享，提升泛化                        | 文本编码器成本过高时       |
| P1      | 三元组组合采样 (sample_triple_combinations.py) | 扩充规则空间，提升 CE 预训练多样性        | 需更丰富前件样本时         |
| P2      | 联合训练（解除 CE 冻结）                       | 潜在提升终端性能                          | 基线稳定后，小心调 lr      |
| P2      | 图级特征（AMR 子图 pooling）                   | 捕捉全局语义                              | 数据量充足、性能瓶颈时     |

### 12.1 情感区分度三元组过滤（详细方案）

> ⚠️ **问题描述**：AMR 解析出的三元组包含大量"功能性"三元组（如 `(I, :ARG0, say-01)`），这些三元组在所有样本中高频出现，但对情感分类毫无区分度。如果不过滤，模型可能选择这些无意义的三元组作为解释。

**现有过滤（不足）**：

- `KeepRelations` 白名单：仅过滤关系类型
- `--min_freq=5`：仅过滤罕见三元组

**增强方案**：

| 方法             | 原理                         | 实现位置                    | 优先级 |
| ---------------- | ---------------------------- | --------------------------- | ------ |
| **TF-IDF 加权**  | 降低在全语料高频的三元组权重 | `build_triple_pool.py` 新增 | P0+    |
| **卡方检验**     | 保留与标签显著相关的三元组   | `build_triple_pool.py` 新增 | P0+    |
| **互信息 (PMI)** | 衡量三元组与标签的共现关系   | `build_triple_pool.py` 新增 | P1     |
| **黑名单过滤**   | 手动列出常见无意义谓词       | `extract_triples.py` 扩展   | P0     |

**卡方检验实现思路**：

```python
from scipy.stats import chi2_contingency

def filter_by_chi2(triple_indices, labels, vocab, top_k=5000, p_threshold=0.05):
    """保留与标签显著相关的三元组"""
    significant_triples = []
    for triple_idx in range(len(vocab)):
        # 构建列联表: [有三元组/无三元组] × [正面/负面]
        has_triple = [triple_idx in sample for sample in triple_indices]
        contingency = pd.crosstab(has_triple, labels)
        chi2, p_value, _, _ = chi2_contingency(contingency)
        if p_value < p_threshold:
            significant_triples.append((triple_idx, chi2))

    # 按卡方值排序，保留前 top_k
    significant_triples.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in significant_triples[:top_k]]
```

**TF-IDF 实现思路**：

```python
def compute_triple_tfidf(per_sample_indices, num_triples, n_samples):
    """计算三元组的 TF-IDF 分数"""
    # TF: 三元组在样本中出现次数 / 样本三元组总数
    # IDF: log(总样本数 / 包含该三元组的样本数)
    doc_freq = np.zeros(num_triples)
    for idxs in per_sample_indices:
        for idx in set(idxs):  # 去重
            doc_freq[idx] += 1
    idf = np.log(n_samples / (doc_freq + 1))

    # 过滤 IDF 过低的三元组（高频无区分度）
    keep_mask = idf > idf_threshold
    return keep_mask, idf
```

**建议实施顺序**：

1. 先跑通基线，观察选中的三元组质量
2. 如果发现大量无意义三元组被选中，优先加入**黑名单过滤**
3. 进一步尝试**卡方检验**筛选

---

## 13. 解释生成与论文模块对应

- 论文里的三个核心模块：Base Encoder（BERT/CLS）、Antecedent Generator（AG：选择原子/三元组）、Consequent Estimator（CE：评估规则后件）。
- 解释生成链路：
  - 输入句子 → AMR → 三元组集合 `T_x`，并生成满足向量 `x_`（或候选列表）。
  - AG（首版用 GRU+mask）在候选三元组上选择 antecedent 组合；训练阶段用 Gumbel-Softmax 硬采样确保可微。
  - CE 对选中三元组序列输出 `mu`（类别概率）、`sigma`（不确定性）、`coverage`（覆盖率）。
  - 最终预测通过平滑公式组合 `mu` 和 `coverage`，与原 SELOR 保持一致；解释即选中的三元组串接（`AND`）。
- 人类可读解释建议：
  - 用 `t.display_str` 直接展示 `(head relation tail)`，必要时映射为中文短语；
  - 展示覆盖率/置信度辅助审阅；
  - 抽样对照预测标签，人工打分“好/最佳”作为人类精度。

---

## 附录 A：文件依赖关系图

```
spring-main/
└── spring_amr/                    # SPRING解析器（已完成AMR解析）

SELOR-main/
├── data/yelp_review_polarity_csv/
│   ├── train_with_amr.csv         # [输入] 含AMR列的训练集
│   └── test_with_amr.csv          # [输入] 含AMR列的测试集
│
├── selor_amr/                     # ★ AMR-SELOR 新增代码目录 ★
│   ├── stage1/
│   │   └── extract_triples.py     # ✅ 从AMR提取三元组
│   ├── stage2/
│   │   ├── build_triple_pool.py   # ✅ 构建三元组池、true_matrix
│   │   └── triple.py              # ✅ 三元组数据结构
│   ├── stage3/
│   │   ├── extract_cls_embedding.py # ✅ 提取BERT CLS嵌入
│   │   └── pretrain_ce_triple.py  # ✅ 预训练后件估计器
│   ├── stage4/
│   │   └── amr_selor.py           # ✅ 主训练脚本
│   ├── stage5/
│   │   └── eval_amr_selor.py      # ✅ 评估与解释导出
│   ├── stage6/
│   │   ├── run_amr_selor.py       # ✅ Pipeline调度器
│   │   └── inference_amr_selor.py # ✅ 纯推理脚本
│   └── utils/
│       └── (可选增强脚本)
│
├── selor_utils/                   # 原SELOR工具库（按需修改）
│   ├── net.py                     # [改] 新增TripleConsequentEstimator ✅
│   ├── dataset.py                 # 未修改（数据集类在stage4内嵌）
│   ├── train_eval.py              # 未修改（评估逻辑在stage5内嵌）
│   └── utils.py                   # 未修改（参数解析在各阶段内嵌）
│
└── result/                        # 输出目录
    ├── triples/
    │   ├── train_triples.pkl      # ✅ 训练集三元组
    │   ├── test_triples.pkl       # ✅ 测试集三元组
    │   ├── global_triple_vocab.pkl # ✅ 三元组词表 (80,497个)
    │   ├── per_sample_indices.pkl # ✅ 样本索引
    │   └── true_matrix.npz        # ✅ 稀疏矩阵 [80497×484511]
    ├── embeddings/
    │   └── train_cls.pt           # ✅ 训练集CLS嵌入
    ├── ce_triple/
    │   ├── ce_triple_best.pt      # ✅ CE最优模型
    │   └── ce_triple_config.pkl   # ✅ CE配置
    ├── amr_selor/
    │   └── amr_selor_best.pt      # ✅ AMR-SELOR最优模型
    └── amr_selor_eval/
        ├── metrics.json           # ✅ 评估指标
        └── predictions.csv        # ✅ 预测与解释
```

---

## 附录 B：开发优先级（已完成基线）

| 优先级 | 任务                                | 状态 | 实际工时 |
| ------ | ----------------------------------- | ---- | -------- |
| P0     | 三元组提取/池构建（含 true_matrix） | ✅   | 2 天     |
| P0     | TripleConsequentEstimator 预训练    | ✅   | 2 天     |
| P0     | GRU+mask 版前件生成器适配           | ✅   | 1.5 天   |
| P0     | 主流程整合与小规模集成测试          | ✅   | 2 天     |
| P0     | 评估与解释导出                      | ✅   | 0.5 天   |
| P1     | **情感区分度三元组过滤**            | ⏳   | 待实施   |
| P1     | **三元组文本编码器**                | ⏳   | 待实施   |
| P1     | 指针网络选择器（可选增强）          | —    | 待评估   |
| P2     | 复述鲁棒性评测与数据增强            | —    | 待评估   |
| P2     | 性能调优                            | —    | 待评估   |

---

_文档版本：v1.2_  
_创建日期：2025-12-10_  
_基线完成日期：2025-12-15_  
_作者：研究团队_
