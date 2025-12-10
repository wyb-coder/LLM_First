# AMR-SELOR: 一种用于生成语义逻辑解释的神经符号框架



**作者：** 王耀彬

**单位：** 北方工业大学人工智能与计算机学院，北京市石景山区晋元庄路5号，100144

\author{Yaobin Wang}

\affiliation{School of Artificial Intelligence and Computer Science, North China University of Technology (NCUT), Beijing 100144, China}

------



## 摘要



随着深度学习在关键决策领域的广泛应用，如何为其决策过程提供忠实且可理解的解释成为构建可信人工智能的核心挑战。现有自解释模型(如基于逻辑规则推理的 SELOR)在保证解释忠实性方面取得进展，但其依赖词汇或统计特征的原子，导致逻辑规则语义浅显，难以刻画复杂因果关系，且易受句法变化干扰。为弥合这一“语义鸿沟”，本文提出 AMR-SELOR 框架，将抽象语义表示(AMR)的深层语义结构与 SELOR 的逻辑推理能力结合，实现从“词法匹配”到“语义推理”的转变。具体而言，框架以 AMR 图中提取的语义三元组替代原子库，从而提升解释与人类认知模型的对齐度，并增强对文本复述的鲁棒性。本文进一步阐述 AMR-SELOR 的理论架构与关键实现技术(包括 SPRING 等先进 AMR 解析器)，并设计包含“复述鲁棒性”测试的综合验证方案，以探索更深刻、鲁棒且符合人类直觉的 AI 可解释性路径。



With the increasing deployment of deep learning in critical decision-making domains, providing faithful and human-understandable explanations has become a central challenge for trustworthy artificial intelligence (AI). Existing self-explanatory models, such as SELOR, achieve explanation fidelity by embedding logical reasoning into model structures. However, their reliance on lexical or statistical atoms yields semantically shallow rules, limiting the capture of complex causal relations and making them vulnerable to syntactic variations. To bridge this “semantic gap,” we propose AMR-SELOR, a neurosymbolic framework that integrates the deep semantic structures of Abstract Meaning Representation (AMR) with SELOR’s logical reasoning. Specifically, semantic triples extracted from AMR graphs replace the original atom library, enabling closer alignment with human cognitive models and improved robustness to paraphrasing. We present the theoretical architecture and key implementation techniques of AMR-SELOR, including the use of advanced AMR parsers such as SPRING, and design a comprehensive evaluation scheme incorporating a novel “paraphrasing robustness” test. This work aims to advance AI interpretability toward deeper, more robust, and cognitively aligned explanations.



**关键词：** 可解释人工智能(XAI)，自解释模型，神经符号推理，抽象语义表示(AMR)，逻辑规则推理

**中国法分类号：** TP393

**DOI号：** 10.1234/cjc.2025.000001

------



## 1 介绍 (Introduction)



在人工智能技术深度融入社会生产与生活的今天，深度学习模型已在自然语言处理、医疗诊断、金融风控等高风险领域展现出卓越的性能。然而，这些模型复杂的内部结构和非线性的计算过程使其决策逻辑往往不透明，形成了所谓的“黑箱”问题。这种不透明性不仅阻碍了用户对模型决策的信任，也为模型的调试、纠错和确保其公平性、合乎伦理带来了巨大挑战 。因此，发展可解释人工智能(Explainable AI, XAI)技术，使模型的决策过程透明化、可理解，已成为AI领域一个紧迫且至关重要的研究方向。

XAI领域的研究大致可分为两大范式：事后解释(post-hoc explanation)与自解释(self-explaining)模型。事后解释方法，如LIME  和SHAP ，在模型训练完成后，通过分析其输入输出关系来推断决策依据。尽管这类方法具有模型无关的灵活性，但其解释与模型的实际推理过程相分离，可能存在“不忠实”(unfaithful)的风险，即生成的解释无法真实反映模型的内在逻辑。这种潜在的不一致性导致了实践者中普遍存在的“不安感”和信任问题 1。

相比之下，自解释模型将解释机制作为其架构不可或缺的一部分，强制模型在预测的同时生成一个人类可理解的解释。SELOR(Self-explaining deep models with logic rule reasoning)是这一范式的杰出代表。

SELOR框架通过一个严谨的概率公式 $p(y|x,b) \propto \sum_{\alpha}p(b|\alpha)p(y|\alpha)p(\alpha|x)$，将预测过程重构为寻找并评估逻辑规则的过程，从根本上保证了解释的忠实性。更重要的是，SELOR引入了“人类精度”(Human Precision)这一核心概念，将其定义为“人类对模型为其预测所提供理由的认同程度”，并将解释质量的评估标准从机器中心转向了人类中心 1。

然而，尽管SELOR在架构上取得了突破，其推理的基础——逻辑规则的“原子”(atom)——仍然存在固有的局限性。SELOR的原子通常是基于词汇或统计特征的布尔表达式，例如“评论中‘awesome’一词出现次数 $\ge 1$”或“用户年龄 $\ge 40$” 。这类原子本质上是浅层的、基于表面形式的，它们能够捕捉特征与标签之间的统计相关性，但往往无法触及更深层次的语义因果关系。SELOR论文中一个极具启发性的案例研究表明，模型可能仅仅因为“vegas”一词在数据集中与负面评论存在虚假相关，就生成了包含该词的解释规则 。这对人类来说显然是不合理、不可信的。这种“语义鸿沟”的存在，即模型推理单元与人类认知单元之间的不匹配，是限制当前自解释模型达到更高层次可解释性的关键瓶颈。





通过这一根本性的变革，我们旨在将模型的推理基础从“句子中出现了什么词”提升到“句子表达了什么含义”。我们预期AMR-SELOR将带来以下主要贡献：

1. **提出AMR-SELOR框架**：一个新颖的、深度集成的神经符号架构，首次将AMR的深层语义分析能力与SELOR的忠实逻辑推理机制相结合。
2. **重构逻辑解释单元**：将解释的“原子”从词法特征重新定义为语义三元组，使生成的逻辑规则在本质上更贴近人类的因果认知模型。
3. **规划技术实现路径**：提供一套完整的、具有可操作性的技术实现方案，涵盖了从利用先进AMR解析器(如SPRING)进行语义解析，到改造SELOR核心模块(后件估计器与前件生成器)的全过程。
4. **设计全面的验证策略**：提出一套严谨的实验验证方案，不仅包括与原SELOR在各项指标上的直接对比，还引入了一项新颖的“复述鲁棒性”测试，以量化评估模型在语义层面上的稳定性。

本报告的后续章节将围绕以上贡献展开。第二章将回顾相关领域的研究工作。第三章将详细阐述AMR-SELOR框架的理论设计与核心优势。第四章将深入探讨关键的实现技术。第五章将提出具体的实验验证方案。最后，第六章将对全文进行总结，并展望未来的研究方向。





{\begin{CJK*}{UTF8}{zhhei}\subsubsection{三级标题 *字体为5号宋体*标题3}\end{CJK*}}





## 2 相关工作 (Related Work)



本研究位于可解释人工智能(XAI)、计算语言学和神经符号AI三个领域的交叉点。本章将分别对这些领域中的相关工作进行综述，以明确AMR-SELOR框架的学术定位和创新性。



### 2.1 可解释AI范式：从事后解释到自解释模型



可解释AI的研究旨在打开深度学习的“黑箱”，其发展历程中形成了两大主流技术范式。

**事后解释(Post-hoc Explanation)**：这是早期XAI研究的焦点。这类方法作用于一个已训练好的模型，试图通过外部扰动或内部探查来解释其行为。代表性工作包括LIME(Local Interpretable Model-agnostic Explanations) 和Anchor 。LIME通过在预测实例的局部邻域内用一个简单的、可解释的模型(如线性模型)来近似复杂模型的行为。Anchor则旨在寻找一个“锚点”，即一个能充分固定预测结果的输入特征子集。尽管这些方法因其模型无关性而应用广泛，但它们的核心缺陷在于解释与模型决策过程的分离，导致了解释的“忠实性”问题备受质疑 。研究者指出，事后解释可能被恶意攻击，产生误导性的结果，或无法捕捉到特征间的复杂交互，从而不能真实反映模型的内在逻辑。

**自解释模型(Self-explaining Models)**：为了从根本上解决忠实性问题，研究重心逐渐转向自解释模型。这类模型将解释的生成过程作为其体系结构的一个内在组成部分。SENN(Self-Explaining Neural Networks) 是一个早期的代表，它将模型分解为对可解释基概念的线性组合，从而生成基于特征重要性的解释。然而，SENN的解释形式较为简单，表达能力有限。SELOR框架则在SENN的基础上迈出了重要一步，它将解释形式从线性权重提升为更具表达力的逻辑规则 1。通过强制模型经由一条全局一致且局部连贯的逻辑规则进行预测，SELOR不仅保证了忠实性，还显著提升了解释的“人类精度”。本研究正是建立在SELOR的坚实基础之上，旨在通过引入深层语义信息，进一步提升其逻辑规则的质量和可理解性。



### 2.2 抽象语义表示(AMR)



**基础理论与形式化**：抽象语义表示(AMR)由Banarescu等人于2013年正式提出，旨在为自然语言句子提供一种规范化的、捕捉核心语义的图表示 2。AMR的核心设计原则是抽象掉表层句法结构，使得意义相同但表述方式不同的句子能够映射到同一个AMR图上 2。一个AMR图是一个有根、有向的无环图，其中节点代表概念(通常是词语的词义或PropBank中的谓词-论元框架)，边则代表它们之间的语义关系(如`:ARG0`代表施事，`:ARG1`代表受事，`:location`代表地点等)5。AMR通常使用PENMAN格式进行文本序列化，例如，对于句子“The boy wants to go”，其PENMAN表示为 `(w / want-01 :arg0 (b / boy) :arg1 (g / go-01 :arg0 b))` 5。这种结构化的表示方法为进行深度的语义分析和推理提供了可能。

**先进的解析技术**：将自然语言文本自动转换成AMR图的过程称为AMR解析(parsing)。近年来，随着预训练语言模型的发展，AMR解析的性能取得了长足的进步。其中，由罗马大学Sapienza NLP实验室开发的**SPRING**框架是当前最先进的AMR解析器之一 8。SPRING将AMR解析和生成任务统一视为序列到序列(seq2seq)的转换问题，通过巧妙的图线性化技术，利用强大的Transformer架构实现了端到端的解析，取得了顶尖的性能，且无需复杂的预处理流水线 9。SPRING的开源和高性能为本研究将AMR集成到SELOR框架中提供了坚实的技术基础。

**应用领域**：AMR作为一种强大的语义中间表示，已被成功应用于多种下游NLP任务，如机器翻译、文本摘要、问答系统和信息抽取，证明了其在捕捉和利用句子核心语义方面的有效性 11。



### 2.3 神经符号AI前沿



AMR-SELOR的构想本质上是一种神经符号(Neuro-Symbolic, NeSy)AI方法。NeSy旨在融合神经网络强大的模式识别、泛化能力与符号系统明确的知识表示、逻辑推理能力，以期构建出既能学习又能推理的更强大、更鲁棒的AI系统 14。这一研究方向与认知科学中的双过程理论相呼应，将神经网络类比为快速、直觉的“系统1”，而将符号推理类比为缓慢、审慎的“系统2” 15。

美国国防部高级研究计划局(DARPA)的ANSR(Assured Neuro Symbolic Learning and Reasoning)等项目的大力投入，也反映了学术界和工业界对NeSy作为构建可信AI关键路径的共识 16。近年来，已有研究开始探索将AMR作为符号知识整合到神经模型中。例如，有工作将AMR解析与指代消解等符号模块结合，以提升语言理解的准确性和鲁棒性 17。这些工作为本研究提供了重要的思想借鉴，即AMR可以作为连接神经网络表示与符号逻辑推理的有效桥梁。



### 2.4 可解释性中的语义表示



将语义信息用于提升AI可解释性的探索尚处于起步阶段，但已展现出巨大潜力。近期的一项代表性工作是AMREx，一个用于可解释事实核查的系统 19。AMREx利用AMR图之间的相似度(通过Smatch度量)来判断一个声明与证据之间是否存在语义蕴含或矛盾关系，并能通过对齐的AMR节点来提供部分解释。AMREx的成功表明，AMR能够为解释任务提供有价值的语义 grounding。

然而，AMREx等现有方法通常以流水线或事后分析的方式使用AMR，AMR分析的结果作为外部信息提供给下游的分类或解释模块。这与AMR-SELOR的核心理念存在本质区别。AMR-SELOR旨在将语义推理*内化*为模型决策过程的核心环节，模型的预测直接依赖于其能否找到一条基于AMR三元组的、具有高置信度的逻辑规则。这种深度、原生的集成方式，是对现有工作的进一步发展，有望实现一种更彻底、更忠实的语义可解释性。

综上所述，AMR-SELOR的提出并非凭空而来，而是站在了多个前沿研究领域的交汇点上。它继承了自解释模型对“忠实性”的追求，利用了AMR解析技术的最新成果，响应了神经符号AI对“融合”的呼唤，并深化了语义表示在可解释性领域的应用。通过将这三个领域的思想进行有机融合，AMR-SELOR有望在构建下一代高可信、高智能的AI系统方面做出独特的贡献。



## 3 AMR-SELOR框架：主要成果论述



本章将详细阐述AMR-SELOR框架的核心设计理念、系统架构，并论证其相较于原版SELOR在可解释性质量上的潜在优势。本框架的核心贡献在于，它通过引入抽象语义表示(AMR)，从根本上重塑了自解释模型的推理基石，推动解释生成从浅层词法匹配向深层语义推理的深刻转变。



### 3.1 概念架构



AMR-SELOR的整体架构继承了SELOR的概率推理框架，但对其核心组件进行了根本性的改造，以无缝集成AMR的语义信息。整个信息处理流程可分解为以下五个关键步骤：

1. **并行语义与上下文编码**：当一个输入文本 $x$ 进入系统时，它被并行送入两个模块：
   - **上下文编码器**：与SELOR一样，一个预训练的深度模型(如BERT或RoBERTa)将文本 $x$ 编码为一个高维的上下文向量 $z$。这个向量捕捉了文本的深层句法和语境信息。
   - **语义解析器**：一个先进的AMR解析器(如SPRING)将文本 $x$ 解析为其对应的AMR图 $G$。这个图显式地表示了文本的核心语义结构。
2. **实例级原子库提取**：一个新增的“原子提取器”(Atom Extractor)模块负责遍历AMR图 $G$，并提取其中所有的语义三元组。这些三元组构成了该输入实例 $x$ 专属的、动态的候选原子库 $A_x = \{t_1, t_2,..., t_n\}$。例如，对于句子“The staff was amazing”，解析后可能提取出三元组 `(staff, :domain-of, amazing-01)`。
3. **语义前件生成**：改造后的“深度前件生成器”(Deep Antecedent Generator)接收上下文向量 $z$ 和实例级原子库 $A_x$ 作为输入。它的任务不再是从一个全局固定的、庞大的词汇库中进行搜索，而是学习如何从这个小而精的、与当前输入高度相关的语义三元组集合 $A_x$ 中，选择出最具解释力的三元组，并组合成一条逻辑规则(前件)$\alpha$。
4. **全局后件评估**：与SELOR类似，“后件估计器”(Consequent Estimator)模块负责评估生成的规则 $\alpha$ 的全局置信度 $p(y|\alpha)$。这个模块预先在整个训练数据集上进行了训练，学习了不同语义规则组合与最终预测标签 $y$ 之间的全局统计关联。
5. **忠实预测生成**：模型的最终预测输出是基于规则置信度的函数，完全遵循SELOR的“无作弊”原则。模型必须找到一条能够被全局验证的、高质量的语义规则，才能对预测结果产生高置信度。

这一架构的精妙之处在于，它将复杂的语义理解任务(由AMR解析器完成)与忠实的逻辑推理任务(由SELOR的核心框架完成)进行了解耦和串联。AMR解析器扮演了一个强大的“语义预处理器”或“结构化注意力机制”的角色，它极大地缩小了前件生成器的搜索空间，使其能够专注于在高度相关的语义单元上进行推理，而不是在海量的词汇海洋中进行盲目探索。



### 3.2 语义原子：从词素到三元组



AMR-SELOR最核心的创新在于对“原子”这一基本解释单元的重新定义。下表清晰地对比了标准SELOR原子与我们提出的AMR-SELOR语义原子的本质区别：

| **特征**     | **标准SELOR原子**   | **提出的AMR-SELOR原子**        |
| :----------- | ------------------- | :----------------------------- |
| **基本单元** | 词汇/特征的存在性   | 语义关系(三元组)               |
| **示例**     | `"amazing" >= 1`    | `(staff, :ARG0-of, praise-01)` |
| **来源**     | 全局词汇表/特征集   | 输入句子的AMR图                |
| **含义**     | “存在单词‘amazing’” | “员工是‘表扬’的施事者”         |
| **抽象层次** | 词法/句法           | 语义                           |

在SELOR中，一个原子是一个作用于输入特征的布尔函数，例如 $o_i(x) = (\text{词频}("tasty") \ge 1)$。它回答的问题是“某个表层特征是否存在？”。而在AMR-SELOR中，一个语义原子(即语义三元组)是一个作用于输入文本语义图的布尔函数，例如 $t_j(x) = ((staff, :ARG0\text{-of}, praise\text{-}01) \in \text{Triples}(\text{Parse}(x)))$。它回答的问题是“某个深层语义关系是否存在？”。

这种转变意义重大。它使得模型的解释语言从一种机器易于处理但人类难以直观理解的“相关性语言”，转变为一种更接近人类认知模型的“因果性语言”。



### 3.3 假设优势



我们预测，这种从词法到语义的根本性转变将为模型带来三个关键优势：



#### 3.3.1 卓越的人类精度



SELOR框架的核心目标是提升“人类精度” 1。我们认为，AMR-SELOR将在此指标上取得质的飞跃。人类在理解和解释语言时，依赖的是对语义角色和事件结构的认知。例如，当被问及为何一条评论是负面时，一个令人信服的理由是“因为评论者抱怨价格过高”，而不是“因为评论中出现了‘price’和‘high’这两个词”。AMR-SELOR生成的解释，如 `(price, :manner, rapacious-01) \Rightarrow \text{负面情绪}`，直接对应了前一种人类认知模式。相比之下，标准SELOR可能会生成的解释，如 ("vegas" $\ge$ 1) \Rightarrow \text{负面情绪}，则仅仅反映了数据中的统计噪声。通过将解释的语言与人类的思维语言对齐，AMR-SELOR有望生成真正符合人类直觉、具有说服力的理由，从而大幅提升人类精度。



#### 3.3.2 增强的复述鲁棒性



这是AMR-SELOR带来的一个独特且关键的优势。AMR的核心设计哲学之一就是对句法结构的不变性，即语义相同的句子，无论其句法结构如何(如主动语态 vs. 被动语态、名词化结构 vs. 动词结构)，都应被解析为同一个AMR图 2。这一特性天然地赋予了AMR-SELOR对文本复述(paraphrasing)的鲁棒性。

例如，考虑以下两个语义等价的句子：

- $x_1$: "The staff was amazing and impressed us."
- $x_2$: "We were impressed by the amazing staff."

标准SELOR可能会为这两个句子生成不同的解释规则。对于 $x_1$，规则可能是 `("staff" >= 1) AND ("amazing" >= 1)`；对于 $x_2$，规则可能是 `("impressed" >= 1) AND ("staff" >= 1)`。尽管都是合理的，但解释的不一致性可能会让用户感到困惑。

而AMR-SELOR则有望为这两个句子生成完全相同的解释。因为它们的AMR图都会包含类似 `(staff, :domain-of, amazing-01)` 和 `(staff, :ARG0-of, impress-01)` 这样的核心语义三元组。因此，生成的规则可能是 `(staff, :domain-of, amazing-01) \Rightarrow \text{正面情绪}`。这种在语义层面上的稳定性，是纯词法模型难以企及的，它代表了一种更高级、更本质的鲁棒性。



#### 3.3.3 更强的表达能力



AMR的表示体系能够精细地刻画多种复杂的语言现象，这极大地扩展了AMR-SELOR规则的表达能力 2。

- **否定(Negation)**：AMR使用 `:polarity -` 来表示否定。这使得模型可以学习到如 `(service, :polarity, -) \Rightarrow \text{负面情绪}` 这样的规则，直接捕捉到“服务不好”的核心语义，而不是依赖于“not”、“bad”等否定词的组合。
- **情态(Modality)**：AMR可以表示情态动词，允许模型区分事实与可能性，例如，区分“The service is good”和“The service could be good”。
- **共指(Co-reference)**：AMR通过变量复用显式地处理共指关系。这使得模型能够构建跨越多个子句的复杂推理链。例如，对于“The waiter was rude, because he ignored us”，模型可以生成规则 `(waiter, :instance, he) AND (he, :ARG0-of, ignore-01) \Rightarrow \text{负面情绪}`，准确地将“粗鲁”的行为归因于“服务员”。

综上所述，AMR-SELOR框架不仅是对SELOR的一个简单升级，更是一次深刻的范式革新。通过将坚实的语义理论基础注入到忠实的自解释架构中，它有望生成更精确、更鲁棒、更具洞察力的解释，从而在通往真正可信赖AI的道路上迈出坚实的一步。



## 4 关键实现技术



要将AMR-SELOR从概念框架转化为可运行的系统，需要在SELOR原有技术栈的基础上，集成先进的AMR处理能力并对核心模块进行深度改造。本章将详细阐述实现AMR-SELOR所需的三大关键技术阶段，并提出具体的技术方案。我们强调，遵循SELOR原有的分阶段训练策略——即先预训练后件估计器，再训练前件生成器——对于管理我们这个更复杂系统的训练过程至关重要。这种解耦设计模式能够创建一个稳定、可靠的“语义神谕”(semantic oracle)，从而极大地简化前件生成器的学习任务，是项目成功的关键 1。



### 4.1 阶段一：语义原子库生成



此阶段的目标是为每个输入文本动态地生成一个高质量的、由语义三元组构成的候选原子库。



#### 4.1.1 文本到AMR的解析



这是整个流程的入口。我们将采用一个预训练好的、性能顶尖的AMR解析器。**SPRING** 8 是理想的选择，因为它基于强大的Transformer架构，提供端到端的解析能力，并且有公开可用的模型 8。

- **输入**：一个自然语言句子。
- **输出**：该句子的线性化AMR图，通常采用PENMAN表示法 23。
- **挑战与考量**：解析器的性能是整个系统的瓶颈。解析错误(“Garbage in, garbage out”)会直接向后续模块引入噪声。因此，选择和评估最先进的解析器是首要任务。



#### 4.1.2 AMR图到三元组的提取算法



获得PENMAN格式的输出后，需要一个算法将其转换为结构化的三元组集合。

1. **解析PENMAN字符串**：使用专门的库，如Python的 `penman` 库 7，将线性的PENMAN字符串解析成一个图数据结构(如邻接表或对象集合)。该库能够正确处理节点变量、概念、关系以及图的重入(re-entrancies)等复杂结构。
2. **遍历图结构**：遍历图中的所有边，提取出形如 `(source_node_variable, :relation, target_node_variable)` 的基本关系。
3. **概念实例化**：将节点变量替换为其对应的概念，形成最终的语义三元组 `(source_concept, :relation, target_concept)` 26。例如，从 `(p / praise-01 :ARG0 (s / staff))` 中，可以提取出三元组 `(praise-01, :ARG0, staff)`。
4. **过滤与规范化策略**：原始AMR图可能包含大量三元组，其中一些可能对解释任务的贡献较小。为了控制原子库的规模和质量，可以引入过滤策略：
   - **核心角色优先**：优先保留核心语义角色，如 `:ARG0`, `:ARG1`, `:ARG2` 等，它们通常承载了句子的主干信息。
   - **剔除泛化节点**：可以考虑过滤掉一些过于泛化的三元组，例如那些涉及未解析的代词(如 `(he, :ARG0-of,...)`)且没有共指信息的。
   - **关系方向规范化**：AMR包含逆关系(如 `:ARG0-of`)。为了减少原子库的冗余，可以将所有逆关系都转换为其正向等价形式，例如将 `(staff, :ARG0-of, praise-01)` 规范化为 `(praise-01, :ARG0, staff)`。



### 4.2 阶段二：后件估计器适配



后件估计器 $p(y|\alpha)$ 是SELOR的“全局真理仲裁者” 1。在AMR-SELOR中，它必须学会评估由语义三元组构成的规则的全局置信度。



#### 4.2.1 应对可组合性爆炸的挑战



标准SELOR的后件估计器面对的是一个由词汇原子构成的、虽然巨大但有限的组合空间 。而由AMR三元组构成的规则空间是动态的、组合性更强的，直接枚举和学习所有可能的规则是不可行的。



#### 4.2.2 组合式神经估计方案



我们提出一种**组合式**的后件估计器。该估计器不再将每个三元组视为一个独立的、不透明的符号，而是利用其内部结构。

- **输入表示**：一条规则 $\alpha$ 由一个或多个三元组构成。每个三元组 `(head, relation, tail)` 将被表示为其三个组成部分嵌入向量的拼接或组合。这些嵌入可以从预训练语言模型或专门的图嵌入模型中获得。
- **网络架构**：一个能够处理集合或序列输入的神经网络，如Transformer编码器或Deep Sets网络，将接收规则中所有三元组的嵌入表示。该网络将学习这些语义单元之间的交互，并最终输出对整个规则的全局置信度 $p(y|\alpha)$ 和覆盖度 $n_{\alpha}$ 的预测。这种组合式方法使得模型能够泛化到在训练中从未见过的规则组合。



#### 4.2.3 预训练语料库的生成



为了训练这个组合式估计器，我们需要一个大规模的`(规则, 置信度)` 数据集。这可以通过一个大规模的离线计算过程来生成，该过程严格遵循SELOR的原始策略 ：

1. **全局解析**：将整个训练数据集(例如，数万条Yelp评论)全部通过AMR解析器进行处理，得到每个样本的AMR图和三元组集合。
2. **规则采样**：从所有样本的三元组集合中，随机采样数百万条候选规则 $\alpha$(可以是一元、二元或更多元的三元组组合)。
3. **经验置信度计算**：对于每一条采样的规则 $\alpha$，遍历整个训练数据集，计算其经验后验概率 $\hat{p}(y|\alpha) = n_{\alpha,y} / n_{\alpha}$ 和覆盖度 $n_{\alpha}$。
4. **模型训练**：使用这个庞大的`(规则, 经验置信度, 经验覆盖度)` 数据集，来训练组合式后件估计器网络。损失函数可以沿用SELOR中的多任务损失函数 $\mathcal{L}_{c}$ [, Eq. (6)]，该函数能同时优化概率和覆盖度的预测，并根据不确定性进行加权。



### 4.3 阶段三：深度前件生成器修改



深度前件生成器 $p(\alpha|x)$ 是SELOR中与输入直接交互的模块，负责为特定输入找到最佳解释规则。



#### 4.3.1 适应动态候选原子



标准SELOR的前件生成器通常有一个固定大小的输出层，其维度对应于全局原子库的大小 1。然而，在AMR-SELOR中，候选原子库 $A_x$ 是动态的，随每个输入 $x$ 而变化。



#### 4.3.2 指针网络架构方案



为了解决这个问题，我们建议采用**指针网络(Pointer Network)**或类似的注意力机制。

- **架构**：该生成器以输入的上下文嵌入 $z$ 作为初始状态。在一个递归的过程中，它逐步生成解释规则。在每一步，它不是从一个固定的词汇表中选择，而是计算一个在当前实例的原子库 $A_x$ 上的注意力分布，然后“指向”并选择一个三元组作为规则的下一个组成部分。
- **优势**：这种架构天然地适应了可变大小的输入(候选原子库)，并强制模型只能从当前文本的语义图中提取解释，进一步加强了解释的“扎根性”(groundedness)。



#### 4.3.3 可微采样与端到端训练



为了实现端到端的训练，离散的选择过程必须是可微的。我们将保留SELOR中使用的关键技术——**直通Gumbel-Softmax(Straight-Through Gumbel-Softmax)** , Eq. (8)]。Gumbel-Softmax将被应用于指针网络的注意力分布之上，从而在允许梯度反向传播的同时，能够对离散的语义三元组进行采样。最终，整个前件生成器将通过优化交叉熵损失进行训练，其目标是最大化由预训练好的、固定的后件估计器给出的奖励，即找到能为正确标签 $y^*$ 导出最高置信度的语义规则。



## 5 验证 (Validation)



为了系统地评估AMR-SELOR框架的有效性，并验证其相较于基线模型的理论优势，我们设计了一套包含定量、定性和人类中心评估的综合验证方案。该方案旨在直接回应我们在第三章中提出的核心假设：AMR-SELOR能够在不牺牲预测性能的前提下，显著提升解释的语义质量、人类认同度以及对语言变化的鲁棒性。



### 5.1 实验设置





#### 5.1.1 数据集



为了确保与原始SELOR研究的可比性，我们将采用其使用的三个标准数据集 ：

- **Yelp Polarity**：一个大规模的文本情感分类数据集，用于评估模型在处理自然语言评论方面的能力。
- **Clickbait News Detection**：一个新闻标题分类数据集，用于测试模型在更具挑战性和微妙性的文本上的表现。
- **Adult Income**：一个经典的表格数据集，用于预测个人年收入是否超过5万美元。对于此数据集，由于其非文本性质，我们将设计一种方法将表格特征转换为“伪AMR”三元组(例如，对于一条记录，生成三元组 `(record, :has-age, <28)` 和 `(record, :has-education, PhD)`)。这使得我们可以在统一的AMR-SELOR架构下测试其在结构化数据上的推理能力，并与SELOR进行公平比较。



#### 5.1.2 基线模型



我们将设置以下三个模型进行全面对比：

1. **基础模型(Base Model)**：不带任何解释模块的、强大的预训练语言模型(如RoBERTa)或深度神经网络(DNN)，作为各项任务的性能上限参考。
2. **SELOR**：原始的SELOR框架。我们将严格按照其论文中的描述进行复现，作为衡量我们改进效果的直接基线。
3. **AMR-SELOR**：我们提出的新框架。



### 5.2 定量指标





#### 5.2.1 任务预测性能



首先，必须验证引入复杂的语义解释机制是否会对模型本身的预测能力造成损害。我们将采用与SELOR论文相同的评估指标 ：

- **PR AUC (Precision-Recall Area Under Curve)**：对于像Adult和Clickbait这样的不平衡数据集，PR AUC比ROC AUC更能反映模型的真实性能。

- F1-Score：综合评估模型的精确率和召回率。

  我们的目标是证明AMR-SELOR的预测性能与SELOR及基础模型相当(on-par)。



#### 5.2.2 对标签噪声的鲁棒性



SELOR的一个显著优势是其对训练数据中标签噪声的鲁棒性，这源于其后件估计器的全局验证机制 。我们将复现这一实验，通过在训练集中随机翻转5%到20%的标签来引入对称噪声。我们假设，由于语义模式比词汇模式在整个数据集中更具全局一致性，AMR-SELOR可能会表现出比原始SELOR更强的鲁棒性，能更有效地忽略由个别噪声样本引起的干扰。



#### 5.2.3 新颖测试：对文本复述的稳定性



这是我们为验证AMR-SELOR核心优势而专门设计的一项新颖测试。

- **测试集构建**：我们将使用一个先进的释义生成模型(例如，基于T5或BART微调的模型)来处理Yelp和Clickbait的测试集，为每个样本生成一个或多个语义等价但词汇和句法结构不同的复述版本。
- **评估方法**：对于每一个“原始-复述”对，我们将分别输入到SELOR和AMR-SELOR中，获取它们生成的解释规则(原子集合)。然后，我们计算这两个规则集合之间的**Jaccard相似度**。
- **核心假设**：我们预测AMR-SELOR在该指标上的得分将显著高于SELOR。高分意味着模型能够“看穿”表面的语言变化，为语义相同的内容提供稳定、一致的解释，这直接证明了其语义抽象能力的优越性。



### 5.3 定性与人类中心指标





#### 5.3.1 人类精度研究



我们将严格复制SELOR论文中的人类精度用户研究 ，这是评估解释质量的黄金标准。

- **研究设计**：招募一批人类评估员(如母语为英语的标注员)，向他们展示来自Yelp和Adult数据集的样本。对于每个样本，同时呈现其真实标签、模型的预测结果，以及由SELOR和AMR-SELOR生成的解释。
- **标注任务**：评估员需要对每个解释进行两项标注：(1)该解释是否是一个“好”(good)的理由？(2)在两个模型的解释中，哪一个是“最佳”(best)的理由？
- **核心假设**：我们的中心假设是，AMR-SELOR生成的解释将在“好”和“最佳”的比例上都取得统计上显著的优于SELOR的结果，从而直接验证其在对齐人类认知方面的进步。



#### 5.3.2 案例研究



为了更直观地展示AMR-SELOR的优势，我们将精心挑选一些具有挑战性的案例进行深入分析。例如：

- **包含讽刺或复杂情感的句子**：分析SELOR是否会抓住误导性的关键词，而AMR-SELOR是否能通过解析深层语义结构(如情感冲突)给出更准确的解释。
- **具有复杂句法结构的句子**：对比两个模型在处理被动语态、从句、名词化短语等情况时的解释差异，突显AMR-SELOR的句法稳健性。
- **与“vegas”类似的虚假相关案例**：检验AMR-SELOR是否能有效避免SELOR所暴露出的、学习到无厘头语义关联的问题。



### 5.4 提出的对比表格



为了清晰地总结和呈现预期的实验结果，我们设计了下表。该表将作为我们验证工作的最终成果汇总，直观地比较各模型在关键维度上的表现。

**表1：AMR-SELOR的综合性能对比分析**

| **数据集**    | **模型**           | **PR AUC**    | **F1分数**    | **人类精度 (最佳比例 %)** | **复述稳定性 (Jaccard)** |
| ------------- | ------------------ | ------------- | ------------- | ------------------------- | ------------------------ |
| **Yelp**      | 基础模型 (RoBERTa) | 97.90         | 97.16         | N/A                       | N/A                      |
|               | SELOR              | 97.78         | 96.26         | 46.7%                     | (基线得分)               |
|               | **AMR-SELOR**      | (预期: ~97.7) | (预期: ~96.2) | (假设: >60%)              | (假设: 显著更高)         |
| **Clickbait** | 基础模型 (RoBERTa) | 63.72         | 74.25         | N/A                       | N/A                      |
|               | SELOR              | 64.14         | 74.20         | (未测试)                  | (基线得分)               |
|               | **AMR-SELOR**      | (预期: ~64.0) | (预期: ~74.0) | (假设: 高)                | (假设: 显著更高)         |
| **Adult**     | 基础模型 (DNN)     | 68.62         | 76.15         | N/A                       | N/A                      |
|               | SELOR              | 70.36         | 77.37         | 65.1%                     | N/A                      |
|               | **AMR-SELOR**      | (预期: ~70.0) | (预期: ~77.0) | (假设: >75%)              | N/A                      |

通过上述多维度、系统化的验证方案，我们旨在为AMR-SELOR框架的优越性提供坚实、全面的经验证据。



## 6 结论



本报告深入探讨并系统性地提出了一个名为AMR-SELOR的新型自解释AI框架。该框架的核心创新在于，通过将抽象语义表示(AMR)的深层语义分析能力与SELOR框架的忠实逻辑推理机制进行深度融合，旨在解决当前自解释模型普遍存在的“语义鸿沟”问题。我们详细阐述了AMR-SELOR的理论基础、系统架构、关键实现技术以及一套全面的验证方案，旨在推动AI可解释性从基于表层特征的统计归纳，向基于深层语义的认知推理迈进。

**框架回顾与潜在影响**：AMR-SELOR用从AMR图中提取的语义三元组替换了SELOR原有的词汇原子，将解释的基本单元从“词”提升到了“义”。这一根本性的转变有望带来多重收益：通过生成与人类认知模型更为一致的、基于语义角色的解释，显著提升“人类精度”；利用AMR对句法结构的抽象能力，获得对文本复述的强大鲁棒性；借助AMR丰富的表示体系，增强解释规则的表达能力，以捕捉否定、情态等复杂语言现象。如果得到成功验证，AMR-SELOR将为构建更值得信赖、更易于人类协作的AI系统提供一个强有力的范例，证明了将符号化的语义知识原生集成到深度学习模型决策回路中的巨大潜力。

**局限性与挑战**：与任何前沿研究一样，AMR-SELOR的实现也面临着诸多挑战与固有的局限性，这与SELOR原作者所秉持的严谨学术态度一致 。

- **对解析器质量的依赖**：整个框架的性能上限受制于上游AMR解析器的准确性。解析错误会直接引入错误的语义原子，对后续的规则生成和评估造成干扰。尽管SPRING等解析器已达到较高水平，但在特定领域或面对复杂、模糊的语言时，其错误仍不可避免。
- **计算复杂性**：引入AMR解析步骤无疑会增加模型的推理时间。更重要的是，为后件估计器生成预训练语料库需要对整个数据集进行离线解析和大规模规则采样，这是一个计算密集型的过程，对计算资源提出了更高的要求。
- **抽象概念的处理**：AMR图中包含一些非词汇化的抽象概念(如`amr-unknown`)，以及复杂的图结构。如何让前件生成器和后件估计器有效地学习和利用这些高度抽象的符号信息，是一个开放的研究问题。

**未来研究方向**：AMR-SELOR框架为未来的研究开辟了广阔的空间。

- **迈向一阶逻辑**：SELOR的作者指出，其模型局限于命题逻辑，而向一阶逻辑的演进是未来的重要方向 。AMR的结构化特性为实现这一目标提供了天然的跳板。未来可以研究如何从AMR三元组中学习带有变量和量词的规则(例如，`FORALL(x) such that (x, :instance, positive-word),...`)，从而实现更具泛化能力的抽象推理。
- **跨句子与文档级推理**：当前的AMR解析和AMR-SELOR框架主要针对单个句子。将该框架扩展到处理文档级的AMR图，有望让模型能够理解和解释段落级的语义连贯、论点发展和篇章关系，从而为文本摘要、对话系统等更复杂的任务提供高质量的解释。
- **端到端联合学习**：尽管本报告为保证可行性提出了分阶段的训练流程，但探索AMR解析器与SELOR模块的端到端联合训练是一个富有吸引力的长期目标。通过联合优化，AMR解析器或许能学会在保证语义准确性的同时，生成更“有利于解释”的图结构，从而实现语义理解与逻辑推理之间更深层次的协同。

总之，AMR-SELOR代表了在追求真正可理解AI道路上的一次有原则的、前瞻性的探索。它不仅是一个具体的模型提案，更是一种思想的倡导：即真正的可解释性必须植根于对“意义”的深刻理解。



## 致谢



感谢课程导师在构思此项工作过程中给予的启发性指导和宝贵建议。同时，感谢匿名审稿人提出的建设性意见。



## 参考文献



1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144).

2. Melis, D. A., & Jaakkola, T. (2018). Towards Robust Interpretability with Self-Explaining Neural Networks. In *Advances in Neural Information Processing Systems 31* (pp. 7775–7784).

3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774).

4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). Anchors: High-Precision Model-Agnostic Explanations. In *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1).

5. Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. In *International Conference on Learning Representations*.

6. Banarescu, L., Bonial, C., Cai, S., Georgescu, M., Griffitt, K., Hermjakob, U., Knight, K., Koehn, P., Palmer, M., & Schneider, N. (2013). Abstract Meaning Representation for Sembanking. In *Proceedings of the 7th Linguistic Annotation Workshop and Interoperability with Discourse* (pp. 178–186). 2

7. Bevilacqua, M., Blloshmi, R., & Navigli, R. (2021). One SPRING to Rule Them Both: Symmetric AMR Semantic Parsing and Generation without a Complex Pipeline. In *Proceedings of the AAAI Conference on Artificial Intelligence, 35*(15), 13215-13223. 8

8. Blloshmi, R., Bevilacqua, M., Fabiano, E., Caruso, V., & Navigli, R. (2021). SPRING Goes Online: End-to-End AMR Parsing and Generation. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (pp. 134–142).

9. Chanin, D., & Hunter, A. (2023). Neuro-symbolic Commonsense Social Reasoning. *arXiv preprint arXiv:2303.08264*. 18

10. Goodman, M. W. (2020). Penman: An Open-Source Library and Tool for AMR Graphs. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations* (pp. 312–319). 25

11. Jayaweera, C., Youm, S., & Dorr, B. J. (2024). AMREx: AMR for Explainable Fact Verification. In *Proceedings of the Seventh Fact Extraction and VERification Workshop (FEVER)* (pp. 234–244). 19

12. Lee, S., Yi, X., Wang, X., Xie, X., Han, S., & Cha, M. (2022). Self-explaining deep models with logic rule reasoning. In *Advances in Neural Information Processing Systems 35* (pp. 30161-30174).

13. Li, Z., & Gildea, D. (2024). A Hybrid Neuro-Symbolic Pipeline for Natural Language Understanding. *Applied Sciences, 16*(7), 529. 17

14. Minsky, M. L. (2022). Neuro-Symbolic AI: The 3rd Wave. *arXiv preprint arXiv:2208.13678*. 15

15. The ANSR Program. (n.d.). DARPA. Retrieved from https://www.darpa.mil/research/programs/assured-neuro-symbolic-learning-and-reasoning 16

16. Valenzuela-Escárcega, M. A., et al. (2020). Neuro-Symbolic AI: Bridging the Gap Between Symbolic Reasoning and Deep Learning. *ResearchGate*. 14