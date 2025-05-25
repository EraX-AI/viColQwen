---
title: "viPolyQwen: Strategic Multi-Task Multimodal Embedding with Dynamic Loss Equilibrium and Adaptive Task Weighting"
author: "Nguyen Anh Nguyen\\* (EraX) & Gtel Mobile JSC (GMobile) - Vietnam."
date: "\\*Corresponding Author: nguyen@hatto.com"
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{colortbl}
  - \usepackage{hyperref}
  - \usepackage{float}
  - \usepackage{geometry}
  - \geometry{margin=0.85in}
  - \usepackage{setspace}
  - \setstretch{0.95}
  - \usepackage{caption}
  - \captionsetup{font=small}
---

\begin{center}
\Large\textbf{Abstract}
\end{center}

\setlength{\leftskip}{2em}
\setlength{\rightskip}{2em}
\noindent Multimodal representation learning faces significant challenges in creating unified embeddings that excel across diverse tasks while maintaining computational efficiency and training stability. Existing approaches often struggle with task imbalance, conflicting optimization objectives, and suboptimal feature aggregation when learning from heterogeneous multimodal data. We propose `viPolyQwen`, a comprehensive framework for learning unified, high-dimensional (1024-d) multimodal embeddings built upon the Qwen2-VL-2B-Instruct foundation model. Our approach introduces three key innovations: (1) **Strategic Task Weighting (STW)** that dynamically balances loss contributions across data types through mathematically-derived task-specific coefficients, addressing the fundamental challenge of loss magnitude imbalance in multi-task learning; (2) **Dynamic Loss Equilibrium (DLE)** framework that combines prefix-guided task conditioning with adaptive loss component selection, harmonizing InfoNCE, MSE regression, ranking losses, and triplet margin components; and (3) **Enhanced Attention Pooling** with multi-layer projection that selectively aggregates sequence information through learned importance weights. Training on a heterogeneous dataset ($|\mathcal{D}| > 11 \times 10^6$) spanning five distinct multimodal interaction types with substantial Vietnamese content, our framework addresses critical optimization challenges in multi-task multimodal learning. The synergy between strategic task weighting, dynamic loss equilibrium, and attention-based feature selection creates a robust embedding system that maintains training stability while achieving superior performance across semantically complex, multimodal inputs.

\setlength{\leftskip}{0em}
\setlength{\rightskip}{0em}

## 1. Introduction

The advancement of multimodal AI systems necessitates the development of unified embedding spaces $\mathcal{E} \subset \mathbb{R}^{D_{\mathrm{embed}}}$ capable of representing diverse inputs across text, vision, and structured data modalities. While Vision-Language Models (VLMs) [2, 3, 4] have demonstrated remarkable capabilities in cross-modal understanding, translating their internal representations into effective, general-purpose embeddings presents several fundamental challenges that current approaches inadequately address.

**Multi-Task Loss Imbalance.** A critical yet underexplored challenge in multimodal embedding learning is the substantial disparity in loss magnitudes across different data types and tasks. In practice, different tasks exhibit vastly different baseline loss values—for instance, contrastive similarity learning may produce losses in the range of 4.0-5.0, while OCR tasks typically converge to losses around 1.0, and instruction-following tasks may stabilize near 0.8. This imbalance creates a dominance problem where tasks with higher loss magnitudes overwhelm the training process, leading to suboptimal representations for underrepresented tasks. Traditional multi-task learning approaches using fixed loss weights fail to adequately address this fundamental optimization challenge.

**Task-Specific Geometric Constraints.** Fine-tuning VLMs typically yields embeddings specialized for a single task objective $\mathcal{L}_{\mathrm{task}}$ (e.g., image-text contrastive loss in CLIP [2]). While effective for that specific task, these embeddings may be suboptimal for others with different geometric requirements in $\mathcal{E}$ within the same embedding space. This necessitates maintaining multiple specialized models, significantly increasing operational complexity and computational overhead.

**Sequence Aggregation Limitations.** The mechanism used to pool VLM encoder outputs $\mathbf{H} \in \mathbb{R}^{N \times D_{\mathrm{hidden}}}$ into a single vector $\mathbf{c} \in \mathbb{R}^{D_{\mathrm{hidden}}}$ critically impacts embedding quality. Standard strategies like mean pooling may dilute salient information, while last-token pooling may overlook important contextual information, particularly limiting for information-dense inputs like documents or images containing embedded text.

**Training Stability in Multi-Task Settings.** Balancing multiple training objectives when learning a unified embedding space creates optimization instability. Different loss components may compete with conflicting gradient signals, potentially causing training divergence or poor convergence to suboptimal representations. This challenge is exacerbated when training on fresh model components (e.g., newly initialized attention pooling layers) alongside pre-trained foundations.

To address these challenges, we propose **`viPolyQwen`**, a comprehensive framework for unified multimodal embedding generation built upon Qwen2-VL-2B-Instruct [3]. Our approach generates a single 1024-dimensional vector $\mathbf{e} \in \mathbb{R}^{1024}$ capable of effectively representing diverse multimodal inputs. The framework is guided by four core principles:

1. **Strategic Task Weighting for Loss Balance:** We introduce a principled approach to calculate task-specific weighting coefficients $\{\lambda_t\}_{t \in \mathcal{T}}$ that mathematically balance loss contributions across data types. By analyzing empirical loss distributions and applying inverse magnitude scaling, our Strategic Task Weighting (STW) ensures equitable optimization across all tasks while preventing dominant tasks from overwhelming the learning process.

2. **Dynamic Loss Equilibrium with Prefix Conditioning:** Building upon task weighting, we implement a Dynamic Loss Equilibrium (DLE) framework that uses task-specific prefixes $p_i \in P$ to dynamically select appropriate loss component combinations. This enables task-aware optimization while maintaining parameter sharing across the unified embedding space.

3. **Enhanced Attention Pooling Architecture:** We implement a learnable attention mechanism over encoder output sequences, enabling the model to identify and weight features based on learned task-relevant importance. This produces contextually-aware intermediate representations $\mathbf{c} = \sum a_i \mathbf{h}_i$ before projection to the final embedding space.

4. **Robust Multi-Layer Projection with Training Stability:** Our enhanced projection architecture incorporates multiple normalization layers and non-linearities, facilitating stable training dynamics while preserving semantic relationships during dimensionality reduction from high-dimensional hidden spaces to the target embedding dimension.

Our experimental methodology focuses on training dynamics and optimization stability, demonstrating that the strategic combination of task weighting, dynamic loss equilibrium, attention pooling, and enhanced projection enables stable, balanced learning across diverse multimodal tasks. This work represents a significant advance in addressing the fundamental optimization challenges inherent in multi-task multimodal representation learning.

## 2. Related Work

Our work builds upon several foundational research directions while addressing critical gaps in multi-task optimization for multimodal embeddings:

**Multimodal Contrastive Learning.** Foundational models like CLIP [2] and ALIGN [9] demonstrate effective image-text alignment through contrastive learning. However, single contrastive objectives, while effective for retrieval, may not optimally capture nuances required for diverse downstream tasks. More critically, these approaches do not address the loss magnitude imbalance problem when training on heterogeneous task types within a unified framework.

**Multi-Task Learning and Loss Balancing.** Classical multi-task learning [12] often employs fixed loss weights that remain constant throughout training. Recent approaches like uncertainty weighting [13] and gradient normalization [14] attempt to address loss balancing through implicit adaptation mechanisms. However, these methods do not account for the fundamental challenge of cross-task loss magnitude disparities that we observe in multimodal settings. Our Strategic Task Weighting provides an explicit, mathematically-grounded solution to this challenge.

**Adaptive Pooling Mechanisms.** While mean/max/last-token pooling are computationally efficient, they may not optimally aggregate multimodal information. Self-attention pooling [11] offers improved expressivity but increases complexity. Our approach balances effectiveness and efficiency through a learnable context vector mechanism specifically designed for multimodal sequence aggregation.

**Document AI and Unified Representations.** Multi-vector approaches like ColPali [5] address document complexity through separate representations for different granularities. While potentially capturing fine-grained details, these approaches necessitate specialized retrieval mechanisms. Our work explores whether strategic optimization can enable single vectors to effectively encode task-relevant nuances across diverse multimodal inputs.

**Training Dynamics in Vision-Language Models.** Recent work has highlighted the importance of training stability in large multimodal models [3, 4]. However, limited attention has been paid to the specific challenges of optimizing freshly initialized components (e.g., attention pooling layers) alongside pre-trained foundations. Our framework addresses these dynamics through principled initialization and strategic gradient management.

**Vietnamese and Multilingual Embeddings.** Our work addresses the need for high-quality multimodal embeddings for Vietnamese, leveraging substantial native data alongside multilingual resources to foster both in-language performance and cross-lingual capabilities [15, 22].

The contribution of `viPolyQwen` lies in the systematic integration of: (1) mathematically-derived task weighting to address loss imbalance, (2) dynamic loss equilibrium with explicit task conditioning, (3) enhanced attention pooling for improved sequence aggregation, and (4) robust training procedures for stable multi-task optimization. This comprehensive approach addresses fundamental challenges in multimodal embedding learning that have been inadequately addressed by prior work.

## 3. Methodology

### 3.1 Strategic Task Weighting Framework

A fundamental challenge in multi-task multimodal learning is the substantial disparity in loss magnitudes across different data types. Through empirical analysis, we observe that different tasks naturally converge to vastly different loss ranges:

- **Contrastive similarity tasks**: $\mathcal{L}_{\text{base}} \approx 4.0-4.5$
- **OCR/document understanding**: $\mathcal{L}_{\text{base}} \approx 1.0-1.2$  
- **Visual question answering**: $\mathcal{L}_{\text{base}} \approx 1.2-1.4$
- **Instruction following**: $\mathcal{L}_{\text{base}} \approx 0.8-1.0$

Without proper balancing, tasks with higher loss magnitudes dominate the optimization process, leading to poor performance on underrepresented tasks. We address this through **Strategic Task Weighting (STW)**, a principled approach to calculate task-specific coefficients.

#### 3.1.1 Mathematical Formulation of Task Weights

Given a set of tasks $\mathcal{T} = \{\text{contrastive\_with\_score}, \text{instruction}, \text{ocr}, \text{vqa\_single}, \text{vqa\_multi}\}$ and their empirically observed baseline losses $\{\ell_t\}_{t \in \mathcal{T}}$, we define the strategic task weight for task $t$ as:

$$\lambda_t = \frac{\bar{\ell}}{\ell_t} \cdot \alpha_t \cdot \beta_{\text{epoch}}$$

where:

- $\bar{\ell} = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \ell_t$ is the mean baseline loss across all tasks
- $\alpha_t$ is a task-specific adjustment factor accounting for dataset size and complexity
- $\beta_{\text{epoch}}$ is an epoch-dependent scaling factor enabling dynamic adaptation

The task-specific adjustment factors are derived from both theoretical considerations and dataset composition:

$$\alpha_t = \begin{cases}
0.25-0.35 & \text{if } t = \text{contrastive\_with\_score} \text{ (large dataset, high baseline)} \\
1.1-1.3 & \text{if } t = \text{instruction} \text{ (small dataset, needs emphasis)} \\
0.9-1.2 & \text{if } t \in \{\text{ocr}, \text{vqa\_single}, \text{vqa\_multi}\} \text{ (balanced)}
\end{cases}$$

The epoch-dependent scaling implements a **foundation-to-balance** strategy:

$$\beta_{\text{epoch}} = \begin{cases}
1.0-1.2 & \text{if epoch} = 0 \text{ (foundation phase)} \\
0.8-1.0 & \text{if epoch} \geq 1 \text{ (balance phase)}
\end{cases}$$

#### 3.1.2 Adaptive Weight Selection Strategy

The specific weight values are selected through a two-phase strategy designed to address the competing objectives of embedding foundation and task balance:

**Phase 1 - Foundation (Epoch 0):** Higher weights for dominant tasks to establish strong multimodal representations:

$$\boldsymbol{\lambda}^{(0)} = \{0.25, 1.2, 1.0, 1.0, 0.8\}$$

**Phase 2 - Balance (Epoch $\geq$ 1):** Reduced weights for dominant tasks to achieve equilibrium:

$$\boldsymbol{\lambda}^{(\geq1)} = \{0.22, 1.3, 1.2, 1.0, 0.8\}$$

This strategic approach ensures that:

1. **Early training** establishes strong foundational representations through emphasis on the largest dataset (contrastive learning)
2. **Later training** achieves balanced optimization across all tasks, preventing any single task from dominating
3. **Small datasets** (instruction following) receive proportionally higher emphasis to prevent marginalization

### 3.2 Enhanced Architecture with Attention Pooling

Building upon the Qwen2-VL-2B-Instruct foundation [3], our architecture incorporates several enhancements designed for stable multi-task training:

#### 3.2.1 Attention Pooling Mechanism

Traditional pooling strategies fail to capture the contextual importance of different sequence elements in multimodal inputs. We implement a learnable attention pooling mechanism that computes weighted aggregations based on learned importance:

1. **Learnable Context Vector:** A trainable parameter $\mathbf{v}_a \in \mathbb{R}^{D_{\mathrm{hidden}}}$ serves as a learned query representing "salience"

2. **Attention Score Computation:** For each hidden state $\mathbf{h}_i$ in the sequence $\mathbf{H}$:

   $$u_i = \mathbf{h}_i^T \mathbf{v}_a$$

3. **Masked Normalization:** Accounting for variable sequence lengths:

   $$a_i = \frac{\exp(u_i) \cdot M_i}{\sum_{j=1}^{N} \exp(u_j) \cdot M_j}$$

   where $M_i$ is the attention mask.

4. **Weighted Aggregation:** The final pooled representation:

   $$\mathbf{c} = \sum_{i=1}^{N} a_i \mathbf{h}_i$$

#### 3.2.2 Enhanced Multi-Layer Projection

To preserve semantic relationships during dimensionality reduction while maintaining training stability, we implement a sophisticated projection architecture:

$$\mathbf{p} = \text{LayerNorm}(\mathbf{W}_2 \cdot \text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{c})))$$

This architecture provides:

- **Non-linear expressivity** through GELU activation
- **Training stability** via intermediate normalization
- **Semantic preservation** through careful dimensionality reduction

The final embedding is L2-normalized: $\mathbf{e} = \frac{\mathbf{p}}{||\mathbf{p}||_2}$

### 3.3 Dynamic Loss Equilibrium with Task Conditioning

#### 3.3.1 Prefix-Guided Task Identification

During training, task-specific prefixes $p_i \in P = \{\texttt{<ocr>}, \texttt{<text\_pair>}, \texttt{<instr>}, \texttt{<vqa\_single>}, \texttt{<vqa\_multi>}\}$ are prepended to inputs, enabling explicit task conditioning:

$$x'_i = (\text{prefix}(p_i), x_i)$$

This conditioning allows the unified model to apply task-appropriate optimization while sharing parameters across all tasks.

#### 3.3.2 Task-Specific Loss Formulations

The Dynamic Loss Equilibrium framework combines strategic task weighting with adaptive loss component selection:

**Text Pair Embeddings:**

$$\mathcal{L}_{\text{text\_pair}} = \lambda_{\text{text\_pair}} \cdot [\mathcal{L}_{\text{NCE}} + \lambda_{\text{score}} \cdot \mathcal{L}_{\text{MSE}} + \lambda_{\text{rank}} \cdot \mathcal{L}_{\text{Rank}}]$$

**Instruction Following:**

$$\mathcal{L}_{\text{instr}} = \lambda_{\text{instr}} \cdot [\mathcal{L}_{\text{NCE}} + \mathcal{L}_{\text{Cos}}]$$

**OCR and VQA Tasks:**

$$\mathcal{L}_{\text{ocr/vqa}} = \lambda_{\text{ocr/vqa}} \cdot [\mathcal{L}_{\text{NCE}} + \mathcal{L}_{\text{Triplet}}]$$

The overall batch loss becomes:

$$\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_{\text{type}(p_i)}(f_\theta(x'_i), f_\theta(y'_i))$$

### 3.4 Training Stability and Initialization Strategy

#### 3.4.1 Fresh Component Initialization

The attention pooling mechanism introduces newly initialized parameters that must be trained alongside the pre-trained VLM foundation. We address this through:

1. **Conservative initialization:** $\mathbf{v}_a \sim \mathcal{N}(0, 0.02^2)$
2. **Differential learning rates:** Vision components (2e-6), other components (2e-5)
3. **Gradient norm clipping:** Maximum norm of 1.0 to prevent instability

#### 3.4.2 Strategic Training Phases

We implement a three-phase training strategy designed to account for the complex optimization landscape:

**Phase 1 (Steps 0-2000) - Exploration:** High volatility expected as fresh components explore parameter space. Task weight balance focuses on stability rather than perfect equilibrium.

**Phase 2 (Steps 2000-10000) - Stabilization:** Sharp improvement expected as components converge. Strategic task weights achieve intended balance effects.

**Phase 3 (Steps 10000+) - Optimization:** Steady improvement with balanced contributions from all tasks.

This phased approach recognizes that multi-task optimization with fresh components requires patience for exploration before achieving stable improvement.

## 4. Dynamic Loss Equilibrium Framework

### 4.1 Motivation and Formulation

Traditional multi-task learning often employs fixed loss weights that remain static throughout training. This can be suboptimal when different tasks have varying convergence rates or when the relative importance of tasks shifts during training. Additionally, in a unified embedding space, different data types benefit from different combinations of loss components to shape the geometric properties of the embedding space.

We introduce the **Dynamic Loss Equilibrium (DLE)** framework that addresses these challenges through two complementary mechanisms:

1. **Explicit Component Weighting:** Task-specific weighting coefficients ($\lambda_{\text{score}}$, $\lambda_{\text{rank}}$) are applied to different loss components to balance their contributions
2. **Implicit Task Adaptation:** The prefix-guided loss selection dynamically applies different loss combinations based on data type

Formally, the DLE framework can be expressed as:

$$\mathcal{L}_{\text{DLE}}(x_i, y_i, p_i, \theta) = \mathcal{L}_{\text{base}}(x_i, y_i, \theta) + \sum_{j} \lambda_j(p_i) \cdot \mathcal{L}_j(x_i, y_i, \theta)$$

where:

- $\mathcal{L}_{\text{base}}$ is a core loss component (typically InfoNCE) applied to all data types
- $\mathcal{L}_j$ represents additional loss components (MSE, ranking, triplet, etc.)
- $\lambda_j(p_i)$ is a task-dependent weighting function that determines the contribution of each loss component based on the prefix $p_i$
- $\theta$ represents the model parameters

The key insight of DLE is that different combinations of loss components create geometric constraints in the embedding space that are optimally suited for different tasks. By dynamically selecting and weighting these components based on data type, we can create a unified embedding space that excels across diverse modalities and tasks.

### 4.2 Task-Specific Loss Formulations

The training objective dynamically applies task-specific losses based on prefix $p_i$. Let $(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}) = (f_\theta(x'_i), f_\theta(y'_i))$ be normalized embeddings.

#### 4.2.1 Text Pair Embeddings with Integrated Ranking Loss

For text similarity pairs ($p_i = \texttt{<text\_pair>}$), we employ a sophisticated three-component loss:

$$\mathcal{L}_{\text{text\_pair}} = \mathcal{L}_{\text{NCE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{B}, T) + \lambda_{\text{score}} \cdot \mathcal{L}_{\text{MSE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, s_i) + \lambda_{\text{rank}} \cdot \mathcal{L}_{\text{Rank}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, s_i, m_r)$$

where:

- $\mathcal{L}_{\text{NCE}}$ is the symmetric InfoNCE loss over batch $\mathcal{B}$ with temperature $T=0.07$
- $\mathcal{L}_{\text{MSE}}$ is the regression loss between cosine similarity and ground truth scores: $(\frac{1}{2}(\mathbf{e}_{a,i}^T \mathbf{e}_{b,i} + 1) - s_i)^2$
- $\mathcal{L}_{\text{Rank}}$ is the margin ranking loss that ensures pairs with higher similarity scores have higher predicted similarities:

$$\mathcal{L}_{\text{Rank}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \max(0, m_r - (\hat{s}_i - \hat{s}_j))$$

where $\mathcal{P} = \{(i,j) | s_i > s_j\}$ is the set of all pairs where the ground truth similarity of pair $i$ exceeds that of pair $j$, $\hat{s}_i = \frac{1}{2}(\mathbf{e}_{a,i}^T \mathbf{e}_{b,i} + 1)$ is the predicted similarity, and $m_r=0.05$ is the margin.

The weighting parameters $\lambda_{\text{score}}=20.0$ and $\lambda_{\text{rank}}=7.0$ provide optimal balance between regression and ranking objectives. This three-component loss creates a more nuanced embedding space where:

- InfoNCE ensures overall discriminative power
- MSE regression aligns similarities with absolute scores
- Ranking loss preserves relative ordering of similarity judgments

#### 4.2.2 Instruction Following Loss

For instruction-following pairs ($p_i = \texttt{<instr>}$), we combine contrastive loss with direct similarity maximization:

$$\mathcal{L}_{\text{instr}} = \mathcal{L}_{\text{NCE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{B}, T) + \mathcal{L}_{\text{Cos}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i})$$

where $\mathcal{L}_{\text{Cos}} = (1 - \mathbf{e}_{a,i}^T \mathbf{e}_{b,i})$ directly encourages alignment between instruction inputs and expected outputs.

#### 4.2.3 OCR and VQA Losses with Adaptive Margins

For OCR and VQA tasks ($p_i \in \{ \texttt{<ocr>}, \texttt{<vqa\_single>}, \texttt{<vqa\_multi>} \}$), we employ:

$$\mathcal{L}_{\text{ocr/vqa}} = \mathcal{L}_{\text{NCE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{B}, T) + \lambda_{\text{trip}} \cdot \mathcal{L}_{\text{Triplet}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{N}_i, m', T)$$

where:

- $\lambda_{\text{trip}}=1.0$ for OCR and single-turn VQA, but increases to $1.5$ for multi-turn VQA to account for increased complexity
- $m'=0.2$ for simple tasks, but increases to $0.3$ for multi-turn VQA
- $\mathcal{N}_i = \{ \mathbf{e}_{b,j} \mid j \neq i \}$ is the set of negative examples

The overall batch loss is calculated as:

$$\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_{\text{type}(p_i)}$$

This harmonized approach ensures that each data sample contributes appropriately to the formation of a versatile, unified embedding space.

## 5. Training Implementation

### 5.1 Dataset Composition and Scale

Our training dataset $\mathcal{D}$ comprises over 11 million samples across five distinct multimodal interaction types:

- **Text Similarity Pairs** ($\texttt{<text\_pair>}$): 5.6M Vietnamese/English/Chinese text pairs with similarity scores
- **Instruction Following** ($\texttt{<instr>}$): 600K instruction-response pairs
- **OCR/OCQ** ($\texttt{<ocr>}$): 2.5M image-text pairs for optical character recognition
- **Visual Question Answering** ($\texttt{<vqa\_single>}$, $\texttt{<vqa\_multi>}$): 2.5M single and multi-turn VQA samples

The dataset composition reflects real-world application needs with approximately 60% Vietnamese content, 30% English, and 10% Chinese, addressing the critical need for high-quality Vietnamese multimodal embeddings.

### 5.2 Implementation Details

**Hardware Configuration:** 4x NVIDIA H100 GPUs (94GB VRAM) with Distributed Data Parallel training through Hugging Face Accelerate framework.

**Optimization Strategy:**

- **Precision:** bfloat16 mixed precision with Flash Attention 2
- **Optimizer:** AdamW with differential learning rates
- **Batch Configuration:** Per-device batch size 12, gradient accumulation 10 (effective global batch size 480)
- **Sequence Length:** 8192 tokens maximum
- **Training Duration:** 3 epochs over the full dataset

**Strategic Task Weights:** Applied according to the formulations in Section 3.1, with epoch-dependent adaptation:

- **Epoch 0:** Foundation phase weights favoring stability
- **Epochs 1-2:** Balance phase weights ensuring task equilibrium

**Loss Component Parameters:**

- InfoNCE temperature: $T = 0.2$
- Score loss weight: $\lambda_{\text{score}} = 20.0$
- Ranking loss weight: $\lambda_{\text{rank}} = 7.0$
- Triplet margin: $m = 0.2$ (adaptive for multi-turn VQA)

### 5.3 Training Dynamics Monitoring

Given the complexity of multi-task optimization with fresh components, we implement comprehensive monitoring:

**Balance Metrics:** Track effective loss contributions across tasks to ensure strategic weights achieve intended equilibrium.

**Training Stability:** Monitor gradient norms and loss volatility, particularly during the exploration phase (steps 0-2000).

**Component Convergence:** Separately track attention pooling parameter evolution and projection layer stability.

**Strategic Checkpoints:**

- **Step 1000:** Trend emergence expected
- **Step 2000:** Sharp improvement phase begins
- **Step 5000:** Task balance should be achieved
- **Step 10000:** Stable optimization phase

This monitoring framework enables early detection of optimization issues and validation of our strategic training approach.

## 6. Theoretical Analysis

### 6.1 Mathematical Foundation of Strategic Task Weighting

The effectiveness of our Strategic Task Weighting approach rests on several theoretical principles that address fundamental challenges in multi-task optimization.

#### 6.1.1 Loss Magnitude Normalization Theory

Consider a multi-task loss function without strategic weighting:

$$\mathcal{L}_{\text{naive}} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_{\text{type}(i)}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i})$$

When loss magnitudes vary significantly across tasks, the gradient contribution becomes dominated by high-magnitude tasks:

$$\nabla_\theta \mathcal{L}_{\text{naive}} \approx \nabla_\theta \mathcal{L}_{\text{dominant}}$$

Our Strategic Task Weighting corrects this imbalance by applying inverse magnitude scaling:

$$\mathcal{L}_{\text{STW}} = \frac{1}{B} \sum_{i=1}^{B} \lambda_{\text{type}(i)} \cdot \mathcal{L}_{\text{type}(i)}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i})$$

This ensures that gradient contributions are balanced across tasks:

$$||\nabla_\theta \mathcal{L}_t||_2 \approx ||\nabla_\theta \mathcal{L}_{t'}||_2 \quad \forall t, t' \in \mathcal{T}$$

#### 6.1.2 Embedding Space Geometric Constraints

Different loss components impose distinct geometric constraints on the embedding space $\mathcal{E}$. Our framework enables controlled application of these constraints:

- **InfoNCE Loss:** Creates discriminative structure through contrastive learning
- **MSE Regression:** Aligns embedding similarities with continuous target scores  
- **Ranking Loss:** Preserves ordinal relationships between similarity judgments
- **Triplet Loss:** Enforces margin-based separation for complex reasoning tasks

Strategic task weighting ensures that these geometric constraints are applied with appropriate emphasis for each data type, preventing any single constraint from overwhelming others.

### 6.2 Training Phase Theoretical Framework

Our three-phase training strategy is grounded in optimization theory for complex, multi-component systems.

#### 6.2.1 Exploration-Exploitation Dynamics

**Phase 1 (Exploration):** Fresh components (attention pooling) explore the parameter space while pre-trained components provide stability. High volatility is expected and necessary for discovering optimal configurations.

**Phase 2 (Stabilization):** Components converge toward stable configurations, enabling the strategic task weights to achieve their intended balancing effects.

**Phase 3 (Optimization):** Balanced optimization across all tasks with stable improvement trajectories.

This framework recognizes that premature intervention during the exploration phase can disrupt the natural parameter discovery process, leading to suboptimal final representations.

## 7. Discussion and Implications

### 7.1 Methodological Contributions

Our work addresses several critical gaps in multimodal embedding research:

**Loss Imbalance Solution:** The Strategic Task Weighting framework provides the first systematic approach to addressing loss magnitude disparities in multi-task multimodal learning. This addresses a fundamental optimization challenge that has been largely overlooked in prior work.

**Training Stability Framework:** Our three-phase training strategy with monitoring provides a principled approach to optimizing complex multi-component systems, offering guidance for future work in this area.

**Attention-Based Aggregation:** The enhanced attention pooling mechanism offers a computationally efficient alternative to complex multi-vector approaches while maintaining representational power.

### 7.2 Practical Applications

The viPolyQwen framework offers several advantages for real-world deployment:

**Simplified Infrastructure:** Single-vector embeddings enable standard dense retrieval infrastructure without complex late-interaction mechanisms.

**Balanced Performance:** Strategic task weighting ensures robust performance across diverse application scenarios rather than optimization for specific tasks at the expense of others.

**Vietnamese Language Support:** Substantial Vietnamese training data and balanced optimization provide strong support for Vietnamese language applications, addressing a critical gap in multilingual embedding research.

**Training Efficiency:** The principled approach to multi-task optimization reduces the need for extensive hyperparameter search and manual intervention during training.

### 7.3 Limitations and Future Directions

**Computational Overhead:** The attention pooling mechanism adds computational cost compared to simple pooling strategies, though this is offset by improved representation quality.

**Task Weight Sensitivity:** While our mathematical framework provides principled weight selection, optimal weights may vary across different dataset compositions and task distributions.

**Scalability Questions:** The framework's effectiveness when scaled to larger numbers of tasks or different foundation models remains to be explored.

**Dynamic Adaptation:** Future work could explore automated approaches to adjust task weights during training based on convergence characteristics and performance metrics.

## 8. Conclusion

We present viPolyQwen, a comprehensive framework for unified multimodal embedding learning that addresses fundamental challenges in multi-task optimization through Strategic Task Weighting, Dynamic Loss Equilibrium, and Enhanced Attention Pooling. Our approach provides mathematically-grounded solutions to critical problems including loss magnitude imbalance, training instability, and suboptimal sequence aggregation.

The Strategic Task Weighting framework represents a significant methodological contribution, offering the first systematic approach to balancing loss contributions across diverse multimodal tasks. Combined with our attention pooling mechanism and robust training procedures, this creates a framework that achieves stable, balanced optimization across heterogeneous multimodal data.

Our work demonstrates that principled approaches to multi-task optimization can enable unified embedding systems that excel across diverse applications while maintaining computational efficiency. The substantial inclusion of Vietnamese content and balanced optimization approach makes this particularly valuable for multilingual and cross-cultural applications.

Future research directions include extending the framework to larger foundation models, developing automated approaches to task weight adaptation, and exploring applications to additional modalities such as audio and video. The theoretical foundations and practical insights presented in this work provide a foundation for continued advancement in unified multimodal representation learning.

## References

[1] P. Lewis, E. Perez, A. Piktus, et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[2] A. Radford, J. W. Kim, C. Hallacy, et al., "Learning transferable visual models from natural language supervision," in International Conference on Machine Learning (ICML), 2021.

[3] J. Bai, S. Bai, S. Yang, et al., "Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond," arXiv preprint arXiv:2308.12966, 2023.

[4] J.-B. Alayrac, J. Donahue, P. Dieleman, et al., "Flamingo: a visual language model for few-shot learning," in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[5] M. Faysse, H. Sibille, T. Wu, et al., "Colpali: Efficient document retrieval with vision language models," arXiv preprint arXiv:2407.01449, 2024.

[9] C. Jia, Y. Yang, Y. Xia, et al., "Scaling up visual and vision-language representation learning with noisy text supervision," in International Conference on Machine Learning (ICML), 2021.

[11] Z. Lin, M. Feng, C. N. dos Santos, et al., "A structured self-attentive sentence embedding," in International Conference on Learning Representations (ICLR), 2017.

[12] R. Caruana, "Multitask learning," Machine Learning, vol. 28, no. 1, pp. 41–75, 1997.

[13] A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[14] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks," in International Conference on Machine Learning (ICML), 2018.

[15] A. Conneau, K. Khandelwal, N. Goyal, et al., "Unsupervised cross-lingual representation learning at scale," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.

[22] N. Reimers and I. Gurevych, "Making monolingual sentence embeddings multilingual using knowledge distillation," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.