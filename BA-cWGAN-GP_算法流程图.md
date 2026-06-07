# BA-cWGAN-GP 算法流程图

```mermaid
flowchart TB
    subgraph INPUT[" 输入数据 "]
        D1["① 2022Q1–2023Q3 完整观测数据<br/>~1,015K 笔贷款，含 DFlag 标签<br/>违约样本可追溯首次违约月份"]
        D2["② 2023Q4–2025Q2 纯违约数据<br/>8,150 笔违约贷款（右删失）<br/>无 DFlag=0 对照样本"]
    end

    D1 --> P1A
    D1 --> P2A

    subgraph P1[" Phase 1：违约倾向得分提取 "]
        direction TB
        P1A["从月度 Performance 数据中<br/>提取首次违约月份"] --> P1B["标注 Early Default (≤12月)<br/>标注 Late Default (>12月)"]
        P1B --> P1C["训练 speed_clf（XGBoost）<br/>二分类：Early vs Late"]
        D2 --> P1D["对右删失违约样本<br/>用 speed_clf 预测 r"]
        P1C --> P1D
        P1C --> P1E["对训练集违约样本<br/>预测 r_train"]
        P1D --> P1OUT["r ∈ [0,1]<br/>r 越大 = 越倾向快速违约"]
        P1E --> P1OUT
    end

    subgraph P2[" Phase 2：决策边界识别 "]
        direction TB
        P2A["完整训练集<br/>（DFlag=0 + DFlag=1）"] --> P2B["训练 base_clf（XGBoost）<br/>二分类：Default vs Non-Default"]
        P2B --> P2C["对少数类样本预测 p"]
        P2C --> P2D["边界距离 d = |p - 0.5|<br/>d 越小 = 越靠近决策边界"]
        P2D --> P2E["采样权重 w = 1 / (d + ε)<br/>边界样本获得更高权重"]
        P2E --> P2OUT["带权重的少数类样本<br/>+ 边界距离标签"]
    end

    P1OUT --> P3
    P2OUT --> P3

    subgraph P3[" Phase 3：BA-cWGAN-GP 定向生成 "]
        direction TB
        P3IN1["随机噪声 z ~ N(0,1)<br/>100 维"] --> P3CAT
        P3IN2["违约倾向得分 r<br/>从训练集 r 分布采样"] --> P3CAT
        P3CAT["拼接 z_cond = [z, r]<br/>101 维"] --> G

        subgraph GAN[" 对抗训练循环 "]
            direction LR
            G["生成器 G(z | r)<br/>101→256→512→256→13"] --> XF["合成样本 x_fake<br/>13 维特征向量"]
            XF --> D["判别器 D(x)<br/>13→256→512→256→1"]
            XR["真实少数类样本 x_real<br/>（按 w 加权采样）"] --> D
            D --> DLOSS["判别器损失 L_D<br/>= D(x_fake) - D(x_real) + λ_gp·GP"]
            XF --> GLOSS["生成器损失 L_G<br/>= -D(x_fake)  ← 对抗"]
            XF --> BLOSS[" + λ_boundary·L_boundary<br/>= -|base_clf(x_fake) - 0.5|  ← 边界感知"]
            XF --> RLOSS[" + λ_r·L_r<br/>= |speed_clf(x_fake) - r|  ← 违约倾向一致"]
            DLOSS -->|"每步更新"| D
            GLOSS -->|"每 5 步更新"| G
            BLOSS -->|"每 5 步更新"| G
            RLOSS -->|"每 5 步更新"| G
        end

        G --> P3GEN["训练完成后批量生成"]
        P3GEN --> P3FILTER["边界筛选<br/>保留 base_clf(x_fake) ∈ [0.3, 0.7]"]
    end

    P3FILTER --> OUTPUT

    subgraph OUTPUT[" 输出 "]
        O1["高质量合成违约样本<br/>集中于决策边界附近"]
        O2["均衡训练集<br/>（真实样本 + 合成样本）"]
    end

    OUTPUT --> EVAL["下游评估：XGBoost 裁判<br/>在 GAN 验证集上对比 AUC / F1 / KS"]

    style INPUT fill:#e8f4f8,stroke:#0366d6
    style P1 fill:#fff3cd,stroke:#f0ad4e
    style P2 fill:#fff3cd,stroke:#f0ad4e
    style P3 fill:#d4edda,stroke:#28a745
    style OUTPUT fill:#e8f4f8,stroke:#0366d6
    style GAN fill:#f8f9fa,stroke:#6c757d
```

## 流程图说明

### 颜色含义

| 颜色 | 模块 | 含义 |
|---|---|---|
| 蓝色边框 | 输入数据 / 最终输出 | 数据流入流出 |
| 黄色边框 | Phase 1 & 2 | 信息提取阶段（为生成做准备） |
| 绿色边框 | Phase 3 | 核心生成阶段（本文创新点） |

### 数据流向

```
完整观测数据 ──→ Phase 1 (speed_clf) ──→ r ──┐
         │                                      ├──→ Phase 3 (BA-cWGAN-GP) ──→ 合成样本
         └──→ Phase 2 (base_clf) ──→ d, w ──┘
                                                      │
右删失违约数据 ──→ Phase 1 (仅预测 r) ──────────────────┘
```

### 三项损失的分工

| 损失项 | 公式 | 告诉生成器什么 |
|---|---|---|
| 对抗损失 L_adv | -D(G(z\|r)) | "生成样本要像真的违约样本" |
| 边界损失 L_boundary | -\|base_clf(x_fake) - 0.5\| | "要在分类器最拿不准的地方生成" |
| r 一致性损失 L_r | \|speed_clf(x_fake) - r\| | "违约速度模式要和条件 r 一致" |

---

*配合 `BA-cWGAN-GP_方法论详解.md` 第三节阅读*
