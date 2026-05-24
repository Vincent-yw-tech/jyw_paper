# imbalanced_dataset_FM24_2022_2025.csv 数据说明

## 概述

本数据集按照 **Mushava & Murray (2024)** "Flexible loss functions for binary classification in gradient-boosted decision trees: An application to credit scoring"（*Expert Systems With Applications*, 238, 121876）的方法论构建，用于信用评分二分类任务，**未做任何重采样/平衡处理**。

## 数据来源

- **原始数据**：Freddie Mac Single-Family Loan-Level Dataset（房地美单户贷款级数据集）
- **时间范围**：2022Q1–2025Q3，共 14 个季度
- **观测窗口**：FM24（贷款发放后 24 个月内的违约状态）
- **处理脚本**：`build_imbalanced_dataset.py`

## 数据集规模

| 指标 | 数值 |
|---|---|
| 样本数 | **1,022,701** 笔贷款 |
| 特征数 | **79** 个预测变量 + 1 个目标变量 = **80 列** |
| 文件大小 | **174.8 MB** |
| 违约样本 | **42,227**（4.13%） |
| 正常样本 | **980,474**（95.87%） |
| 不平衡比 (IR) | **23** |

## 数据处理流程

| 步骤 | 说明 | 结果 |
|---|---|---|
| 1. 扫描 Performance | 读取 14 个季度的月度表现数据（约 3 亿行），按贷款口径聚合 max_loan_age 和窗口内违约状态 | 4,203,883 笔有表现记录的贷款 |
| 2. 违约判定 | 24 个月观测窗口内：逾期 ≥ 90 天（DLQ ≥ 3）或 止赎/处置（ZBC = 03/09） | 50,166 笔违约 |
| 3. 完整观测筛选 | 仅保留"已违约"或"观测 ≥ 24 个月且未违约"的贷款，剔除观测不足的样本 | 2,023,649 笔完整观测 |
| 4. 信用分筛选 | 剔除 Credit_Score ≥ 754 的样本 | 并入 Step 5 边读边筛 |
| 5. 特征工程 | 按完整观测 ID 集过滤 Origination → 信用分 300–753 → 数值特征 + One-Hot 编码 | 1,022,701 笔贷款 × 80 列 |

### 内存优化策略

Performance 和 Origination 文件总量约 50 GB，采用**两遍扫描、边读边写**策略：
- **第一遍**：逐 chunk 扫描 Performance → 聚合 per-loan 的 max_loan_age + 违约状态 → 构建有效贷款 ID 集合（dict 存储）
- **第二遍**：逐 chunk 读取 Origination → 按 valid_ids 过滤 → 特征工程 → `dict.map()` 挂载 DFlag → 追加写入 CSV

全程不将完整 DataFrame 加载到内存，峰值内存 < 2 GB。

### 两个核心筛选逻辑

**1. 信用分截断（300 ≤ Credit_Score < 754）**

论文依据：信用评分的信息值（IV）远大于 0.5，属于"过度预测"变量。若保留高分样本，模型可能退化为仅依赖信用分的单特征模型，失去对其他特征的敏感性。

此外，原始数据中 Credit_Score = 9999 表示缺失，这些样本同样被剔除。

**2. 完整 24 个月观测期**

违约标签需要在固定的观测窗口内才能准确标注：
- **DFlag = 1**：贷款在发放后 24 个月内出现严重逾期（≥ 90 天）或被止赎/处置（Zero_Balance_Code = 03/09）
- **DFlag = 0**：贷款至少被观测了 24 个月且未发生上述事件
- 观测不足 24 个月且未违约的贷款被**排除**（右删失）

> 论文使用了 FM12/FM36/FM60 三种窗口。考虑到本数据的时间跨度（2022–2025年9月）和 OOT 验证需求，选择 FM24 窗口——既保证足够的违约样本量，又留有跨时期的测试数据。

## 季度分布与右删失

| 季度 | 贷款数 | 违约数 | 违约率 | 说明 |
|---|---|---|---|---|
| 2022Q1 | 282,795 | 6,955 | 2.46% | 完整 24m 观测 |
| 2022Q2 | 208,423 | 6,775 | 3.25% | 完整 24m 观测 |
| 2022Q3 | 156,518 | 5,860 | 3.74% | 完整 24m 观测 |
| 2022Q4 | 97,785 | 3,861 | 3.95% | 完整 24m 观测 |
| 2023Q1 | 82,331 | 2,984 | 3.62% | 完整 24m 观测 |
| 2023Q2 | 113,005 | 3,883 | 3.44% | 完整 24m 观测 |
| 2023Q3 | 73,694 | 3,759 | 5.10% | 完整 24m 观测 |
| 2023Q4 | 2,410 | 2,410 | 100% | **仅违约样本**（观测 21–23 月，未违约的被剔除） |
| 2024Q1 | 1,848 | 1,848 | 100% | **仅违约样本**（观测 17–20 月） |
| 2024Q2 | 1,908 | 1,908 | 100% | **仅违约样本**（观测 14–17 月） |
| 2024Q3 | 1,276 | 1,276 | 100% | **仅违约样本**（观测 11–14 月） |
| 2024Q4 | 526 | 526 | 100% | **仅违约样本**（观测 8–11 月） |
| 2025Q1 | 146 | 146 | 100% | **仅违约样本**（观测 5–8 月） |
| 2025Q2 | 36 | 36 | 100% | **仅违约样本**（观测 2–5 月） |

> **2023Q4 之后全部为违约样本**，原因是数据截止于 2025 年 9 月，这些季度的贷款观测时长不足 24 个月，未违约的样本无法确认其完整状态（右删失）。建模时应仅使用 **2022Q1–2023Q3** 的完整观测数据，或按需对 2023Q4+ 做截断处理。

## 列名说明

### 一、数值特征（8 列）

| 列名 | 来源字段 | 类型 | 说明 |
|---|---|---|---|
| `Loanref` | `Loan_Sequence_Number` | str | 贷款唯一标识，Freddie Mac 格式（如 F22Q10000002，F=Freddie, 22=2022年, Q1=Q1季度） |
| `Credit_Score` | `Credit_Score` | int | 借款人信用评分（FICO），原始范围 301–850（9999=缺失），本数据集截断为 **300–753** |
| `Mortgage_Insurance` | `Mortgage_Insurance_Percentage` | float | 按揭保险覆盖率，**已从原始整数转为小数**（原始 000–099 → 0.00–0.99，999=缺失填 0） |
| `Number_of_units` | `Number_of_Units` | int | 房产单元数：1=独栋单户，2=双户，3=三户，4=四户，9=缺失填 0 |
| `CLoan_to_value` | `Original_Combined_Loan_To_Value_CLTV` | float | 合并贷款价值比（Combined LTV），含二押等次贷在内的总贷款额 / 房产价值 × 100，范围 1–200（999=缺失填 0） |
| `Debt_to_income` | `Original_Debt_To_Income_DTI_Ratio` | float | 债务收入比（DTI），月债务支出 / 月收入 × 100，范围 1–65（**999 = 缺失，未清洗**） |
| `OLoan_to_value` | `Original_Loan_To_Value_LTV` | float | 第一抵押贷款价值比（LTV），仅首贷金额 / 房产价值 × 100，范围 6–135（999=缺失填 0） |
| `Single_borrower` | `Number_of_Borrowers` → 衍生 | int | 是否只有 1 个借款人：`Number_of_Borrowers == "01"` → 1，否则 0。均值 0.535 |

### 二、One-Hot 编码特征（71 列）

所有 one-hot 编码的特征取值均为 **0（否）** 或 **1（是）**。

#### 2.1 贷款目的（3 列）— 来源：`Loan_Purpose`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_Loan_purpose_purc` | P（Purchase） | 购房贷款 |
| `is_Loan_purpose_cash` | C（Cash-out Refinance） | 套现再融资 |
| `is_Loan_purpose_noca` | N（No-cash Refinance） | 无现金再融资 |

#### 2.2 首次购房者（3 列）— 来源：`First_Time_Homebuyer_Flag`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_First_time_homeowner` | Y | 是首次购房者 |
| `is_First_time_homeowner_No` | N | 不是首次购房者 |
| `is_First_time_homeowner_miss` | 9 / 空 / 其他 | 缺失或未知 |

#### 2.3 入住状态（3 列）— 来源：`Occupancy_Status`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_Occupancy_status_prim` | P（Primary Residence） | 主要居所（自住房） |
| `is_Occupancy_status_inve` | I（Investment Property） | 投资房 |
| `is_Occupancy_status_seco` | S（Second Home） | 第二居所 |

> 自住房（P）通常违约风险最低，投资房（I）风险最高。

#### 2.4 贷款渠道（3 列）— 来源：`Channel`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_Origination_channel_reta` | R（Retail） | 零售渠道 |
| `is_Origination_channel_brok` | B（Broker） | 经纪人渠道 |
| `is_Origination_channel_corr` | C（Correspondent） | 代理渠道 |

#### 2.5 房产类型（5 列）— 来源：`Property_Type`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_Property_type_cond` | CO（Condominium） | 公寓 |
| `is_Property_type_coop` | CP（Cooperative） | 合作公寓 |
| `is_Property_type_manu` | MH（Manufactured Housing） | 预制/移动房屋 |
| `is_Property_type_pud` | PU（Planned Unit Development） | 规划单元开发 |
| `is_Property_type_sing` | SF（Single-Family） | 独栋住宅 |

#### 2.6 房产所在州（52 列）— 来源：`Property_State`

命名格式 `is_property_state_XX`，覆盖 52 个州/地区代码。

| 代码 | 州/地区 | 代码 | 州/地区 | 代码 | 州/地区 |
|---|---|---|---|---|---|
| AK | 阿拉斯加 | ME | 缅因 | SC | 南卡罗来纳 |
| AL | 阿拉巴马 | MI | 密歇根 | SD | 南达科他 |
| AR | 阿肯色 | MN | 明尼苏达 | TN | 田纳西 |
| AZ | 亚利桑那 | MO | 密苏里 | TX | 得克萨斯 |
| CA | 加利福尼亚 | MS | 密西西比 | UT | 犹他 |
| CO | 科罗拉多 | MT | 蒙大拿 | VA | 弗吉尼亚 |
| CT | 康涅狄格 | NC | 北卡罗来纳 | VI | 维尔京群岛 |
| DC | 哥伦比亚特区 | ND | 北达科他 | VT | 佛蒙特 |
| DE | 特拉华 | NE | 内布拉斯加 | WA | 华盛顿 |
| FL | 佛罗里达 | NH | 新罕布什尔 | WI | 威斯康星 |
| GA | 佐治亚 | NJ | 新泽西 | WV | 西弗吉尼亚 |
| GU | 关岛 | NM | 新墨西哥 | WY | 怀俄明 |
| HI | 夏威夷 | NV | 内华达 | | |
| IA | 爱荷华 | NY | 纽约 | | |
| ID | 爱达荷 | OH | 俄亥俄 | | |
| IL | 伊利诺伊 | OK | 俄克拉荷马 | | |
| IN | 印第安纳 | OR | 俄勒冈 | | |
| KS | 堪萨斯 | PA | 宾夕法尼亚 | | |
| KY | 肯塔基 | PR | 波多黎各 | | |
| LA | 路易斯安那 | RI | 罗德岛 | | |
| MA | 马萨诸塞 | | | | |

### 三、目标变量（1 列）

| 列名 | 类型 | 说明 |
|---|---|---|
| `DFlag` | int | **违约标识**。1 = 贷款在发放后 **24 个月内**出现严重逾期（≥ 90 天）或被止赎/处置（ZBC = 03/09）；0 = 贷款已被观测 ≥ 24 个月且未违约 |

## 数值特征统计

| 特征 | 均值 | 标准差 | 最小值 | 最大值 |
|---|---|---|---|---|
| Credit_Score | 708.21 | 32.82 | 300 | 753 |
| Mortgage_Insurance | 0.091 | 0.130 | 0.00 | 0.35 |
| Number_of_units | 1.029 | 0.217 | 1 | 4 |
| CLoan_to_value | 75.02 | 18.22 | 3 | 196 |
| Debt_to_income | 38.26 | 14.38 | 1 | 65 (999=缺失) |
| OLoan_to_value | 74.84 | 18.17 | 3 | 196 |
| Single_borrower | 0.535 | 0.499 | 0 | 1 |

## 缺失值处理

| 字段 | 原始缺失编码 | 处理方式 |
|---|---|---|
| `Credit_Score` | 9999 | 筛选阶段排除（9999 ≥ 754） |
| `Mortgage_Insurance` | 999 | 填充为 0 |
| `CLoan_to_value`, `OLoan_to_value` | 999 | 填充为 0 |
| `Debt_to_income` | 999 | **未清洗**（999 为有效数值，建议建模前替换为 0 或剔除） |
| `Number_of_units` | 9 | 填充为 0 |
| `Loan_Purpose` | 9 / 空 | 未命中 P/C/N 的自然归 0 |
| `First_Time_Homebuyer_Flag` | 9 / 空 | 归入 `is_First_time_homeowner_miss` |
| `Occupancy_Status` | 9 / 空 | 未命中 P/I/S 的自然归 0 |
| `Channel` | 9 / 空 | 未命中 R/B/C 的自然归 0 |
| `Property_Type` | 空 | 未命中已知类型的自然归 0 |

## OOS/OOT 划分建议

基于季度分布，建议如下划分：

| 数据集 | 时间范围 | 贷款数 | 违约数 | 违约率 | 用途 |
|---|---|---|---|---|---|
| 训练 + OOS | 2022Q1–Q4 | ~745,521 | ~23,451 | ~3.15% | 模型训练与超参调优（同分布交叉验证） |
| OOT 测试 | 2023Q1–Q3 | ~269,030 | ~10,626 | ~3.95% | 跨时期泛化能力验证 |

- **2022 年**（低利率环境末期）与 **2023 年**（加息周期高峰）宏观经济环境差异显著，OOT 测试可有效检验模型在利率政策剧变下的稳健性
- 2023Q4–2025Q3 的贷款因右删失仅含违约样本，不能直接用于测试，如有需要可对正常样本做截断处理后纳入

## 特征工程对照表

| 输出列 | 原始 Freddie Mac 字段 | 转换方式 |
|---|---|---|
| `Loanref` | `Loan_Sequence_Number` | 直接映射 |
| `Credit_Score` | `Credit_Score` | 转数值，筛选 300–753 |
| `Mortgage_Insurance` | `Mortgage_Insurance_Percentage` | 转数值 ÷ 100，999 填 0 |
| `Number_of_units` | `Number_of_Units` | 转数值，9 填 0 |
| `CLoan_to_value` | `Original_Combined_Loan_To_Value_CLTV` | 转数值，999 填 0 |
| `Debt_to_income` | `Original_Debt_To_Income_DTI_Ratio` | 转数值 |
| `OLoan_to_value` | `Original_Loan_To_Value_LTV` | 转数值，999 填 0 |
| `Single_borrower` | `Number_of_Borrowers` | `== "01"` → 1, 否则 0 |
| `is_Loan_purpose_*` | `Loan_Purpose` | One-Hot: P / C / N |
| `is_First_time_homeowner*` | `First_Time_Homebuyer_Flag` | One-Hot: Y / N / miss |
| `is_Occupancy_status_*` | `Occupancy_Status` | One-Hot: P / I / S |
| `is_Origination_channel_*` | `Channel` | One-Hot: R / B / C |
| `is_Property_type_*` | `Property_Type` | One-Hot: CO / CP / MH / PU / SF |
| `is_property_state_*` | `Property_State` | One-Hot: 52 个州/地区代码 |
| `DFlag` | `Current_Loan_Delinquency_Status` + `Zero_Balance_Code` | 24m 窗口内 delinq ≥ 3 或 ZBC ∈ {03, 09} |

## 与论文数据集对比

| 指标 | 论文 FM60 | 论文 FM36 | 论文 FM12 | 本数据集 FM24 |
|---|---|---|---|---|
| 样本量 | 649,608 | — | 231,065 | 1,022,701 |
| 违约率 | ~2.84% | — | ~0.26% | 4.13% |
| 不平衡比 (IR) | ~34 | ~165 | ~390 | **23** |
| 特征数 | 80 | 80 | 80 | 80 |
| 观测窗口 | 60 个月 | 36 个月 | 12 个月 | 24 个月 |
| 数据年份 | 1999–2015 | 1999–2015 | 1999–2015 | 2022–2025 |
| 贷款机构 | Freddie Mac | Freddie Mac | Freddie Mac | Freddie Mac |

> 本数据集的 IR（23）低于论文所有三个窗口的 IR，属于**轻度类不平衡**（IR < 30）。按照论文结论，此类场景下 CE loss + GEV link 的 XGBoost 变体表现最优，且 WGAN-GP 生成违约样本的边际收益可能受限——论文指出 GAN 类方法在 IR < 50 时相对于普通重采样方法的优势有限。

## 参考文献

Mushava, J., & Murray, M. (2024). Flexible loss functions for binary classification in gradient-boosted decision trees: An application to credit scoring. *Expert Systems With Applications*, 238, 121876.
