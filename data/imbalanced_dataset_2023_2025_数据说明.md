# imbalanced_dataset_2023_2025.csv 数据说明

## 概述

本数据集按照 **Mushava & Murray (2024)** "Flexible loss functions for binary classification in gradient-boosted decision trees: An application to credit scoring"（*Expert Systems With Applications*, 238, 121876）的方法论构建，用于信用评分二分类任务，**未做任何重采样/平衡处理**。

## 数据来源

- **原始数据**：Freddie Mac Single-Family Loan-Level Dataset（房地美单户贷款级数据集）
- **时间范围**：2023Q1–2025Q3，共 11 个季度
- **处理脚本**：`build_imbalanced_dataset.py`

## 数据集规模

| 指标 | 数值 |
|---|---|
| 样本数 | **688,592** 笔贷款 |
| 特征数 | **79** 个预测变量 + 1 个目标变量 = **80 列** |
| 文件大小 | **117.7 MB** |
| 违约样本 | **9,000**（1.31%） |
| 正常样本 | **679,592**（98.69%） |
| 不平衡比 (IR) | **75** |

## 数据处理流程

| 步骤 | 说明 | 结果 |
|---|---|---|
| 1. 合并 | 合并 11 个季度的 origination 与 performance 数据 | 2,623,132 笔贷款 |
| 2. 违约判定 | 12 个月观测窗口内：逾期 ≥ 90天 或 止赎/处置 | 10,885 笔违约 |
| 3. 信用分筛选 | 剔除 Credit_Score ≥ 754 的样本（含缺失值 9999） | 保留 1,138,250 笔 |
| 4. 观测期筛选 | 仅保留违约或观测 ≥ 12 个月的样本 | 保留 688,592 笔 |
| 5. 特征工程 | 数值特征 + 分类变量 One-Hot 编码 | 80 列输出 |

### 两个核心筛选逻辑

**1. 信用分截断（Credit_Score < 754）**

论文依据：信用评分的信息值（IV）远大于 0.5，属于"过度预测"变量。若保留高分样本，模型可能退化为仅依赖信用分的单特征模型，失去对其他特征的敏感性。论文的 FM60 数据集（649,608 笔贷款）全部为信用分低于 754 的贷款。

**2. 完整观测期（12 个月窗口）**

违约标签需要在固定的观测窗口内才能准确标注：
- **Default_Flag = 1**：贷款在发放后 12 个月内出现严重逾期（≥ 90 天）或被止赎/处置（Zero_Balance_Code = 03/09）
- **Default_Flag = 0**：贷款至少被观测了 12 个月且未发生上述事件
- 观测不足 12 个月且未违约的贷款被**排除**

> 论文中使用了 12、36、60 个月三种窗口（FM12/FM36/FM60）。鉴于 2023–2025 年数据的最新报告期为 2025 年 9 月，最大观测时长仅约 32 个月，因此采用 FM12（12 个月）窗口。

## 列名说明

### 一、数值特征（8 列）

| 列名 | 来源字段 | 类型 | 说明 |
|---|---|---|---|
| `Loanref` | `Loan_Sequence_Number` | str | 贷款唯一标识，Freddie Mac 格式（如 F23Q10000002，F=Freddie, 23=2023年, Q1=Q1季度） |
| `Credit_Score` | `Credit_Score` | int | 借款人信用评分（FICO），原始范围 301–850（9999=缺失），本数据集截断为 **< 754** |
| `Mortgage_Insurance` | `Mortgage_Insurance_Percentage` | float | 按揭保险覆盖率，**已从原始整数转为百分比小数**（原始 000–099 表示 0%–99%，999=缺失） |
| `Number_of_units` | `Number_of_Units` | int | 房产单元数：1=独栋单户，2=双户，3=三户，4=四户，9=缺失 |
| `CLoan_to_value` | `Original_Combined_Loan_To_Value_CLTV` | float | 合并贷款价值比（Combined LTV），含二押等次贷在内的总贷款额 / 房产价值 × 100，范围 1–200（999=缺失） |
| `Debt_to_income` | `Original_Debt_To_Income_DTI_Ratio` | float | 债务收入比（DTI），月债务支出 / 月收入 × 100，范围 1–65（999=缺失），越高还款压力越大 |
| `OLoan_to_value` | `Original_Loan_To_Value_LTV` | float | 第一抵押贷款价值比（LTV），仅首贷金额 / 房产价值 × 100，范围 6–135（999=缺失） |
| `Single_borrower` | `Number_of_Borrowers` → 衍生 | int | 是否只有 1 个借款人：`Number_of_Borrowers == "01"` → 1，否则 0。论文发现单一借款人违约率更高 |

### 二、One-Hot 编码特征（72 列）

所有 one-hot 编码的特征取值均为 **0（否）** 或 **1（是）**。每条记录在每组内**有且仅有一列为 1**（含缺失/异常值类别）。

#### 2.1 贷款目的（3 列）— 来源：`Loan_Purpose`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_Loan_purpose_purc` | P（Purchase） | 购房贷款 |
| `is_Loan_purpose_cash` | C（Cash-out Refinance） | 套现再融资（借新还旧并提取现金） |
| `is_Loan_purpose_noca` | N（No-cash Refinance） | 无现金再融资（仅借新还旧，不提取现金） |

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
| `is_Occupancy_status_inve` | I（Investment Property） | 投资房（出租等） |
| `is_Occupancy_status_seco` | S（Second Home） | 第二居所（度假房等） |

> 自住房（P）通常违约风险最低，投资房（I）风险最高。

#### 2.4 贷款渠道（3 列）— 来源：`Channel`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_Origination_channel_reta` | R（Retail） | 零售渠道（银行直接面向客户） |
| `is_Origination_channel_brok` | B（Broker） | 经纪人渠道（通过贷款经纪人中介） |
| `is_Origination_channel_corr` | C（Correspondent） | 代理渠道（代理机构承销后转售） |

#### 2.5 房产类型（5 列）— 来源：`Property_Type`

| 列名 | 原始值 | 含义 |
|---|---|---|
| `is_Property_type_cond` | CO（Condominium） | 公寓（共有产权） |
| `is_Property_type_coop` | CP（Cooperative） | 合作公寓（股份产权） |
| `is_Property_type_manu` | MH（Manufactured Housing） | 预制/移动房屋 |
| `is_Property_type_pud` | PU（Planned Unit Development） | 规划单元开发（联排别墅等，有 HOA） |
| `is_Property_type_sing` | SF（Single-Family） | 独栋住宅 |

> CO/CP/MH 属于非传统房产类型，通常违约风险较高。

#### 2.6 房产所在州（52 列）— 来源：`Property_State`

共 52 列，命名格式为 `is_property_state_XX`，其中 `XX` 为两位州代码：

| 代码 | 州/地区 | 代码 | 州/地区 | 代码 | 州/地区 |
|---|---|---|---|---|---|
| AK | 阿拉斯加 | ME | 缅因 | SC | 南卡罗来纳 |
| AL | 阿拉巴马 | MI | 密歇根 | SD | 南达科他 |
| AR | 阿肯色 | MN | 明尼苏达 | TN | 田纳西 |
| AZ | 亚利桑那 | MO | 密苏里 | TX | **得克萨斯**（贷款量大） |
| CA | **加利福尼亚**（贷款量大） | MS | 密西西比 | UT | 犹他 |
| CO | 科罗拉多 | MT | 蒙大拿 | VA | 弗吉尼亚 |
| CT | 康涅狄格 | NC | 北卡罗来纳 | VI | 维尔京群岛 |
| DC | 哥伦比亚特区 | ND | 北达科他 | VT | 佛蒙特 |
| DE | 特拉华 | NE | 内布拉斯加 | WA | 华盛顿 |
| FL | **佛罗里达**（贷款量大） | NH | 新罕布什尔 | WI | 威斯康星 |
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

> 各州经济环境、房价走势、司法程序（追索权/非追索权州）不同，对违约率有显著影响。

### 三、目标变量（1 列）

| 列名 | 类型 | 说明 |
|---|---|---|
| `DFlag` | int | **违约标识**。1 = 贷款在发放后 **12 个月内**出现严重逾期（≥ 90 天，即 `Current_Loan_Delinquency_Status ≥ 3`）或被止赎/处置（`Zero_Balance_Code = 03` 止赎、`09` 其他处置）；0 = 贷款已被观测 ≥ 12 个月且未违约 |

### 四、特征工程对照表

| 输出列 | 原始 Freddie Mac 字段 | 转换方式 |
|---|---|---|
| `Loanref` | `Loan_Sequence_Number` | 直接映射 |
| `Credit_Score` | `Credit_Score` | 转数值，筛选 < 754 |
| `Mortgage_Insurance` | `Mortgage_Insurance_Percentage` | 转数值 ÷ 100 |
| `Number_of_units` | `Number_of_Units` | 转数值 |
| `CLoan_to_value` | `Original_Combined_Loan_To_Value_CLTV` | 转数值 |
| `Debt_to_income` | `Original_Debt_To_Income_DTI_Ratio` | 转数值 |
| `OLoan_to_value` | `Original_Loan_To_Value_LTV` | 转数值 |
| `Single_borrower` | `Number_of_Borrowers` | `== "01"` → 1, 否则 0 |
| `is_Loan_purpose_*` | `Loan_Purpose` | One-Hot: P / C / N |
| `is_First_time_homeowner*` | `First_Time_Homebuyer_Flag` | One-Hot: Y / N / miss |
| `is_Occupancy_status_*` | `Occupancy_Status` | One-Hot: P / I / S |
| `is_Origination_channel_*` | `Channel` | One-Hot: R / B / C |
| `is_Property_type_*` | `Property_Type` | One-Hot: CO / CP / MH / PU / SF |
| `is_property_state_*` | `Property_State` | One-Hot: 52 个州/地区代码 |
| `DFlag` | `Current_Loan_Delinquency_Status` + `Zero_Balance_Code` | 12m 窗口内 delinq≥3 或 ZBC∈{03,09} |

## 缺失值处理

| 字段 | 原始缺失编码 | 处理方式 |
|---|---|---|
| `Credit_Score` | 9999 | 已在筛选阶段排除（9999 ≥ 754） |
| `Mortgage_Insurance` | 999 | 填充为 0 |
| `CLoan_to_value`, `OLoan_to_value` | 999 | 填充为 0 |
| `Debt_to_income` | 999 | 填充为 0 |
| `Number_of_units` | 9 | 填充为 0 |
| `Loan_Purpose` | 9 / 空 | 未命中 P/C/N 的自然归 0（不激活任何列），XGBoost 可自行处理 |
| `First_Time_Homebuyer_Flag` | 9 / 空 | 归入 `is_First_time_homeowner_miss` |
| `Occupancy_Status` | 9 / 空 | 未命中 P/I/S 的自然归 0 |
| `Channel` | 9 / 空 | 未命中 R/B/C 的自然归 0 |
| `Property_Type` | 空 | 未命中已知类型的自然归 0 |

## 与论文数据集对比

| 指标 | 论文 FM60 | 论文 FM12 | 本数据集 |
|---|---|---|---|
| 样本量 | 649,608 | 231,065 | 688,592 |
| 违约率 | ~2.84% | ~0.26% | 1.31% |
| IR | ~34 | ~390 | 75 |
| 特征数 | 79+1 | 79+1 | 79+1 |
| 观测窗口 | 60 个月 | 12 个月 | 12 个月 |
| 数据年份 | 早期（FM 标准数据集） | 同左 | 2023–2025 |

本数据集的 IR（75）介于论文 FM36（IR≈165）和 FM60（IR≈34）之间，属于**非严重**类不平衡（IR < 99），按照论文结论，在此场景下 CE loss + GEV link 的 XGBoost 变体表现最优。

## 参考文献

Mushava, J., & Murray, M. (2024). Flexible loss functions for binary classification in gradient-boosted decision trees: An application to credit scoring. *Expert Systems With Applications*, 238, 121876.
