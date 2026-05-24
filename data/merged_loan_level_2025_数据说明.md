# merged_loan_level_2025.csv 数据集说明

## 数据来源

数据源自 **Freddie Mac Single-Family Loan-Level Dataset**（房地美单户贷款级数据集），原始文件为管道符(`|`)分隔的 `.txt` 文件，按季度发布，包含两部分：

- **Origination（发起数据）**：贷款发起时的静态信息，每笔贷款一行
- **Performance（表现数据）**：贷款逐月动态表现记录，每笔贷款每月一行

## 数据处理流程

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 1 | `convert_to_csv.py` | 将原始 `.txt` 转为带列名的 `.csv` |
| 2 | `merge_data.py` | 合并所有季度文件 → `all_origination_2023_2025.csv` + `all_performance_2023_2025.csv` |
| 3 | `drop_2023.py` | 删除 2023 年贷款，仅保留 2024–2025 年 |
| 4 | `aggregate_performance.py` | 将 performance 面板数据按贷款聚合为一行（数值取 max，日期取最近值），与 origination 一对一合并 → `merged_loan_level.csv` |

`merged_loan_level_2025.csv` 为 `merged_loan_level.csv` 的子集，仅包含 **2025 年发起的贷款**。

## 数据集规模

| 指标 | 数值 |
|---|---|
| 贷款数 | **812,873** 笔 |
| 列数 | **63** 列 |
| 文件大小 | 约 **213 MB** |
| 数据报告期 | 2025 年 9 月（`Monthly_Reporting_Period = 202509`） |

## 列名说明

### 一、贷款发起信息（Origination）

| 列名 | 类型 | 说明 |
|---|---|---|
| `Credit_Score` | int | 借款人信用评分（301–850，9999=缺失） |
| `First_Payment_Date` | str | 首次还款日期（YYYYMM） |
| `First_Time_Homebuyer_Flag` | str | 首次购房者标识（Y/N/9） |
| `Maturity_Date` | str | 贷款到期日（YYYYMM） |
| `Metropolitan_Statistical_Area` | str | 大都市统计区代码（MSA，2010 年普查标准） |
| `Mortgage_Insurance_Percentage` | str | 按揭保险覆盖率（000–099，999=缺失） |
| `Number_of_Units` | str | 房产单元数（1–4，9=缺失） |
| `Occupancy_Status` | str | 入住状态：P=自住，I=投资，S=第二居所，9=缺失 |
| `Original_Combined_Loan_To_Value_CLTV` | str | 合并贷款价值比（含二押等，1–200，999=缺失） |
| `Original_Debt_To_Income_DTI_Ratio` | str | 债务收入比（1–65，999=缺失） |
| `Original_UPB` | str | 原始未偿还本金余额（贷款金额） |
| `Original_Loan_To_Value_LTV` | str | 贷款价值比（6–135，999=缺失） |
| `Original_Interest_Rate` | str | 原始票面利率 |
| `Channel` | str | 贷款渠道：R=零售，B=经纪人，C=代理，9=缺失 |
| `Prepayment_Penalty_Mortgage_Flag` | str | 提前还款罚金标识（Y/N） |
| `Amortization_Type` | str | 还款方式：FRM=固定利率，ARM=可调利率 |
| `Property_State` | str | 房产所在州（两字母缩写） |
| `Property_Type` | str | 房产类型：CO=公寓，CP=合作公寓，MH=移动房屋，PU=联排，SF=独栋 |
| `Postal_Code` | str | 邮政编码（前 3 位） |
| `Loan_Sequence_Number` | str | 贷款唯一标识（如 F23Q10186379） |
| `Loan_Purpose` | str | 贷款目的：N=无现金再融资，P=购房，C=套现再融资 |
| `Original_Loan_Term` | str | 原始贷款期限（月数，如 360=30 年，180=15 年） |
| `Number_of_Borrowers` | str | 借款人数（01–04，09=缺失） |
| `Seller_Name` | str | 贷款发起机构（卖方） |
| `Servicer_Name` | str | 贷款服务机构 |
| `Super_Conforming_Flag` | str | 超级合规贷款标识（Y/N/9） |
| `Pre_Relief_Refinance_Loan_Sequence_Number` | str | 救济再融资前贷款 ID |
| `Special_Eligibility_Program` | str | 特殊资格计划：9=无，H=HFA/HomeReady |
| `Relief_Refinance_Indicator` | str | 救济再融资标识 |
| `Property_Valuation_Method` | str | 房产估值方式（2=全面评估等） |
| `Interest_Only_Indicator` | str | 只还利息贷款标识（Y/N） |
| `MI_Cancellation_Indicator` | str | 按揭保险取消标识 |

### 二、贷后表现信息（Performance，聚合后）

| 列名 | 类型 | 说明 |
|---|---|---|
| `Current_Actual_UPB` | float | 当前实际未偿还本金余额 |
| `Current_Loan_Delinquency_Status` | float | 贷款逾期状态：0=正常，1=30天，2=60天，3=90天+ |
| `Loan_Age` | float | 贷款已存续月数 |
| `Remaining_Months_to_Legal_Maturity` | float | 距离法定到期剩余月数 |
| `Current_Interest_Rate` | float | 当前利率 |
| `Current_Non_Interest_Bearing_UPB` | float | 当前不计息 UPB 部分 |
| `Interest_Bearing_UPB` | float | 计息 UPB 部分 |
| `Estimated_Loan_To_Value_ELTV` | float | 当前估计贷款价值比（999=缺失） |
| `Monthly_Reporting_Period` | str | 最近数据报告月份（YYYYMM） |
| `Zero_Balance_Code` | str | 零余额原因码（03=止赎，09=其他等） |
| `Zero_Balance_Effective_Date` | str | 零余额生效日期 |
| `Zero_Balance_Removal_UPB` | float | 零余额事件时移除的 UPB |
| `Due_Date_of_Last_Paid_Installment_DDLPI` | str | 最后一次还款到期日 |
| `Delinquent_Accrued_Interest` | float | 逾期应计利息 |
| `Modification_Flag` | str | 贷款修改标识 |
| `Step_Modification_Flag` | str | 阶梯利率修改标识 |
| `Payment_Deferral` | str | 还款延期标识 |
| `Delinquency_Due_to_Disaster` | str | 自然灾害导致逾期标识 |
| `Borrower_Assistance_Status_Code` | str | 借款人救助计划状态 |
| `Defect_Settlement_Date` | str | 缺陷结算日期 |

### 三、损失与回收信息

| 列名 | 类型 | 说明 |
|---|---|---|
| `MI_Recoveries` | float | 按揭保险回收额 |
| `Net_Sale_Proceeds` | float | 资产处置净收益 |
| `Non_MI_Recoveries` | float | 非按揭保险回收额 |
| `Total_Expenses` | float | 处置总费用 |
| `Legal_Costs` | float | 法律费用 |
| `Maintenance_and_Preservation_Costs` | float | 维护保养费 |
| `Taxes_and_Insurance` | float | 税费及保险垫付 |
| `Miscellaneous_Expenses` | float | 其他费用 |
| `Actual_Loss_Calculation` | float | 实际净损失 |
| `Cumulative_Modification_Cost` | float | 累计贷款修改成本 |
| `Current_Month_Modification_Cost` | float | 当月修改成本 |

### 四、衍生字段

| 列名 | 类型 | 说明 |
|---|---|---|
| `Default_Flag` | int | **违约标识（0/1）**，定义：最严重逾期 >= 3（90天+）或零余额原因为止赎/其他处置（03/09） |

## 关键特点

1. **一对一结构**：每行代表一笔独立贷款，performance 的时序信息已聚合为汇总值
2. **Performance 聚合逻辑**：数值型取 `max`（如最严重逾期状态、最大损失额），日期型取最近值
3. **仅含 2025 年发起的贷款**：`First_Payment_Date` 均在 2025 年之后
4. **低违约率**：由于贷款均为近期发起（2025年），违约样本较少，适合用于信用风险建模的早期预警研究
5. **缺失值编码**：部分发起字段以 `999`、`9999` 或 `9` 表示缺失/不适用，需在建模前处理
