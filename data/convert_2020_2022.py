"""仅转换 2020-2022 的 TXT -> CSV，跳过已存在的 CSV"""
import os, sys
import pandas as pd

data_root = os.path.dirname(os.path.abspath(__file__))

ORIGINATION_COLUMNS = [
    'Credit_Score','First_Payment_Date','First_Time_Homebuyer_Flag','Maturity_Date',
    'Metropolitan_Statistical_Area','Mortgage_Insurance_Percentage','Number_of_Units',
    'Occupancy_Status','Original_Combined_Loan_To_Value_CLTV','Original_Debt_To_Income_DTI_Ratio',
    'Original_UPB','Original_Loan_To_Value_LTV','Original_Interest_Rate','Channel',
    'Prepayment_Penalty_Mortgage_Flag','Amortization_Type','Property_State','Property_Type',
    'Postal_Code','Loan_Sequence_Number','Loan_Purpose','Original_Loan_Term',
    'Number_of_Borrowers','Seller_Name','Servicer_Name','Super_Conforming_Flag',
    'Pre_Relief_Refinance_Loan_Sequence_Number','Special_Eligibility_Program',
    'Relief_Refinance_Indicator','Property_Valuation_Method','Interest_Only_Indicator',
    'MI_Cancellation_Indicator',
]

PERFORMANCE_COLUMNS = [
    'Loan_Sequence_Number','Monthly_Reporting_Period','Current_Actual_UPB',
    'Current_Loan_Delinquency_Status','Loan_Age','Remaining_Months_to_Legal_Maturity',
    'Defect_Settlement_Date','Modification_Flag','Zero_Balance_Code',
    'Zero_Balance_Effective_Date','Current_Interest_Rate','Current_Non_Interest_Bearing_UPB',
    'Due_Date_of_Last_Paid_Installment_DDLPI','MI_Recoveries','Net_Sale_Proceeds',
    'Non_MI_Recoveries','Total_Expenses','Legal_Costs','Maintenance_and_Preservation_Costs',
    'Taxes_and_Insurance','Miscellaneous_Expenses','Actual_Loss_Calculation',
    'Cumulative_Modification_Cost','Step_Modification_Flag','Payment_Deferral',
    'Estimated_Loan_To_Value_ELTV','Zero_Balance_Removal_UPB','Delinquent_Accrued_Interest',
    'Delinquency_Due_to_Disaster','Borrower_Assistance_Status_Code',
    'Current_Month_Modification_Cost','Interest_Bearing_UPB',
]

quarters = ['2020Q1','2020Q2','2020Q3','2020Q4',
            '2021Q1','2021Q2','2021Q3','2021Q4',
            '2022Q1','2022Q2','2022Q3','2022Q4']

for qtr in quarters:
    d = os.path.join(data_root, f'historical_data_{qtr}')

    # Origination
    txt_f = os.path.join(d, f'historical_data_{qtr}.txt')
    csv_f = os.path.join(d, f'historical_data_{qtr}.csv')
    if os.path.exists(txt_f) and not os.path.exists(csv_f):
        print(f'Converting origination: {qtr} ...', flush=True)
        df = pd.read_csv(txt_f, sep='|', header=None, names=ORIGINATION_COLUMNS,
                         dtype=str, low_memory=False, encoding='utf-8')
        df.to_csv(csv_f, index=False, encoding='utf-8-sig')
        print(f'  -> {len(df):,} loans, {os.path.getsize(csv_f)/1024**2:.1f} MB', flush=True)
    elif os.path.exists(csv_f):
        print(f'Skip origination: {qtr} (CSV exists)', flush=True)

    # Performance
    txt_f = os.path.join(d, f'historical_data_time_{qtr}.txt')
    csv_f = os.path.join(d, f'historical_data_time_{qtr}.csv')
    if os.path.exists(txt_f) and not os.path.exists(csv_f):
        print(f'Converting performance: {qtr} ...', flush=True)
        n = 0
        first = True
        for chunk in pd.read_csv(txt_f, sep='|', header=None, names=PERFORMANCE_COLUMNS,
                                 dtype=str, low_memory=False, encoding='utf-8',
                                 chunksize=300_000):
            chunk.to_csv(csv_f, mode='w' if first else 'a', header=first,
                        index=False, encoding='utf-8-sig')
            first = False
            n += len(chunk)
            if n % 3_000_000 == 0:
                print(f'    {n:,} rows ...', flush=True)
        print(f'  -> {n:,} rows, {os.path.getsize(csv_f)/1024**2:.1f} MB', flush=True)
    elif os.path.exists(csv_f):
        print(f'Skip performance: {qtr} (CSV exists)', flush=True)

print('\nDone converting 2020-2022.', flush=True)
