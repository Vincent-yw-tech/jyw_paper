"""
Freddie Mac Single-Family Loan-Level Dataset: TXT → CSV converter.
Reads pipe-delimited .txt files and writes .csv files with column headers.
Supports both origination and monthly performance files.
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Column layouts – Freddie Mac SF LLD (2021+ format, 32 columns per file type)
# ---------------------------------------------------------------------------

ORIGINATION_COLUMNS = [
    "Credit_Score",                                    # 0   Borrower credit score (301-850, 9999=N/A)
    "First_Payment_Date",                              # 1   YYYYMM of first scheduled payment
    "First_Time_Homebuyer_Flag",                       # 2   Y / N / 9
    "Maturity_Date",                                   # 3   YYYYMM of final payment
    "Metropolitan_Statistical_Area",                   # 4   MSA / MD code (2010 census)
    "Mortgage_Insurance_Percentage",                   # 5   MI coverage % (000-099, 999=N/A)
    "Number_of_Units",                                 # 6   1-4, 9=N/A
    "Occupancy_Status",                                # 7   P=Primary, I=Investment, S=Second, 9=N/A
    "Original_Combined_Loan_To_Value_CLTV",            # 8   Combined LTV at origination (1-200, 999=N/A)
    "Original_Debt_To_Income_DTI_Ratio",               # 9   Debt-to-Income ratio (1-65, 999=N/A)
    "Original_UPB",                                    # 10  Unpaid Principal Balance at origination
    "Original_Loan_To_Value_LTV",                      # 11  Loan-to-Value at origination (6-135, 999=N/A)
    "Original_Interest_Rate",                          # 12  Note rate at origination
    "Channel",                                         # 13  R=Retail, B=Broker, C=Correspondent, 9=N/A
    "Prepayment_Penalty_Mortgage_Flag",                # 14  Y / N
    "Amortization_Type",                               # 15  FRM / ARM
    "Property_State",                                  # 16  Two-letter US state code
    "Property_Type",                                   # 17  CO, CP, MH, PU, SF
    "Postal_Code",                                     # 18  First 3 digits of ZIP
    "Loan_Sequence_Number",                            # 19  Unique loan ID (e.g. F24Q10000001)
    "Loan_Purpose",                                    # 20  P=Purchase, C=Cash-out Refi, N=No-cash Refi
    "Original_Loan_Term",                              # 21  Term in months (e.g. 360)
    "Number_of_Borrowers",                             # 22  01-04, 09=N/A
    "Seller_Name",                                     # 23  Loan originator
    "Servicer_Name",                                   # 24  Loan servicer
    "Super_Conforming_Flag",                           # 25  Y / N / 9
    "Pre_Relief_Refinance_Loan_Sequence_Number",       # 26  Pre-HARP/relief refinance loan ID
    "Special_Eligibility_Program",                     # 27  9=N/A, H=HFA/HomeReady
    "Relief_Refinance_Indicator",                      # 28  Relief refinance indicator
    "Property_Valuation_Method",                       # 29  Appraisal valuation method code
    "Interest_Only_Indicator",                         # 30  Y / N
    "MI_Cancellation_Indicator",                       # 31  Mortgage insurance cancellation indicator
]

PERFORMANCE_COLUMNS = [
    "Loan_Sequence_Number",                         # 0   Unique loan ID
    "Monthly_Reporting_Period",                     # 1   YYYYMM of observation
    "Current_Actual_UPB",                           # 2   Current unpaid principal balance
    "Current_Loan_Delinquency_Status",              # 3   0=current, 1=30d, 2=60d, 3=90d, etc.
    "Loan_Age",                                     # 4   Months since origination
    "Remaining_Months_to_Legal_Maturity",           # 5   Months until scheduled maturity
    "Defect_Settlement_Date",                       # 6   Defect settlement date
    "Modification_Flag",                            # 7   Loan modification indicator
    "Zero_Balance_Code",                            # 8   Code for zero-balance reason
    "Zero_Balance_Effective_Date",                  # 9   YYYYMM when zero balance took effect
    "Current_Interest_Rate",                        # 10  Current note rate
    "Current_Non_Interest_Bearing_UPB",             # 11  Non-interest-bearing UPB portion
    "Due_Date_of_Last_Paid_Installment_DDLPI",      # 12  DDLPI
    "MI_Recoveries",                                # 13  Mortgage insurance recoveries
    "Net_Sale_Proceeds",                            # 14  Net proceeds from property sale
    "Non_MI_Recoveries",                            # 15  Non-MI recoveries
    "Total_Expenses",                               # 16  Total disposition expenses
    "Legal_Costs",                                  # 17  Legal fees
    "Maintenance_and_Preservation_Costs",           # 18  Property maintenance costs
    "Taxes_and_Insurance",                          # 19  Tax & insurance advances
    "Miscellaneous_Expenses",                       # 20  Other expenses
    "Actual_Loss_Calculation",                      # 21  Net loss on disposition
    "Cumulative_Modification_Cost",                 # 22  Cumulative cost of loan modification
    "Step_Modification_Flag",                       # 23  Step-rate modification indicator
    "Payment_Deferral",                             # 24  Payment deferral indicator
    "Estimated_Loan_To_Value_ELTV",                 # 25  Current estimated LTV (999=N/A)
    "Zero_Balance_Removal_UPB",                     # 26  UPB removed at zero balance event
    "Delinquent_Accrued_Interest",                  # 27  Delinquent accrued interest amount
    "Delinquency_Due_to_Disaster",                  # 28  Delinquency due to natural disaster flag
    "Borrower_Assistance_Status_Code",              # 29  Borrower assistance plan status code
    "Current_Month_Modification_Cost",              # 30  Current month modification cost
    "Interest_Bearing_UPB",                         # 31  Interest-bearing portion of UPB
]


def detect_file_type(filename):
    """Return 'origination' or 'performance' based on filename."""
    base = os.path.basename(filename)
    if "_time_" in base:
        return "performance"
    return "origination"


def convert_file(src_path, dst_path, columns, chunksize=200_000):
    """Convert a pipe-delimited .txt to .csv with headers, using chunked reads."""
    print(f"  Converting: {os.path.basename(src_path)}")
    try:
        # First pass: read all and write CSV
        # For small files read at once; for large files use chunking
        file_size_mb = os.path.getsize(src_path) / (1024 * 1024)

        if file_size_mb < 200:
            # Small enough to read in one go
            df = pd.read_csv(
                src_path,
                sep="|",
                header=None,
                names=columns,
                dtype=str,
                low_memory=False,
                encoding="utf-8",
            )
            df.to_csv(dst_path, index=False, encoding="utf-8-sig")
        else:
            # Large files – chunk
            first_chunk = True
            for chunk in pd.read_csv(
                src_path,
                sep="|",
                header=None,
                names=columns,
                dtype=str,
                low_memory=False,
                encoding="utf-8",
                chunksize=chunksize,
            ):
                chunk.to_csv(
                    dst_path,
                    index=False,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    encoding="utf-8-sig",
                )
                first_chunk = False
                print(f"    ... processed chunk ({len(chunk)} rows)")

        print(f"    -> {os.path.basename(dst_path)}  "
              f"({os.path.getsize(dst_path) / (1024 * 1024):.1f} MB)")
    except Exception as exc:
        print(f"    ERROR: {exc}")


def main():
    data_root = os.path.dirname(os.path.abspath(__file__))
    txt_files = []

    # Collect all .txt files under data/
    for dirpath, _, filenames in os.walk(data_root):
        for fn in filenames:
            if fn.endswith(".txt"):
                txt_files.append(os.path.join(dirpath, fn))

    if not txt_files:
        print("No .txt files found under", data_root)
        sys.exit(1)

    print(f"Found {len(txt_files)} .txt file(s) to convert.\n")

    for i, txt_path in enumerate(sorted(txt_files), 1):
        ftype = detect_file_type(txt_path)
        columns = ORIGINATION_COLUMNS if ftype == "origination" else PERFORMANCE_COLUMNS
        csv_path = txt_path.replace(".txt", ".csv")

        print(f"[{i}/{len(txt_files)}] ({ftype})")
        convert_file(txt_path, csv_path, columns)

    print("\nDone. All files converted to CSV.")


if __name__ == "__main__":
    main()
