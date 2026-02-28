import os
import pandas as pd
import re

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_DIR  = "Data"
OUTPUT_DIR = "Data Preparation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# FILE 1 ─ rep_s_00150.csv
# Customer Orders (Delivery) – by Branch
# ══════════════════════════════════════════════
def process_file1():
    filepath = os.path.join(INPUT_DIR, "rep_s_00150.csv")
    rows = []
    current_branch = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    branch_keywords = {"Conut - Tyre", "Conut", "Conut Jnah", "Main Street Coffee"}

    for line in lines:
        stripped = line.strip()

        # Detect branch name rows
        if stripped.rstrip(",") in branch_keywords:
            current_branch = stripped.rstrip(",")
            continue

        # Skip header/footer/page lines
        if any(kw in stripped for kw in [
            "Customer Name", "From Date", "To Date", "Page",
            "REP_S_00150", "Copyright", "www.", "Total By Branch",
            "Customer Orders", "30-Jan-26"
        ]):
            continue

        parts = [p.strip().strip('"') for p in stripped.split(",")]
        parts = [p for p in parts if p != ""]

        # Expect: Name, Address(maybe), Phone, FirstOrder, LastOrder, Total, NumOrders
        if len(parts) >= 5 and re.match(r"Person_\d+", parts[0]):
            name         = parts[0]
            phone        = parts[1] if len(parts) > 1 else ""
            first_order  = parts[2] if len(parts) > 2 else ""
            last_order   = parts[3] if len(parts) > 3 else ""
            total        = parts[4] if len(parts) > 4 else ""
            num_orders   = parts[5] if len(parts) > 5 else ""

            rows.append({
                "Branch":         current_branch,
                "Customer_Name":  name,
                "Phone_Number":   phone,
                "First_Order":    first_order,
                "Last_Order":     last_order,
                "Total_Revenue":  total,
                "Num_Orders":     num_orders
            })

    df = pd.DataFrame(rows, columns=[
        "Branch", "Customer_Name", "Phone_Number",
        "First_Order", "Last_Order", "Total_Revenue", "Num_Orders"
    ])
    out = os.path.join(OUTPUT_DIR, "rep_s_00150_labelled.csv")
    df.to_csv(out, index=False)
    print(f"[✓] File 1 saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# FILE 2 ─ REP_S_00136_SMRY.csv
# Summary By Division (Delivery / Table / Take Away / Total)
# ══════════════════════════════════════════════
def process_file2():
    filepath = os.path.join(INPUT_DIR, "REP_S_00136_SMRY.csv")
    rows = []
    current_branch = None

    branch_names = {"Conut", "Conut - Tyre", "Conut Jnah", "Main Street Coffee"}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = [p.strip().strip('"') for p in stripped.split(",")]
        parts = [p for p in parts if p]

        # Detect branch row (first non-empty token is a branch name)
        if parts and parts[0] in branch_names:
            current_branch = parts[0]
            # Check if rest of row has division data
            if len(parts) >= 5:
                rows.append({
                    "Branch":    current_branch,
                    "Division":  parts[1] if len(parts) > 1 else "",
                    "Delivery":  parts[2] if len(parts) > 2 else "0",
                    "Table":     parts[3] if len(parts) > 3 else "0",
                    "Take_Away": parts[4] if len(parts) > 4 else "0",
                    "Total":     parts[5] if len(parts) > 5 else "0"
                })
            continue

        # Skip non-data rows
        if any(kw in stripped for kw in [
            "Summary By Division", "From Date", "DELIVERY", "TABLE",
            "REP_S_00136", "Copyright", "www.", "30-Jan-26", "Year:"
        ]):
            continue

        # Division data rows: ,Division,Delivery,Table,TakeAway,Total
        if current_branch and len(parts) >= 5:
            rows.append({
                "Branch":    current_branch,
                "Division":  parts[0],
                "Delivery":  parts[1],
                "Table":     parts[2],
                "Take_Away": parts[3],
                "Total":     parts[4]
            })

    df = pd.DataFrame(rows, columns=[
        "Branch", "Division", "Delivery", "Table", "Take_Away", "Total"
    ])
    out = os.path.join(OUTPUT_DIR, "REP_S_00136_SMRY_labelled.csv")
    df.to_csv(out, index=False)
    print(f"[✓] File 2 saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# FILE 3 ─ rep_s_00191_SMRY.csv
# Sales by Items By Group
# ══════════════════════════════════════════════
def process_file3():
    filepath = os.path.join(INPUT_DIR, "rep_s_00191_SMRY.csv")
    rows = []
    current_branch   = None
    current_division = None
    current_group    = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Branch detection
        if stripped.startswith("Branch:"):
            current_branch = stripped.replace("Branch:", "").strip().rstrip(",")
            continue

        # Division detection
        if stripped.startswith("Division:"):
            current_division = stripped.replace("Division:", "").strip().rstrip(",")
            continue

        # Group detection
        if stripped.startswith("Group:"):
            current_group = stripped.replace("Group:", "").strip().rstrip(",")
            continue

        # Skip totals and metadata
        if any(kw in stripped for kw in [
            "Total by", "Description,Barcode", "Sales by Items",
            "REP_S_00191", "Copyright", "www.", "30-Jan-26", "Years:"
        ]):
            continue

        parts = [p.strip().strip('"') for p in stripped.split(",")]
        parts = [p for p in parts if p]

        # Item row: Description, (Barcode=empty), Qty, Total Amount
        if len(parts) >= 2 and current_branch:
            description = parts[0]
            qty         = parts[1] if len(parts) > 1 else ""
            total_amt   = parts[2] if len(parts) > 2 else ""

            rows.append({
                "Branch":       current_branch,
                "Division":     current_division,
                "Group":        current_group,
                "Description":  description,
                "Qty":          qty,
                "Total_Amount": total_amt
            })

    df = pd.DataFrame(rows, columns=[
        "Branch", "Division", "Group", "Description", "Qty", "Total_Amount"
    ])
    out = os.path.join(OUTPUT_DIR, "rep_s_00191_SMRY_labelled.csv")
    df.to_csv(out, index=False)
    print(f"[✓] File 3 saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# FILE 4 ─ REP_S_00194_SMRY.csv
# Tax Report
# ══════════════════════════════════════════════
def process_file4():
    filepath = os.path.join(INPUT_DIR, "REP_S_00194_SMRY.csv")
    rows = []
    current_branch = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if "Branch Name:" in stripped:
            current_branch = stripped.replace("Branch Name:", "").strip().rstrip(",")
            continue

        if "Total By Branch" in stripped:
            parts = [p.strip().strip('"') for p in stripped.split(",")]
            parts = [p for p in parts if p]
            vat = parts[1] if len(parts) > 1 else "0"
            total = parts[-1] if parts else "0"
            rows.append({
                "Branch":        current_branch,
                "VAT_11_Pct":    vat,
                "Tax_2":         "0",
                "Tax_3":         "0",
                "Tax_4":         "0",
                "Tax_5":         "0",
                "Service":       "0",
                "Total":         total
            })

    df = pd.DataFrame(rows, columns=[
        "Branch", "VAT_11_Pct", "Tax_2", "Tax_3",
        "Tax_4", "Tax_5", "Service", "Total"
    ])
    out = os.path.join(OUTPUT_DIR, "REP_S_00194_SMRY_labelled.csv")
    df.to_csv(out, index=False)
    print(f"[✓] File 4 saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# FILE 5 ─ rep_s_00334_1_SMRY.csv
# Monthly Sales by Branch
# ══════════════════════════════════════════════
def process_file5():
    filepath = os.path.join(INPUT_DIR, "rep_s_00334_1_SMRY.csv")
    rows = []
    current_branch = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if "Branch Name:" in stripped:
            current_branch = stripped.replace("Branch Name:", "").strip().rstrip(",")
            continue

        parts = [p.strip().strip('"') for p in stripped.split(",")]
        parts = [p for p in parts if p]

        # Skip non-data
        if any(kw in stripped for kw in [
            "Monthly Sales", "Year:", "Page", "Total for",
            "Total by Branch", "Grand Total", "REP_S_00334", "Copyright", "www.", "30-Jan-26"
        ]):
            continue

        # Month data row: Month, Year, Total
        if current_branch and len(parts) >= 3:
            rows.append({
                "Branch": current_branch,
                "Month":  parts[0],
                "Year":   parts[1],
                "Total":  parts[2]
            })

    df = pd.DataFrame(rows, columns=["Branch", "Month", "Year", "Total"])
    out = os.path.join(OUTPUT_DIR, "rep_s_00334_1_SMRY_labelled.csv")
    df.to_csv(out, index=False)
    print(f"[✓] File 5 saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# FILES 6 & 7 ─ rep_s_00435_SMRY (1).csv & rep_s_00435_SMRY.csv
# Average Sales By Menu
# ══════════════════════════════════════════════
def process_file_avg_sales(filename, label):
    filepath = os.path.join(INPUT_DIR, filename)
    rows = []
    current_branch = None
    branch_names = {"Conut - Tyre", "Conut", "Conut Jnah", "Main Street Coffee"}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = [p.strip().strip('"') for p in stripped.split(",")]
        parts = [p for p in parts if p]

        if parts and parts[0] in branch_names:
            current_branch = parts[0]
            continue

        if any(kw in stripped for kw in [
            "Average Sales", "Menu Name", "Total :", "REP_S_00435",
            "Copyright", "www.", "30-Jan-26", "Year:"
        ]):
            continue

        # Data row: Menu_Type, Num_Customers, Sales, Avg_Customer
        if current_branch and len(parts) >= 4:
            rows.append({
                "Branch":       current_branch,
                "Menu_Type":    parts[0],
                "Num_Customers": parts[1],
                "Sales":        parts[2],
                "Avg_Customer": parts[3]
            })
        elif current_branch and len(parts) == 3 and "Total By Branch" in parts[0]:
            rows.append({
                "Branch":       current_branch,
                "Menu_Type":    "TOTAL",
                "Num_Customers": parts[1],
                "Sales":        parts[2],
                "Avg_Customer": parts[3] if len(parts) > 3 else ""
            })

    df = pd.DataFrame(rows, columns=[
        "Branch", "Menu_Type", "Num_Customers", "Sales", "Avg_Customer"
    ])
    out = os.path.join(OUTPUT_DIR, label)
    df.to_csv(out, index=False)
    print(f"[✓] Saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# FILE 8 ─ REP_S_00461  (Time & Attendance)
# ══════════════════════════════════════════════
def process_file8():
    filepath = os.path.join(INPUT_DIR, "REP_S_00461")
    # Also try with .csv extension
    if not os.path.exists(filepath):
        filepath = filepath + ".csv"

    rows = []
    current_emp_id   = None
    current_emp_name = None
    current_branch   = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    branch_names = {"Conut - Tyre", "Conut", "Conut Jnah",
                    "Main Street Coffee", "Conut Jnah", "Conut - Tyre"}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Employee ID / Name line
        if "EMP ID" in stripped and "NAME" in stripped:
            parts = [p.strip() for p in stripped.split(",") if p.strip()]
            for part in parts:
                if "EMP ID" in part:
                    try:
                        current_emp_id = part.split(":")[1].strip().rstrip(".")
                    except IndexError:
                        pass
                if "NAME" in part:
                    try:
                        current_emp_name = part.split(":")[1].strip()
                    except IndexError:
                        pass
            continue

        # Branch line (standalone branch name)
        clean = stripped.rstrip(",")
        if clean in branch_names:
            current_branch = clean
            continue

        # Skip header/footer/metadata
        if any(kw in stripped for kw in [
            "PUNCH IN", "PUNCH OUT", "Work Duration", "Total :",
            "Time & Attendance", "From Date", "REP_S_00461",
            "Copyright", "www.", "30-Jan-26"
        ]):
            continue

        # Attendance row: Date, PunchIn, Date, PunchOut, Duration
        parts = [p.strip() for p in stripped.split(",")]
        parts = [p for p in parts if p]

        if len(parts) >= 5 and re.match(r"\d{2}-\w{3}-\d{2}", parts[0]):
            rows.append({
                "Emp_ID":        current_emp_id,
                "Emp_Name":      current_emp_name,
                "Branch":        current_branch,
                "Punch_In_Date": parts[0],
                "Punch_In_Time": parts[1],
                "Punch_Out_Date":parts[2],
                "Punch_Out_Time":parts[3],
                "Work_Duration": parts[4]
            })

    df = pd.DataFrame(rows, columns=[
        "Emp_ID", "Emp_Name", "Branch",
        "Punch_In_Date", "Punch_In_Time",
        "Punch_Out_Date", "Punch_Out_Time",
        "Work_Duration"
    ])
    out = os.path.join(OUTPUT_DIR, "REP_S_00461_labelled.csv")
    df.to_csv(out, index=False)
    print(f"[✓] File 8 saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# FILE 9 ─ REP_S_00502.csv
# Sales by Customer in Details (Delivery)
# ══════════════════════════════════════════════
def process_file9():
    filepath = os.path.join(INPUT_DIR, "REP_S_00502.csv")
    rows = []
    current_branch   = None
    current_customer = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Branch detection
        if stripped.startswith("Branch :"):
            current_branch = stripped.replace("Branch :", "").strip().rstrip(",")
            continue

        # Skip headers/footers
        if any(kw in stripped for kw in [
            "Full Name,Qty", "From Date", "To Date", "Page",
            "Sales by customer", "REP_S_00502", "Copyright",
            "www.", "30-Jan-26", "Total Branch:"
        ]):
            continue

        parts = [p.strip().strip('"') for p in stripped.split(",")]
        parts = [p for p in parts if p]

        # Customer name row (Person_XXXX with no qty/price on same line)
        if parts and re.match(r"[\d\s]*Person_\d+", parts[0]):
            current_customer = re.sub(r"^\d+\s*", "", parts[0]).strip()
            continue

        # "Total :" row → summary row per customer
        if parts and parts[0] == "Total :":
            qty   = parts[1] if len(parts) > 1 else ""
            total = parts[2] if len(parts) > 2 else ""
            rows.append({
                "Branch":        current_branch,
                "Customer_Name": current_customer,
                "Row_Type":      "TOTAL",
                "Qty":           qty,
                "Description":   "",
                "Price":         total
            })
            continue

        # Item detail row: Qty, Description, Price
        if current_customer and len(parts) >= 2:
            qty   = parts[0]
            desc  = parts[1] if len(parts) > 1 else ""
            price = parts[2] if len(parts) > 2 else "0"
            rows.append({
                "Branch":        current_branch,
                "Customer_Name": current_customer,
                "Row_Type":      "ITEM",
                "Qty":           qty,
                "Description":   desc,
                "Price":         price
            })

    df = pd.DataFrame(rows, columns=[
        "Branch", "Customer_Name", "Row_Type", "Qty", "Description", "Price"
    ])
    out = os.path.join(OUTPUT_DIR, "REP_S_00502_labelled.csv")
    df.to_csv(out, index=False)
    print(f"[✓] File 9 saved → {out}  ({len(df)} rows)")


# ══════════════════════════════════════════════
# MAIN – Run all processors
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Data Labelling Pipeline – Starting")
    print("=" * 55)

    process_file1()   # rep_s_00150.csv          → Customer Delivery Orders
    process_file2()   # REP_S_00136_SMRY.csv     → Summary By Division
    process_file3()   # rep_s_00191_SMRY.csv     → Sales By Items & Group
    process_file4()   # REP_S_00194_SMRY.csv     → Tax Report
    process_file5()   # rep_s_00334_1_SMRY.csv   → Monthly Sales
    process_file_avg_sales(              # Average Sales (copy 1)
        "rep_s_00435_SMRY (1).csv",
        "rep_s_00435_SMRY_1_labelled.csv"
    )
    process_file_avg_sales(              # Average Sales (copy 2)
        "rep_s_00435_SMRY.csv",
        "rep_s_00435_SMRY_labelled.csv"
    )
    process_file8()   # REP_S_00461              → Time & Attendance
    process_file9()   # REP_S_00502.csv          → Sales by Customer (Delivery)

    print("=" * 55)
    print("  All files processed and saved to 'Data Preparation'")
    print("=" * 55)
