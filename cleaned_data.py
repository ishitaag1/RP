
import pandas as pd
import glob
import os

input_folder = r"D:\RP\data\Brajrajnagar"
output_file = r"D:\RP\cleaned_data.xlsx"

files = (
    glob.glob(os.path.join(input_folder, "*.csv"))
    + glob.glob(os.path.join(input_folder, "*.xlsx"))
)

if not files:
    raise FileNotFoundError("No CSV or Excel files found!")

dataframes = []

for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df["source_file"] = os.path.basename(file)
    dataframes.append(df)

    print(f"Loaded: {file}, Shape: {df.shape}")

# ---------------------------------
# 3. CLEAN EACH FILE
# ---------------------------------

def clean_data(df):

    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove extra spaces from text
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Convert timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

    # Convert measurement columns to numbers
    for col in df.columns:
        if col not in ["timestamp", "source_file"]:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    return df


cleaned_dfs = [clean_data(df) for df in dataframes]

# ---------------------------------
# 4. MERGE ALL FILES
# ---------------------------------

final_df = pd.concat(
    cleaned_dfs,
    ignore_index=True,
    sort=False
)

# ---------------------------------
# 5. REMOVE DUPLICATES
# ---------------------------------

# Remove identical rows
before = len(final_df)
final_df = final_df.drop_duplicates()
duplicate_rows_removed = before - len(final_df)

# Sort by timestamp
if "timestamp" in final_df.columns:
    final_df = final_df.sort_values(
        "timestamp", na_position="last"
    )

# ---------------------------------
# 6. MISSING VALUE REPORT
# ---------------------------------

missing_report = pd.DataFrame({
    "Column": final_df.columns,
    "Missing_Count": final_df.isna().sum().values,
    "Missing_Percentage": (
        final_df.isna().mean().values * 100
    ).round(2)
})

# ---------------------------------
# 7. DATA QUALITY SUMMARY
# ---------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Number of files",
        "Total rows after merging",
        "Total columns",
        "Duplicate rows removed",
        "Missing cells",
        "Rows with missing timestamps"
    ],
    "Value": [
        len(files),
        len(final_df),
        len(final_df.columns),
        duplicate_rows_removed,
        int(final_df.isna().sum().sum()),
        int(final_df["timestamp"].isna().sum())
        if "timestamp" in final_df.columns else 0
    ]
})

# ---------------------------------
# 8. EXPORT TO EXCEL
# ---------------------------------

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    final_df.to_excel(
        writer, sheet_name="Cleaned_Data", index=False
    )

    missing_report.to_excel(
        writer, sheet_name="Missing_Values", index=False
    )

    summary.to_excel(
        writer, sheet_name="Summary", index=False
    )

print("Cleaning and merging completed!")
print("Final shape:", final_df.shape)
print("Output file:", output_file)