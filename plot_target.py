import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
CSV_PATH = "features/all_subjects.csv"
RESULTS_DIR = "results"

# Load data
df = pd.read_csv(CSV_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# Find important columns
def find_column(columns, possible_names):
    """
    Find a column using a list of possible column-name patterns.
    """
    for name in possible_names:
        for column in columns:
            if column.lower() == name.lower():
                return column

    for name in possible_names:
        for column in columns:
            if name.lower() in column.lower():
                return column

    return None
subject_col = find_column(
    df.columns,
    [
        "Subject",
        "Subject_ID",
        "SubjectID",
        "subject_id",
        "subject"
    ]
)
test_col = find_column(
    df.columns,
    [
        "Test",
        "Test_ID",
        "TestID",
        "test_id",
        "test"
    ]
)
target_col = find_column(
    df.columns,
    [
        "MetCost",
        "MetabolicCost",
        "Metabolic_Cost",
        "metabolic_cost",
        "Target",
        "target"
    ]
)

# Check columns
print("\nDetected columns:")
print("Subject column:", subject_col)
print("Test column:", test_col)
print("Target column:", target_col)

if subject_col is None:
    raise ValueError(
        "Could not identify the subject column. "
        "Check the column names printed above."
    )

if test_col is None:
    raise ValueError(
        "Could not identify the test column. "
        "Check the column names printed above."
    )

if target_col is None:
    raise ValueError(
        "Could not identify the metabolic-cost target column. "
        "Check the column names printed above."
    )

# Prepare output directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Convert target to numeric
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

df = df.dropna(
    subset=[subject_col, test_col, target_col]
).copy()

# Plot 1: Target per Subject
plt.figure(figsize=(10, 6))

subjects = sorted(df[subject_col].unique())

data_per_subject = [
    df.loc[df[subject_col] == subject, target_col]
    for subject in subjects
]

plt.boxplot(
    data_per_subject,
    tick_labels=[str(subject) for subject in subjects]
)

plt.xlabel("Subject")
plt.ylabel("Metabolic Cost (W/kg)")
plt.title("Metabolic Cost Distribution per Subject")

plt.tight_layout()

plot1_path = os.path.join(
    RESULTS_DIR,
    "target_per_subject.png"
)

plt.savefig(plot1_path, dpi=300)
plt.show()

print("\nSaved:")
print(plot1_path)

# Convert test names to strings
df[test_col] = df[test_col].astype(str).str.strip()

# Identify T1-T15 and T16-T30
def get_test_number(value):
    """
    Extract the numerical test number from values such as:
    T1, T2, ..., T30
    """
    text = str(value).strip().upper()

    if text.startswith("T"):
        text = text[1:]

    try:
        return int(text)
    except ValueError:
        return None


df["Test_Number"] = df[test_col].apply(get_test_number)

# Remove rows where a test number could not be identified

df = df.dropna(subset=["Test_Number"]).copy()

df["Test_Number"] = df["Test_Number"].astype(int)


# Create two groups
df["Test_Group"] = df["Test_Number"].apply(
    lambda x: "T1-T15" if 1 <= x <= 15
    else ("T16-T30" if 16 <= x <= 30 else None)
)

df = df.dropna(subset=["Test_Group"])

# Plot 2: T1-T15 vs T16-T30
plt.figure(figsize=(10, 6))

groups = ["T1-T15", "T16-T30"]

data_per_group = [
    df.loc[df["Test_Group"] == group, target_col]
    for group in groups
]

plt.boxplot(
    data_per_group,
    tick_labels=groups
)

plt.xlabel("Test Group")
plt.ylabel("Metabolic Cost (W/kg)")
plt.title("Metabolic Cost: T1-T15 vs T16-T30")

plt.tight_layout()

plot2_path = os.path.join(
    RESULTS_DIR,
    "target_T1_T15_vs_T16_T30.png"
)

plt.savefig(plot2_path, dpi=300)
plt.show()

print("\nSaved:")
print(plot2_path)


# Plot 3: T1-T15 vs T16-T30 for each subject
plt.figure(figsize=(12, 7))

plot_data = []
plot_labels = []

for subject in subjects:

    subject_data = df[df[subject_col] == subject]

    group_1 = subject_data.loc[
        subject_data["Test_Group"] == "T1-T15",
        target_col
    ]

    group_2 = subject_data.loc[
        subject_data["Test_Group"] == "T16-T30",
        target_col
    ]

    if len(group_1) > 0:
        plot_data.append(group_1)
        plot_labels.append(f"S{subject}\nT1-T15")

    if len(group_2) > 0:
        plot_data.append(group_2)
        plot_labels.append(f"S{subject}\nT16-T30")


plt.boxplot(
    plot_data,
    tick_labels=plot_labels
)

plt.xlabel("Subject and Test Group")
plt.ylabel("Metabolic Cost (W/kg)")
plt.title("Metabolic Cost by Subject: T1-T15 vs T16-T30")

plt.xticks(rotation=45)

plt.tight_layout()

plot3_path = os.path.join(
    RESULTS_DIR,
    "target_by_subject_T1_T15_vs_T16_T30.png"
)

plt.savefig(plot3_path, dpi=300)
plt.show()

print("\nSaved:")
print(plot3_path)



print("\n-----------------------------------------")
print("Plotting completed successfully.")
print("-----------------------------------------")

print("\nGenerated files:")

print(plot1_path)
print(plot2_path)
print(plot3_path)