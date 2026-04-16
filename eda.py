import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/healthcare_dataset.csv"
OUTPUT_DIR = "outputs"


def clean_text(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value.title()


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]

    text_columns = [
        "Name",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Doctor",
        "Hospital",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results",
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df["length_of_stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    df = df.dropna(subset=["Age", "Date of Admission", "Discharge Date", "length_of_stay"])
    df = df[df["length_of_stay"] >= 0].copy()

    return df


def save_histogram(df):
    plt.figure(figsize=(9, 5))
    plt.hist(df["length_of_stay"], bins=20, edgecolor="black")
    plt.title("Distribution of Length of Stay")
    plt.xlabel("Length of Stay (days)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "los_distribution.png"))
    plt.close()


def save_condition_plot(df):
    grouped = (
        df.groupby("Medical Condition")["length_of_stay"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    grouped.plot(kind="bar")
    plt.title("Average Length of Stay by Medical Condition")
    plt.xlabel("Medical Condition")
    plt.ylabel("Average Length of Stay (days)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "condition_vs_los.png"))
    plt.close()


def save_admission_type_plot(df):
    grouped = (
        df.groupby("Admission Type")["length_of_stay"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")
    plt.title("Average Length of Stay by Admission Type")
    plt.xlabel("Admission Type")
    plt.ylabel("Average Length of Stay (days)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "admission_type_vs_los.png"))
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_prepare_data(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nLength of stay summary:")
    print(df["length_of_stay"].describe())

    save_histogram(df)
    save_condition_plot(df)
    save_admission_type_plot(df)

    print("\nEDA outputs saved in outputs/ folder.")


if __name__ == "__main__":
    main()
