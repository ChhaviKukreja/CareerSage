import pandas as pd

def clean_data(df, target_column):
    # Example: convert GPA if present
    if "GPA" in df.columns:
        df["GPA"] = pd.to_numeric(df["GPA"], errors="coerce")
        df["GPA"] = df["GPA"].fillna(df["GPA"].mean())

    # Merge rare categories into 'Other'
    min_count = 2
    class_counts = df[target_column].value_counts()
    rare_classes = class_counts[class_counts < min_count].index
    df[target_column] = df[target_column].replace(rare_classes, "Other")

    return df
