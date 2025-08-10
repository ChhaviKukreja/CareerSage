import pandas as pd

def preprocess_for_prediction(input_csv, output_csv):
    """
    This function preprocesses NEW data to match the format of career_dataset_processed.csv
    """
    df = pd.read_csv(input_csv)

    # Make sure columns match the training dataset features
    expected_columns = [
        "Education Level", "GPA",
        # Add all your one-hot encoded Subjects_, Skills_, Interests_ columns here
        "Subjects_Math", "Subjects_Science", "Subjects_English",
        "Skills_Coding", "Skills_Communication",
        "Interests_Research", "Interests_Design"
    ]

    # If a column is missing in new data, add it filled with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only the expected columns in the same order
    df = df[expected_columns]

    df.to_csv(output_csv, index=False)
    print(f"✅ Preprocessed new data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_for_prediction("data/test_data.csv", "data/preprocessed_test.csv")
