from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from preprocessing.clean_data import clean_data
from model.train_model import train_and_evaluate

def main():
    # Load dataset
    df = pd.read_csv("data/noisy_student_dataset.csv")  # change path if needed

    target_column = "Career Path"  # adjust if different

    # Clean data
    df = clean_data(df,target_column)

    # Merge rare categories into 'Other' before splitting
    min_count = 2
    class_counts = df[target_column].value_counts()
    rare_classes = class_counts[class_counts < min_count].index
    df[target_column] = df[target_column].replace(rare_classes, "Other")

    # Drop rows with NaN in target or text columns
    df.dropna(subset=[target_column, "Skills", "Subjects", "Interests"], inplace=True)

    # Combine multiple text fields into one
    df["combined_text"] = (
        df["Skills"].astype(str) + " " +
        df["Subjects"].astype(str) + " " +
        df["Interests"].astype(str)
    )

    # Encode target labels
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])

    print("Class distribution before splitting:\n", df[target_column].value_counts())

    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[target_column], random_state=42
    )

    # Train & evaluate model
    train_and_evaluate(
        train_df["combined_text"],
        train_df[target_column],
        test_df["combined_text"],
        test_df[target_column],
        label_encoder
    )

if __name__ == "__main__":
    main()
