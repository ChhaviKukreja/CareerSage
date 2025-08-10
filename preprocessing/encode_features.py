# preprocessing/encode_features.py
import re
import pandas as pd

def extract_keywords_from_training(df, text_columns):
    """Extract all unique keywords from specified columns in training dataset."""
    keywords = set()
    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)
        for text in df[col]:
            words = re.findall(r'\b\w[\w#+.-]*\b', text)  # includes C++, C#, HTML5
            keywords.update(words)
    return sorted(keywords)

def encode_features_dynamic(df, keywords=None, text_columns=None):
    """Encodes dataset features dynamically based on given or extracted keywords."""
    if text_columns is None:
        text_columns = df.select_dtypes(include=['object']).columns.tolist()

    # If no keywords provided, extract from dataset (training mode)
    if keywords is None:
        keywords = extract_keywords_from_training(df, text_columns)

    for col in text_columns:
        for kw in keywords:
            clean_kw = kw.strip().replace(" ", "_")
            df[f"{col}_{clean_kw}"] = df[col].str.contains(
                rf"\b{re.escape(kw)}\b", case=False, na=False
            ).astype(int)

    return df, keywords
