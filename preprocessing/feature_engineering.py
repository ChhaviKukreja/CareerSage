import numpy as np
import pandas as pd
import ast
import os
import joblib
from collections import Counter
from sklearn.preprocessing import LabelEncoder

ENCODERS_DIR = "models/encoders"
os.makedirs(ENCODERS_DIR, exist_ok=True)

TOP_N = 50  # limit per category

def parse_list_column(df, column):
    """Safely parse list-like strings into Python lists."""
    def parse(x):
        if isinstance(x, list):
            return [str(i).strip() for i in x]
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    return [str(i).strip() for i in val]
            except:
                return [i.strip() for i in x.split(';')]
        return []
    df[column] = df[column].apply(parse)
    return df

def build_vocab(df, column, top_n):
    """Build top-N most frequent items from a list column."""
    counter = Counter()
    for row in df[column]:
        counter.update(row)
    return [item for item, _ in counter.most_common(top_n)]

def encode_with_vocab(df, column, vocab):
    """Create binary columns for each vocab item."""
    for term in vocab:
        df[f"{column}_{term}"] = df[column].apply(lambda items: int(term in items))
    return df

def engineer_features(df, training=True):
    df = df[['Education Level', 'GPA', 'Subjects', 'Skills', 'Interests', 'Career Path']].copy()

    df = parse_list_column(df, 'Subjects')
    df = parse_list_column(df, 'Skills')
    df = parse_list_column(df, 'Interests')

    # -----------------------------
    # Education Level Encoding
    # -----------------------------
    le_path = os.path.join(ENCODERS_DIR, "label_encoder_edu.pkl")
    le = LabelEncoder()

    if training:
        df['Education Level'] = le.fit_transform(df['Education Level'].astype(str))
        joblib.dump(le, le_path)
    else:
        le = joblib.load(le_path)
        df['Education Level'] = df['Education Level'].apply(lambda x: x if x in le.classes_ else "Other")
        if "Other" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Other")
        df['Education Level'] = le.transform(df['Education Level'])

    # -----------------------------
    # Top-N Encoding for list columns
    # -----------------------------
    vocabs_path = os.path.join(ENCODERS_DIR, "topn_vocabs.pkl")
    vocabs = {}

    if training:
        for col in ['Subjects', 'Skills', 'Interests']:
            vocabs[col] = build_vocab(df, col, TOP_N)
        joblib.dump(vocabs, vocabs_path)
    else:
        vocabs = joblib.load(vocabs_path)
        # Replace unseen values with nothing (ignore)
        for col in ['Subjects', 'Skills', 'Interests']:
            df[col] = [[item for item in row if item in vocabs[col]] for row in df[col]]

    for col in ['Subjects', 'Skills', 'Interests']:
        df = encode_with_vocab(df, col, vocabs[col])

    X = df.drop(columns=['Career Path', 'Subjects', 'Skills', 'Interests'])
    y = df['Career Path']

    return X, y
