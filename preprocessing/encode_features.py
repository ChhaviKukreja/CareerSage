# preprocessing/encode_features.py
import pandas as pd
from collections import Counter

def build_vocab_from_lists(df, list_columns, top_n=50):
    """
    Builds a vocabulary of the top N most common items across specified list columns.
    """
    counter = Counter()
    for col in list_columns:
        if col in df.columns:
            for items in df[col]:
                if isinstance(items, list):
                    counter.update(items)
                elif pd.notna(items):
                    counter.update([str(items)])
    vocab = [item for item, _ in counter.most_common(top_n)]
    return vocab

def encode_with_vocab(df, vocab, list_columns):
    """
    Encodes list-based columns using a fixed vocabulary.
    """
    for term in vocab:
        df[f"Feat_{term}"] = df[list_columns].apply(
            lambda row: int(any(term == item for col in list_columns for item in (row[col] if isinstance(row[col], list) else []))),
            axis=1
        )
    return df

def encode_features(df, list_columns, vocab=None, top_n=50, training=True):
    """
    Encodes features for ML model. 
    In training mode, builds vocab; in prediction mode, uses provided vocab.
    """
    if training:
        vocab = build_vocab_from_lists(df, list_columns, top_n=top_n)
    
    df = encode_with_vocab(df, vocab, list_columns)
    
    return df, vocab
