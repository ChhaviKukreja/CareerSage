import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

def extract_features(df, text_column, numeric_columns):
    """Convert text to TF-IDF features and combine with numeric features."""
    # Text vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df[text_column].astype(str))

    # Numeric scaling
    X_numeric = df[numeric_columns]
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrix compatibility
    X_num_scaled = scaler.fit_transform(X_numeric)

    # Combine features
    X = hstack([X_text, X_num_scaled])
    return X, tfidf, scaler
