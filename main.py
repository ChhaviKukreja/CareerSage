from preprocessing import load_data, save_data, clean_data, normalize_gpa, engineer_features

def main():
    df = load_data('data/improved_train.csv')
    df = clean_data(df)
    df = normalize_gpa(df)

    X, y = engineer_features(df, training=True)

    processed = X.copy()
    processed['Career Path'] = y
    save_data(processed, 'data/improved_train_processed.csv')

    print("✅ Training preprocessing complete. Feature space frozen.")

if __name__ == "__main__":
    main()
