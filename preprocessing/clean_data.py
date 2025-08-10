import pandas as pd

def clean_degrees(df):
    if 'Education Level' in df.columns:
        df['Education Level'] = df['Education Level'].astype(str).str.replace(r'\.', '', regex=True)
        df['Education Level'] = df['Education Level'].str.strip().str.lower()
    return df

def clean_list_columns(df, list_columns):
    """
    Ensures list-based columns have trimmed, title-cased items.
    If they're stored as strings, converts them to lists first.
    """
    for col in list_columns:
        if col in df.columns:
            # Convert string representation of lists to actual lists
            df[col] = df[col].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )

            # Ensure we have lists
            df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [v])

            # Clean each item in the list
            df[col] = df[col].apply(lambda lst: [str(item).strip().title() for item in lst if pd.notna(item)])
    return df

def clean_column_text(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

def handle_missing_values(df):
    # Temporarily convert lists to tuples
    df_temp = df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    
    # Drop duplicates
    df = df.loc[~df_temp.duplicated()]
    
    df['Name'] = df['Name'].fillna('Unknown')
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
    df['GPA'] = df['GPA'].fillna(df['GPA'].mean())
    df = df.fillna('Unknown')
    return df


def clean_data(df):
    df = clean_degrees(df)

    # Handle list-based columns separately
    df = clean_list_columns(df, ['Skills', 'Interests', 'Subjects'])

    # Handle normal text columns
    df = clean_column_text(df, ['Name', 'Career Path'])

    df = handle_missing_values(df)
    return df
