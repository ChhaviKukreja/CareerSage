import pandas as pd
import csv

def load_data(filepath):
    fixed_rows = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = len(header)
        fixed_rows.append(header)

        for i, row in enumerate(reader, start=2):
            if len(row) > expected_cols:
                merge_index_start = 3
                merge_index_end = len(row) - (expected_cols - merge_index_start - 1)
                merged_skills = ", ".join(row[merge_index_start:merge_index_end]).strip()
                row = row[:merge_index_start] + [merged_skills] + row[merge_index_end:]

            elif len(row) < expected_cols:
                row += ["Unknown"] * (expected_cols - len(row))

            fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows[1:], columns=fixed_rows[0])
    return df

def save_data(df, filepath):
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)
