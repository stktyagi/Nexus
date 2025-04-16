import pandas as pd
import json

# Read and parse each line as JSON
records = []
with open('ember2018/train_features_0.jsonl', 'r') as file:
    for line in file:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping malformed line: {e}")

# Convert to DataFrame
df = pd.DataFrame(records)

# Normalize nested fields
def safe_normalize(df, column_name):
    if column_name in df.columns:
        return pd.json_normalize(df[column_name]).add_prefix(f'{column_name}.')
    return pd.DataFrame()

# Extract and normalize known nested fields
nested_fields = ['strings', 'general', 'header', 'section']
normalized_dfs = []

for field in nested_fields:
    normalized_df = safe_normalize(df, field)
    if not normalized_df.empty:
        normalized_dfs.append(normalized_df)
        df = df.drop(columns=[field])

# Combine all into final dataframe
final_df = pd.concat([df] + normalized_dfs, axis=1)

# Display the top rows
print(final_df.head())

# Optional: Save to CSV
# final_df.to_csv('flattened_samples.csv', index=False)

