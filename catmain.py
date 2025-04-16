import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

# -----------------------
# Feature Extraction
# -----------------------
def extract_features(sample):
    features = {
        'size': sample.get('general', {}).get('size', 0),
        'vsize': sample.get('general', {}).get('vsize', 0),
        'has_debug': sample.get('general', {}).get('has_debug', 0),
        'exports': sample.get('general', {}).get('exports', 0),
        'imports': sample.get('general', {}).get('imports', 0),
        'has_relocations': sample.get('general', {}).get('has_relocations', 0),
        'has_resources': sample.get('general', {}).get('has_resources', 0),
        'has_signature': sample.get('general', {}).get('has_signature', 0),
        'has_tls': sample.get('general', {}).get('has_tls', 0),
        'symbols': sample.get('general', {}).get('symbols', 0),
        'numstrings': sample.get('strings', {}).get('numstrings', 0),
        'avlength': sample.get('strings', {}).get('avlength', 0.0),
        'entropy': sample.get('strings', {}).get('entropy', 0.0),
        'label': sample.get('label', 0)
    }
    return features

# -----------------------
# Load Data
# -----------------------
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f"Loading {file_path}"):
                sample = json.loads(line)
                features = extract_features(sample)
                data.append(features)
    return pd.DataFrame(data)

# -----------------------
# Paths
# -----------------------
train_files = [f"ember2018/train_features_{i}.jsonl" for i in range(6)]
test_file = "ember2018/test_features.jsonl"

# -----------------------
# Load and Split
# -----------------------
train_df = load_data(train_files)
test_df = load_data([test_file])

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# -----------------------
# Train CatBoost
# -----------------------
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------
# Predict and Evaluate
# -----------------------
y_pred = model.predict(X_test)

print("\nCatBoost Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

