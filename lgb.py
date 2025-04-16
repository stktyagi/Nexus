import pandas as pd
import json
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Function to extract features from a JSON sample
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

# Load data from .jsonl files
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f"Loading {file_path}"):
                sample = json.loads(line)
                features = extract_features(sample)
                data.append(features)
    return pd.DataFrame(data)

# File paths
train_files = [f"ember2018/train_features_{i}.jsonl" for i in range(6)]
test_file = "ember2018/test_features.jsonl"

# Load datasets
train_df = load_data(train_files)
test_df = load_data([test_file])

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create LightGBM datasets
train_dataset = lgb.Dataset(X_train_scaled, label=y_train)
test_dataset = lgb.Dataset(X_test_scaled, label=y_test, reference=train_dataset)

# Set parameters
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42
}

# Train LightGBM model
model = lgb.train(params, train_dataset, num_boost_round=100)

# Predict and evaluate
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

print("LightGBM Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

