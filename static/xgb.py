import pandas as pd
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Feature extraction from each sample
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

# Load data from multiple JSONL files
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
# Fix invalid class labels
y_train = y_train.replace(-1, 2)
y_test = y_test.replace(-1, 2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Classifier
clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=1,
    use_label_encoder=False
)
#clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_scaled)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

