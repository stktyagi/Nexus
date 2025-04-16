import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# ------------------------ Data Loading ------------------------

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

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f"Loading {file_path}"):
                sample = json.loads(line)
                features = extract_features(sample)
                data.append(features)
    return pd.DataFrame(data)

# ------------------------ Load and Prepare ------------------------

train_files = [f"ember2018/train_features_{i}.jsonl" for i in range(6)]
test_file = "ember2018/test_features.jsonl"

train_df = load_data(train_files)
test_df = load_data([test_file])

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------ Models ------------------------

xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
#cat = CatBoostClassifier(verbose=0, iterations=100, random_state=42)
lgb = LGBMClassifier(n_estimators=100, random_state=42)

# Ensemble with VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('xgb', xgb),
#    ('cat', cat),
    ('lgb', lgb)
], voting='soft')

# ------------------------ Train and Evaluate ------------------------

voting_clf.fit(X_train_scaled, y_train)
y_pred = voting_clf.predict(X_test_scaled)

print("Ensemble Accuracy (XGBoost + CatBoost + LightGBM):", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

