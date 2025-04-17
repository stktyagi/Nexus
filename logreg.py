import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Function to extract features from a single JSON line
def extract_features(sample):
    features = {}
    # Basic features
    features['size'] = sample.get('general', {}).get('size', 0)
    features['vsize'] = sample.get('general', {}).get('vsize', 0)
    features['has_debug'] = sample.get('general', {}).get('has_debug', 0)
    features['exports'] = sample.get('general', {}).get('exports', 0)
    features['imports'] = sample.get('general', {}).get('imports', 0)
    features['has_relocations'] = sample.get('general', {}).get('has_relocations', 0)
    features['has_resources'] = sample.get('general', {}).get('has_resources', 0)
    features['has_signature'] = sample.get('general', {}).get('has_signature', 0)
    features['has_tls'] = sample.get('general', {}).get('has_tls', 0)
    features['symbols'] = sample.get('general', {}).get('symbols', 0)
    features['numstrings'] = sample.get('strings', {}).get('numstrings', 0)
    features['avlength'] = sample.get('strings', {}).get('avlength', 0.0)
    features['entropy'] = sample.get('strings', {}).get('entropy', 0.0)
    features['label'] = sample.get('label', 0)
    return features

# Function to load data from JSONL files
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f"Processing {file_path}"):
                sample = json.loads(line)
                features = extract_features(sample)
                data.append(features)
    return pd.DataFrame(data)

# Paths to training and testing files
train_files = [f"ember2018/train_features_{i}.jsonl" for i in range(6)]
test_file = "ember2018/test_features.jsonl"

# Load training and testing data
train_df = load_data(train_files)
test_df = load_data([test_file])

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize and train model
model = LogisticRegression(max_iter=1000)
print("\nTraining Logistic Regression...")
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
