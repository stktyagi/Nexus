import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import json

# Load all training JSONL files
data_folder = "ember2018"
files = sorted([f for f in os.listdir(data_folder) if f.startswith("train_features_") and f.endswith(".jsonl")],
               key=lambda x: int(x.split("_")[-1].split(".")[0]))

train_data = []
for file in files:
    with open(os.path.join(data_folder, file), 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

# Load test file
test_data = []
with open(os.path.join(data_folder, "test_features.jsonl"), 'r') as f:
    for line in f:
        test_data.append(json.loads(line))

# Flatten nested features
def flatten_features(row):
    flat_row = {}
    for key, val in row.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat_row[f"{key}_{subkey}"] = subval
        else:
            flat_row[key] = val
    return flat_row

train_df = pd.DataFrame([flatten_features(row) for row in train_data])
test_df = pd.DataFrame([flatten_features(row) for row in test_data])

# Drop missing values and SHA
train_df.drop(columns=["sha256"], errors="ignore", inplace=True)
test_df.drop(columns=["sha256"], errors="ignore", inplace=True)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Split features and labels
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=14,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

