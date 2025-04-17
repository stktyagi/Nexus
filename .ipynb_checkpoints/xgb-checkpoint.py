import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Function to extract a wide range of features from sample



def extract_features(entry):
    features = {}

    # Label
    features['label'] = entry.get('label', 0)

    # histogram
    hist = entry.get("histogram", [])
    for i, val in enumerate(hist):
        features[f"histogram_{i}"] = val

    # byteentropy
    entropy = entry.get("byteentropy", [])
    for i, val in enumerate(entropy):
        features[f"byteentropy_{i}"] = val

    # strings
    strings = entry.get("strings", {})
    features["numstrings"] = strings.get("numstrings", 0)
    features["avlength"] = strings.get("avlength", 0)
    features["entropy_strings"] = strings.get("entropy", 0)
    features["printables"] = strings.get("printables", 0)

    # general
    general = entry.get("general", {})
    features["size"] = general.get("size", 0)
    features["vsize"] = general.get("vsize", 0)
    features["exports"] = general.get("exports", 0)
    features["imports"] = general.get("imports", 0)
    features["has_debug"] = int(general.get("has_debug", 0))
    features["has_relocations"] = int(general.get("has_relocations", 0))
    features["has_resources"] = int(general.get("has_resources", 0))
    features["has_signature"] = int(general.get("has_signature", 0))
    features["has_tls"] = int(general.get("has_tls", 0))
    features["symbols"] = general.get("symbols", 0)

    # header
    header = entry.get("header", {})
    coff = header.get("coff", {})
    optional = header.get("optional", {})

    features["coff_timestamp"] = coff.get("timestamp", 0)
    features["sizeof_code"] = optional.get("sizeof_code", 0)
    features["sizeof_headers"] = optional.get("sizeof_headers", 0)
    features["sizeof_heap_commit"] = optional.get("sizeof_heap_commit", 0)

    # imports (count total functions across all DLLs)
    imports = entry.get("imports", {})
    features["total_import_functions"] = sum(len(funcs) for funcs in imports.values())

    return features






# Load .jsonl data from files
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
y_train = y_train.replace(-1, 2)
y_test = y_test.replace(-1, 2)
# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost classifier
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test_scaled)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

