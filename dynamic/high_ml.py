import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Load data
df = pd.read_csv("detonation_overviews.csv")

# Drop rows with missing 'verdict'
df = df.dropna(subset=['verdict'])

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical with 'Unknown'
df['vx_family'] = df['vx_family'].fillna("Unknown")
df['type'] = df['type'].fillna("Unknown")

# Encode categorical columns
le_verdict = LabelEncoder()
le_family = LabelEncoder()
le_type = LabelEncoder()

df['verdict'] = le_verdict.fit_transform(df['verdict'])
df['vx_family'] = le_family.fit_transform(df['vx_family'])
df['type'] = le_type.fit_transform(df['type'])

# Drop unneeded columns
df = df.drop(columns=['sha256', 'multiscan_result'])

# Feature-target split
X = df.drop(columns=['verdict'])
y = df['verdict']

# Normalize numeric features only for logistic regression
scaler = StandardScaler()
X_scaled = X.copy()  # Create a copy
X_scaled = pd.DataFrame(scaler.fit_transform(X_scaled), columns=X.columns)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=14,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        use_label_encoder=False,
        eval_metric="logloss")
}

# Perform Cross-Validation for each model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

for name, model in models.items():
    print(f"\nTraining {name} with Cross-Validation...")

    # Only scale for Logistic Regression
    if name == "Logistic Regression":
        X_scaled_cv = scaler.fit_transform(X)
    else:
        X_scaled_cv = X  # No scaling for Random Forest or XGBoost

    # Perform cross-validation
    scores = cross_val_score(model, X_scaled_cv, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-Validation Scores for {name}: {scores}")
    print(f"Mean Cross-Validation Accuracy for {name}: {scores.mean():.4f}")
    print(f"Standard Deviation for {name}: {scores.std():.4f}")

    # Fit the model to the entire training data for final evaluation
    model.fit(X_scaled_cv, y)
    y_pred = model.predict(X_scaled_cv)

    print(f"\n{name} Classification Report:")
    print(classification_report(y, y_pred, target_names=le_verdict.classes_))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

