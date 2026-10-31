
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)

# =============================================
# 1. Load and prepare the dataset
# =============================================

df = pd.read_csv("data/flight_data.csv")
df.columns = df.columns.str.strip()
df = df.dropna().copy()

df["Delay"] = (df["Arrival Delay"] > 15).astype(int)

def extract_hour(value):
    try:
        return int(str(value).split(":")[0])
    except (ValueError, TypeError):
        return 0

df["Dep_Hour"] = df["Scheduled Departure"].apply(extract_hour)
df["Arr_Hour"] = df["Scheduled Arrival"].apply(extract_hour)

df = pd.get_dummies(
    df,
    columns=["Airline", "From", "To"],
    drop_first=True
)

features = [
    "Dep_Hour",
    "Arr_Hour",
    "weather__hourly__visibility",
    "weather__hourly__humidity",
    "weather__hourly__cloudcover"
] + [
    col for col in df.columns
    if col.startswith(("Airline_", "From_", "To_"))
]

X = df[features]
y = df["Delay"]

print("Records after cleaning:", len(df))
print("Delayed flights:", int(y.sum()))

# Keep the original 80/20 split for comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =============================================
# 2. Define the Random Forest model
# =============================================

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# =============================================
# 3. Training-only cross-validation
# =============================================

print("\nFinding a decision threshold...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Out-of-fold predictions on training data only
cv_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

# Minimum acceptable precision, adjustable
MIN_PRECISION = 0.55

results = []

for threshold in np.arange(0.20, 0.801, 0.025):

    predictions = (
        cv_probabilities >= threshold
    ).astype(int)

    results.append({
        "threshold": round(float(threshold), 3),
        "precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_train, predictions, zero_division=0
        ),
        "f2": fbeta_score(
            y_train, predictions,
            beta=2, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

# Select by F2 score, subject to precision floor
eligible = results_df[
    results_df["precision"] >= MIN_PRECISION
]

default_recall = recall_score(
    y_train,
    (cv_probabilities >= 0.5).astype(int),
    zero_division=0
)

if eligible.empty:
    best_threshold = 0.5
    print("No threshold met minimum precision.")
else:
    best = eligible.sort_values(
        ["f2", "recall"],
        ascending=[False, False]
    ).iloc[0]

    best_threshold = float(best["threshold"])

    if best["recall"] <= default_recall:
        best_threshold = 0.5
        print("No recall improvement; using 0.50.")

print(f"Selected threshold: {best_threshold:.3f}")

print("\nTop validation candidates:")
print(
    results_df.sort_values(
        "f2", ascending=False
    ).head(10).round(3).to_string(index=False)
)

# =============================================
# 4. Train final model
# =============================================

model.fit(X_train, y_train)

test_probabilities = model.predict_proba(X_test)[:, 1]

def evaluate(threshold, label):

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    print(f"\n===== {label} =====")
    print(f"Threshold: {threshold:.3f}")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )
    print(
        "Balanced accuracy:",
        round(
            balanced_accuracy_score(y_test, predictions), 4
        )
    )
    print(
        "Delayed precision:",
        round(
            precision_score(
                y_test, predictions, zero_division=0
            ), 4
        )
    )
    print(
        "Delayed recall:",
        round(
            recall_score(
                y_test, predictions, zero_division=0
            ), 4
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ))

    return predictions

# Majority class determined from training data
majority_class = y_train.mode()[0]

baseline_predictions = np.full(
    len(y_test), majority_class
)

print(
    "\nMajority-class baseline:",
    round(
        accuracy_score(y_test, baseline_predictions), 4
    )
)

# Compare both thresholds on untouched test set
default_predictions = evaluate(
    0.5, "DEFAULT MODEL"
)

tuned_predictions = evaluate(
    best_threshold, "TUNED MODEL"
)

print("\nFinal classification report:")
print(classification_report(
    y_test,
    tuned_predictions,
    labels=[0, 1],
    target_names=["Not Delayed", "Delayed"],
    zero_division=0
))

# =============================================
# 5. Save model and selected threshold
# =============================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/flight_delay_model.pkl",
    compress=3
)

with open(
    "model/decision_threshold.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump({
        "delay_threshold": best_threshold,
        "minimum_validation_precision": MIN_PRECISION,
        "selection_method": "5-fold CV with F2 score"
    }, f, indent=2)

print("\nModel and threshold saved successfully.")
print("model/flight_delay_model.pkl")
print("model/decision_threshold.json")
