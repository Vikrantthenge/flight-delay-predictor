import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Load Dataset ---
df = pd.read_csv("data/flight_data.csv")  # Make sure the file is in the correct folder
df.columns = df.columns.str.strip()       # Clean column names

# --- Drop rows with missing values ---
df = df.dropna()

# --- Create Binary Target from Arrival Delay ---
df["Delay"] = df["Arrival Delay"].apply(lambda x: 1 if x > 15 else 0)

# --- Convert Time Strings to Hour Integers ---
def extract_hour(time_str):
    try:
        return int(str(time_str).split(":")[0])
    except:
        return 0

df["Dep_Hour"] = df["Scheduled Departure"].apply(extract_hour)
df["Arr_Hour"] = df["Scheduled Arrival"].apply(extract_hour)

# --- Encode Categorical Columns ---
df = pd.get_dummies(df, columns=["Airline", "From", "To"], drop_first=True)

# --- Select Features ---
features = [
    "Dep_Hour", "Arr_Hour",
    "weather__hourly__visibility",
    "weather__hourly__humidity",
    "weather__hourly__cloudcover"
] + [col for col in df.columns if col.startswith("Airline_") or col.startswith("From_") or col.startswith("To_")]

X = df[features]
y = df["Delay"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# --- Evaluate ---
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score
)

y_pred = model.predict(X_test)

# Dataset size after dropping missing rows
print("Records after cleaning:", len(df))
print("Training records:", len(X_train))
print("Test records:", len(X_test))

# Target distribution
print("\nTarget distribution (0 = not delayed, 1 = delayed):")
print(df["Delay"].value_counts())
print("\nTarget percentages:")
print((df["Delay"].value_counts(normalize=True) * 100).round(2))

# Model test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

# Baseline: always predict the majority class in the training set
majority_class = y_train.mode()[0]
baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Training majority class: {majority_class}")
print(f"Majority-class baseline accuracy: {baseline_accuracy * 100:.2f}%")

# Delayed-flight recall
delay_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
print(f"Delayed-flight recall: {delay_recall * 100:.2f}%")

# Full evaluation
print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Not delayed (15 min or less)", "Delayed (over 15 min)"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

import os
os.makedirs("model", exist_ok=True)

# --- Save Model ---
joblib.dump(model, "model/flight_delay_model.pkl")
print("✅ Model saved to model/flight_delay_model.pkl")
