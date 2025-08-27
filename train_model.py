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
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import os
os.makedirs("model", exist_ok=True)

# --- Save Model ---
joblib.dump(model, "model/flight_delay_model.pkl")
print("✅ Model saved to model/flight_delay_model.pkl")