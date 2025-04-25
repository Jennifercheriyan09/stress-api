import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the balanced dataset
df = pd.read_csv("synthetic_stress_data_balanced.csv")

# Drop the 'br' (breathing rate) column
df = df.drop(columns=["br"])

# Encode target labels
df['stress_level'] = df['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Split features and labels
X = df.drop("stress_level", axis=1)
y = df["stress_level"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

# Save the model
joblib.dump(model, "stress_model_balanced.pkl")
print("\n✅ Model saved to stress_model_balanced.pkl")
