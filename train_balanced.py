
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("synthetic_stress_data_balanced.csv")

# Encode labels
df['stress_level'] = df['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Features and target
X = df.drop('stress_level', axis=1)
y = df['stress_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])
print("Classification Report:\n", report)

# Save model
joblib.dump(clf, "stress_model_balanced.pkl")
print("Model saved as 'stress_model_balanced.pkl'")
