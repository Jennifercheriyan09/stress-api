
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("stress_model_balanced.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            data['heart_rate'],
            data['steps'],
            data['calories'],
            data['azm'],
            data['resting_hr'],
            data['hrv'],
            data['br'],
            data['sleep_minutes'],
            data['sleep_efficiency']
        ]

        prediction = model.predict([features])[0]
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        return jsonify({'stress_level': label_map[prediction]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)

