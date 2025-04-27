from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env


# NEW: Import for Gemini
import google.generativeai as genai

# Initialize Flask App
app = Flask(__name__)

# Load the trained stress prediction model
model = joblib.load("stress_model_balanced.pkl")

# Initialize Gemini API (⚡ insert your API key here)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# --- STRESS PREDICTION ENDPOINT ---
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
            data['sleep_minutes'],
            data['sleep_efficiency']
        ]

        prediction = model.predict([features])[0]
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        return jsonify({'stress_level': label_map[prediction]})

    except Exception as e:
        return jsonify({'error': str(e)})

# --- CHATBOT ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided."}), 400

        template_message = (
            "You are an intelligent chatbot integrated into a stress-level detection and "
            "recommendation app. Your role is to help users understand their stress levels, "
            "provide tips to manage stress, and answer questions about stress-related topics. "
            "Please ensure your responses are concise, empathetic, and actionable. "
            "Keep in mind that the app's purpose is to assist users in managing their stress."
        )

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        response = model.generate_content(f"{template_message}\n\nUser: {user_message}")

        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
