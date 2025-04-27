from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import json
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

# Initialize Flask App
app = Flask(__name__)

# Load the trained stress prediction model
model = joblib.load("stress_model_balanced.pkl")

# Initialize Gemini API
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
        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        return jsonify({'stress_level': label_map[prediction]})

    except Exception as e:
        return jsonify({'error': str(e)})

# --- CHATBOT ENDPOINT ---
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

        # ⚡ Correct way to extract text:
        bot_reply = response.candidates[0].content.parts[0].text

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# --- AI-BASED STRESS DETECTION ENDPOINT (NO TIPS) ---
@app.route('/ai_predict', methods=['POST'])
def ai_predict():
    try:
        data = request.get_json()

        features = {
            'heart_rate': data.get('heart_rate'),
            'steps': data.get('steps'),
            'calories': data.get('calories'),
            'azm': data.get('azm'),
            'resting_hr': data.get('resting_hr'),
            'hrv': data.get('hrv'),
            'sleep_minutes': data.get('sleep_minutes'),
            'sleep_efficiency': data.get('sleep_efficiency')
        }

        prompt = (
            "Analyze the following fitness data and predict the stress level and probability."
            " Respond strictly in JSON format only without extra explanation.\n\n"
            f"Heart Rate: {features['heart_rate']} bpm\n"
            f"Steps: {features['steps']}\n"
            f"Calories Burned: {features['calories']} kcal\n"
            f"Active Zone Minutes: {features['azm']}\n"
            f"Resting Heart Rate: {features['resting_hr']} bpm\n"
            f"Heart Rate Variability (HRV): {features['hrv']} ms\n"
            f"Sleep Duration: {features['sleep_minutes']} minutes\n"
            f"Sleep Efficiency: {features['sleep_efficiency']} %\n\n"
            "⚡ Strict JSON Format:\n"
            "{\n"
            "  \"stress_level\": \"Low / Moderate / High\",\n"
            "  \"stress_probability\": \"xx%\"\n"
            "}"
        )

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)

        structured_data = json.loads(response.text)

        return jsonify(structured_data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# --- AI-BASED RECOMMENDATIONS ENDPOINT ---
@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.get_json()

        features = {
            'heart_rate': data.get('heart_rate'),
            'steps': data.get('steps'),
            'calories': data.get('calories'),
            'azm': data.get('azm'),
            'resting_hr': data.get('resting_hr'),
            'hrv': data.get('hrv'),
            'sleep_minutes': data.get('sleep_minutes'),
            'sleep_efficiency': data.get('sleep_efficiency')
        }

        prompt = (
            "Analyze the following fitness data and explain why the user's stress level might be low, moderate, or high."
            " Also suggest a short advice to improve it. Respond strictly in JSON format.\n\n"
            f"Heart Rate: {features['heart_rate']} bpm\n"
            f"Steps: {features['steps']}\n"
            f"Calories Burned: {features['calories']} kcal\n"
            f"Active Zone Minutes: {features['azm']}\n"
            f"Resting Heart Rate: {features['resting_hr']} bpm\n"
            f"Heart Rate Variability (HRV): {features['hrv']} ms\n"
            f"Sleep Duration: {features['sleep_minutes']} minutes\n"
            f"Sleep Efficiency: {features['sleep_efficiency']} %\n\n"
            "⚡ Strict JSON format only:\n"
            "{\n"
            "  \"reason\": \"Why stress is low/moderate/high based on the data\",\n"
            "  \"advice\": \"Simple advice to improve stress or maintain wellness\"\n"
            "}"
        )

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)

        structured_data = json.loads(response.text)

        return jsonify(structured_data)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
