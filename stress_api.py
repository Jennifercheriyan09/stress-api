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

        prediction_proba = model.predict_proba([features])[0]
        prediction = model.predict([features])[0]
        label_map = {0: "Low", 1: "Medium", 2: "High"}

        return jsonify({
            "stress_level": label_map[prediction],
            "score": float(max(prediction_proba))  # highest probability
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- RECOMMENDATIONS ENDPOINT ---
@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.get_json()

        # Extract features
        heart_rate = data.get('heart_rate', 70)
        steps = data.get('steps', 3000)
        calories = data.get('calories', 1800)
        azm = data.get('azm', 40)
        resting_hr = data.get('resting_hr', 70)
        hrv = data.get('hrv', 30.0)
        sleep_minutes = data.get('sleep_minutes', 420)
        sleep_efficiency = data.get('sleep_efficiency', 0.85)

        summary = []
        warnings = ""
        recommendations = []

        # Analyze indicators
        if hrv < 40:
            summary.append(f"HRV is low ({hrv} ms) indicating elevated stress.")
            warnings = "Low HRV is the strongest indicator of stress today."
        else:
            summary.append(f"HRV is normal ({hrv} ms), indicating good recovery.")

        if sleep_efficiency < 0.9:
            summary.append(f"Sleep efficiency is {sleep_efficiency*100:.0f}%, suggesting possible sleep quality issues.")
        else:
            summary.append(f"Sleep efficiency is good ({sleep_efficiency*100:.0f}%).")

        if resting_hr > 70:
            summary.append(f"Resting heart rate is slightly above average ({resting_hr} bpm).")
        else:
            summary.append(f"Resting heart rate is within normal range ({resting_hr} bpm).")

        if steps > 3000 and azm >= 30:
            summary.append("Good daily activity achieved (3000+ steps and sufficient active minutes).")
        else:
            summary.append("Physical activity could be slightly improved for stress reduction.")

        # Recommendations
        recommendations.append("Practice deep breathing for 5–10 minutes today.")
        recommendations.append("Do a 10–15 minute mindfulness meditation or progressive muscle relaxation.")
        recommendations.append("Take a short nature walk if possible to refresh your mind.")

        note = "This analysis is for informational purposes only. Please consult a doctor for any serious concerns."

        return jsonify({
            "summary": summary,
            "warnings": warnings,
            "recommendations": recommendations,
            "note": note
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
