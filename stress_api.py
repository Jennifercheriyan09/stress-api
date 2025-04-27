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

        # Get ML model prediction
        prediction = model.predict([features])[0]
        label_map = {0: "Low", 1: "Moderate", 2: "High"}
        stress_level = label_map.get(prediction, "Unknown")

        # Ask Gemini to create a detailed analysis
        prompt = f"""
        Analyze the following health parameters:
        Heart Rate: {data['heart_rate']} bpm,
        Steps: {data['steps']},
        Calories: {data['calories']} kcal,
        Active Zone Minutes (AZM): {data['azm']},
        Resting HR: {data['resting_hr']} bpm,
        HRV: {data['hrv']} ms,
        Sleep Minutes: {data['sleep_minutes']} min,
        Sleep Efficiency: {data['sleep_efficiency']}

        Based on this, give a probability or confidence score for the stress level ({stress_level})
        and explain why you rated it so. Also, suggest a small action the user can take.
        """

        model_ai = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model_ai.generate_content(prompt)

        return jsonify({
            "stress_level": stress_level,
            "detailed_insight": response.text  # ✅ Now safely extract text
        })

    except Exception as e:
        print(f"Error in /ai_predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- AI-BASED RECOMMENDATIONS ENDPOINT ---
@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.json

        # Create an intelligent recommendation prompt
        prompt = f"""
        Given this data:
        Heart Rate: {data['heart_rate']} bpm,
        Steps: {data['steps']},
        Calories: {data['calories']} kcal,
        Active Zone Minutes: {data['azm']},
        Resting Heart Rate: {data['resting_hr']} bpm,
        HRV: {data['hrv']} ms,
        Sleep Minutes: {data['sleep_minutes']},
        Sleep Efficiency: {data['sleep_efficiency']},

        Analyze if the person shows signs of stress. 
        Identify one good thing in the data and one warning sign.
        Then recommend 2-3 specific actions they can do today to reduce stress.
        """

        model_ai = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model_ai.generate_content(prompt)

        return jsonify({
            "recommendations": response.text  # ✅ Properly extract text from Gemini
        })

    except Exception as e:
        print(f"Error in /recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
