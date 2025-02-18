from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from deepface import DeepFace
import numpy as np
import base64
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download, login

app = Flask(__name__)
CORS(app)

# Secure Hugging Face authentication token (must be set in Render environment variables)
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Login to Hugging Face if token is provided
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)

# Ensure TensorFlow/PyTorch uses CPU to avoid memory overload in limited environments
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU usage for TensorFlow

##############################################
# IMAGE-BASED EMOTION DETECTION (DeepFace API)
##############################################
@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        # Decode the base64 image data
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Analyze the face for emotion using DeepFace
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Validate response
        if not analysis or 'dominant_emotion' not in analysis[0]:
            return jsonify({'error': 'Could not detect emotion'}), 500

        dominant_emotion = analysis[0]['dominant_emotion']
        emotion_confidence = analysis[0]['emotion'].get(dominant_emotion, 0)
        print(f"Detected Emotion (image): {dominant_emotion}, Confidence: {emotion_confidence}")
        
        return jsonify({
            'time': time.time(),
            'dominant_emotion': dominant_emotion,
            'emotion_confidence': float(emotion_confidence)
        })
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

##################################################
# TEXT-BASED EMOTION DETECTION (Transformers API)
##################################################

# Use Hugging Face repo instead of local path
MODEL_NAME = "BSNSSWB/emotion-model"  # Replace with your actual Hugging Face model repo

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_auth_token=HUGGINGFACE_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer, model = None, None  # Avoids crashes

@app.route('/analyze_text', methods=['POST'])
def analyze_text_emotion():
    if tokenizer is None or model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    try:
        text = data['text']
        # Tokenize the input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Compute probabilities using softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_list = probs.tolist()
        predicted_index = torch.argmax(probs).item()
        dominant_emotion = model.config.id2label[predicted_index]
        predicted_probability = probs_list[0][predicted_index]
        print(f"Detected Emotion (text): {dominant_emotion}, Confidence: {predicted_probability}")

        return jsonify({
            'time': time.time(),
            'dominant_emotion': dominant_emotion,
            'emotion_confidence': predicted_probability
        })
    except Exception as e:
        print(f"Error in text analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure correct port binding for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
