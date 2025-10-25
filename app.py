from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask import send_file
from utils.intake_analyzer import analyze_intake
import base64
import cv2
import numpy as np
from fer import FER
from datetime import datetime
import re
import os
import tempfile
import soundfile as sf
import librosa
import whisper
import assemblyai as aai

app = Flask(__name__)

emotion_log = []  # Store detected emotions and transcripts

model = whisper.load_model("base")

@app.route('/')
def home():
    return redirect(url_for('intake'))

@app.route('/intake', methods=['GET'])
def intake():
    return render_template('intake.html')

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    data = request.get_json()
    image_data = data['image']

    img_str = re.sub('^data:image/.+;base64,', '', image_data)
    img_bytes = base64.b64decode(img_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detector = FER()
    top_emotion = detector.top_emotion(img)

    if top_emotion:
        emotion, score = top_emotion
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        emotion_log.append(f"{timestamp} - Face: {emotion}")
        return jsonify({'emotion': emotion})
    else:
        return jsonify({'emotion': 'No face detected'})

aai.settings.api_key = "b3432bac2f69408489d0a3a98b38eb2a"

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    path = "temp.wav"
    audio_file.save(path)

    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(path)

        text = transcript.text.strip()
        if not text:
            return jsonify({'error': 'Could not understand audio'}), 400

        emotion = infer_emotion_from_text(text)
        suggestion = generate_text_based_suggestion(text)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        emotion_log.append(f"{timestamp} - Voice: {emotion} - \"{text}\"")

        return jsonify({
            'transcript': text,
            'emotion': emotion,
            'suggestion': suggestion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_suggestion')
def get_suggestion():
    if not emotion_log:
        return jsonify({'suggestion': "No emotions recorded yet. Please start analyzing."})

    recent = emotion_log[-5:]
    voice_emotions = []
    face_emotions = []

    for entry in recent:
        if "Voice:" in entry:
            match = re.search(r'Voice: (\w+)', entry)
            if match:
                voice_emotions.append(match.group(1))
        elif "Face:" in entry:
            match = re.search(r'Face: (\w+)', entry)
            if match:
                face_emotions.append(match.group(1))

    voice = voice_emotions[-1] if voice_emotions else None
    face = face_emotions[-1] if face_emotions else None

    if face == "sad" or voice == "sad":
        suggestion = "You seem down. Want to talk about what’s bothering you?"
    elif face == "angry" or voice == "angry":
        suggestion = "It might help to slow down and take a deep breath. I'm here to listen."
    elif face == "happy" or voice == "happy":
        suggestion = "You seem happy! Let’s build on that feeling."
    elif face == "anxious" or voice == "anxious":
        suggestion = "You seem anxious. A few deep breaths or journaling might help."
    else:
        suggestion = "Keep going. Feel free to express anything on your mind."

    return jsonify({'suggestion': suggestion})

@app.route('/emotion_log')
def emotion_log_view():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Log</title>
        <style>
            body {{
                font-family: 'Poppins', sans-serif;
                padding: 40px;
                background-color: #f7f9fb;
                color: #333;
            }}
            h2 {{
                margin-bottom: 20px;
            }}
            ul {{
                background: #fff;
                border-radius: 10px;
                padding: 20px;
                list-style-type: none;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            li {{
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <h2>Logged Emotions During Video Session</h2>
        <ul>
            {''.join(f'<li>{e}</li>' for e in emotion_log)}
        </ul>
    </body>
    </html>
    """

@app.route("/getreport", methods=["GET", "POST"])
def get_report():
    report = None
    if request.method == "POST":
        report = "This is a sample psychological report based on your speech input."
    return render_template("getreport.html", report=report)

@app.route('/capture', methods=['POST'])
def capture():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if not ret:
        return jsonify({'error': 'Camera failed'}), 500

    detector = FER()
    top_emotion = detector.top_emotion(frame)

    if top_emotion:
        emotion, score = top_emotion
        emotion_log.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Face: {emotion}")
        return jsonify({'emotion': emotion})
    else:
        return jsonify({'emotion': 'No face detected'})

@app.route('/analyze', methods=['POST'])
def analyze():
    reason = request.form['reason']
    mood = request.form['mood']
    animal = request.form['animal']
    struggles = request.form['struggles']
    symptoms = request.form['symptoms']
    therapy = request.form['therapy']

    analysis = analyze_intake({
        'reason': reason,
        'mood': mood,
        'animal': animal,
        'struggles': struggles,
        'symptoms': symptoms,
        'therapy': therapy
    })

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Initial Analysis</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0;
                font-family: 'Poppins', sans-serif;
                background-color: #f0f4f8;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .container {{
                background: #ffffff;
                padding: 30px 40px;
                border-radius: 16px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
                max-width: 600px;
                width: 90%;
                text-align: center;
            }}
            h2 {{
                color: #333;
                font-weight: 600;
                margin-bottom: 20px;
            }}
            pre {{
                background-color: #f7f9fb;
                padding: 20px;
                border-radius: 10px;
                text-align: left;
                overflow-x: auto;
                color: #444;
                font-size: 15px;
                line-height: 1.6;
            }}
            button {{
                margin-top: 20px;
                padding: 12px 24px;
                font-size: 16px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s ease;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Initial Analysis</h2>
            <pre>{analysis}</pre>
            <form action="/video">
                <button type="submit">Start Video Session</button>
            </form>
        </div>
    </body>
    </html>
    """
@app.route('/transcript_log')
def transcript_log():
    transcripts = [entry for entry in emotion_log if "Voice:" in entry]
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Transcript Log</title>
        <style>
            body {{
                font-family: 'Poppins', sans-serif;
                padding: 40px;
                background-color: #f2f7fc;
                color: #333;
            }}
            h2 {{
                margin-bottom: 20px;
            }}
            ul {{
                background: #fff;
                border-radius: 10px;
                padding: 20px;
                list-style-type: none;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            li {{
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <h2>Transcript Log (Voice to Text)</h2>
        <ul>
            {''.join(f'<li>{t}</li>' for t in transcripts)}
        </ul>
    </body>
    </html>
    """


@app.route('/video')
def video():
    return render_template('video.html')

def generate_text_based_suggestion(text):
    text = text.lower()

    if "anxious" in text or "overwhelmed" in text:
        return "It sounds like you're feeling anxious. Would breathing exercises help right now?"
    elif "lonely" in text or "alone" in text:
        return "You mentioned feeling lonely. Let’s explore ways to feel more connected."
    elif "angry" in text or "frustrated" in text:
        return "Frustration is valid. Want to talk about what’s triggering it?"
    elif "happy" in text or "excited" in text:
        return "You seem happy! That’s great. Want to share what’s going well?"
    else:
        return "Thank you for sharing. Feel free to talk more—I’m here to listen."

def infer_emotion_from_text(text):
    text = text.lower()

    if any(word in text for word in ["angry", "mad", "furious", "annoyed"]):
        return "angry"
    elif any(word in text for word in ["sad", "down", "upset", "depressed"]):
        return "sad"
    elif any(word in text for word in ["happy", "joyful", "excited", "glad"]):
        return "happy"
    elif any(word in text for word in ["anxious", "nervous", "worried", "stressed"]):
        return "anxious"
    elif any(word in text for word in ["okay", "fine", "calm"]):
        return "calm"
    else:
        return "neutral"

if __name__ == '__main__':
    app.run(debug=True)
