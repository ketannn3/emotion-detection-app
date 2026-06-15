from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
import time

app = Flask(__name__)

# ── PRELOAD ALL MODELS ONCE AT STARTUP ────────────────────────────────────────

def try_load(path):
    try:
        m = load_model(path, compile=False)
        print(f"  Loaded: {path}")
        return m
    except Exception as e:
        print(f"  Skipped: {path} ({e})")
        return None

print("Preloading emotion models...")

model_v2        = try_load("emotion_model_v2.hdf5")
model_v3        = try_load("emotion_model_v3.keras")
model_mobilenet = try_load("mobilenet_finetune.keras")

if model_v2:        model_v2.predict(np.zeros((1, 48, 48, 1)), verbose=0)
if model_v3:        model_v3.predict(np.zeros((1, 48, 48, 1)), verbose=0)
if model_mobilenet: model_mobilenet.predict(np.zeros((1, 96, 96, 3)), verbose=0)

ensemble_models = {k: v for k, v in {
    "cnn_v2":    model_v2,
    "cnn_v3":    model_v3,
    "mobilenet": model_mobilenet,
}.items() if v is not None}

print(f"Ensemble ready with {len(ensemble_models)} model(s): {list(ensemble_models.keys())}")

primary_model = model_v2

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Build heatmap feature model from primary CNN
last_conv = None
for layer in reversed(primary_model.layers):
    if "conv" in layer.name:
        last_conv = layer.name
        break
feature_model = tf.keras.models.Model(
    inputs=primary_model.inputs,
    outputs=primary_model.get_layer(last_conv).output
)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Browser session state (in-memory)
browser_session = {
    "emotion_history": [],
    "timeline":        [],
    "start_time":      None
}


# ── ENSEMBLE PREDICT ──────────────────────────────────────────────────────────

def ensemble_predict(face_norm_48):
    gray_1ch = np.reshape(face_norm_48, (1, 48, 48, 1))
    face_96  = cv2.resize(face_norm_48, (96, 96))
    rgb_3ch  = np.reshape(np.stack([face_96]*3, axis=-1), (1, 96, 96, 3))
    preds = []
    for name, m in ensemble_models.items():
        inp = rgb_3ch if "mobilenet" in name else gray_1ch
        preds.append(m.predict(inp, verbose=0))
    return np.mean(preds, axis=0)


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- CAMERA (laptop) ----------------

@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/start_camera")
def start_camera():
    from live_heatmap import run_detection
    run_detection(primary_model=primary_model, ensemble_models=ensemble_models)

    with open("session_result.json") as f:
        data = json.load(f)

    analysis = data["analysis"]
    emotion  = max(data["distribution"], key=data["distribution"].get)
    return render_template("assistant.html", analysis=analysis, emotion=emotion)


# ---------------- BROWSER CAMERA ----------------

@app.route("/browser_camera")
def browser_camera():
    # Reset browser session
    browser_session["emotion_history"] = []
    browser_session["timeline"]        = []
    browser_session["start_time"]      = time.time()
    return render_template("browser_camera.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        file  = request.files["frame"]
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid frame"})

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"error": "No face"})

        x, y, w, h = faces[0]

        # Prepare face
        face_crop = gray[y:y+h, x:x+w]
        face_48   = cv2.resize(face_crop, (48, 48))
        face_norm = face_48 / 255.0

        # Ensemble predict
        avg_preds   = ensemble_predict(face_norm)
        emotion_idx = int(np.argmax(avg_preds))
        emotion     = emotion_labels[emotion_idx]
        confidence  = float(avg_preds[0][emotion_idx] * 100)

        # Record in session
        browser_session["emotion_history"].append(emotion)
        elapsed = int(time.time() - (browser_session["start_time"] or time.time()))
        browser_session["timeline"].append({"time": elapsed, "emotion": emotion})

        # Generate heatmap
        face_input = np.reshape(face_norm, (1, 48, 48, 1))
        features   = feature_model.predict(face_input, verbose=0)
        heatmap    = np.mean(features[0], axis=-1)
        heatmap    = np.maximum(heatmap, 0)
        heatmap   /= np.max(heatmap) + 1e-8
        heatmap    = cv2.resize(heatmap, (w, h))
        heatmap    = np.uint8(255 * heatmap)
        heatmap    = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Encode heatmap as base64 PNG
        _, buf     = cv2.imencode(".png", heatmap)
        heatmap_b64 = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "emotion":    emotion,
            "confidence": confidence,
            "face_x":     int(x),
            "face_y":     int(y),
            "face_w":     int(w),
            "face_h":     int(h),
            "heatmap_img": heatmap_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/finish_browser_session", methods=["POST"])
def finish_browser_session():
    try:
        history = browser_session["emotion_history"]

        if not history:
            history = ["Neutral"]

        emotion_count = Counter(history)
        total         = len(history)
        distribution  = {e: round((c/total)*100, 2) for e, c in emotion_count.items()}
        most_common   = emotion_count.most_common(1)[0][0]

        analysis_map = {
            "Happy":    "You appeared mostly happy and positive.",
            "Sad":      "You seemed a bit sad during the session.",
            "Neutral":  "You appeared calm and neutral.",
            "Angry":    "You showed signs of frustration.",
            "Fear":     "You seemed slightly anxious.",
            "Surprise": "You showed moments of surprise.",
            "Disgust":  "Mixed emotions were detected.",
        }
        analysis = analysis_map.get(most_common, "Mixed emotions were detected.")

        data = {
            "distribution": distribution,
            "analysis":     analysis,
            "timeline":     browser_session["timeline"]
        }

        with open("session_result.json", "w") as f:
            json.dump(data, f)

        return jsonify({"status": "ok", "emotion": most_common})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/browser_assistant")
def browser_assistant():
    emotion = request.args.get("emotion", "Neutral")
    with open("session_result.json") as f:
        data = json.load(f)
    analysis = data.get("analysis", "")
    return render_template("assistant.html", analysis=analysis, emotion=emotion)


# ---------------- SAVE ANSWERS ----------------

@app.route("/save_answers", methods=["POST"])
def save_answers():
    try:
        payload = request.get_json()
        with open("session_result.json") as f:
            data = json.load(f)
        data["qa_pairs"]          = payload.get("qa_pairs", [])
        data["therapist_summary"] = payload.get("therapist_summary", "")
        data["detected_emotion"]  = payload.get("detected_emotion", "")
        with open("session_result.json", "w") as f:
            json.dump(data, f)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------- UPLOAD ----------------

@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files["image"]
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img  = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    result_emotion = "No face detected"
    for (x, y, w, h) in faces:
        roi    = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
        roi_96 = cv2.resize(roi, (96, 96))
        gray_input = np.reshape(roi, (1, 48, 48, 1))
        rgb_input  = np.reshape(np.stack([roi_96]*3, axis=-1), (1, 96, 96, 3))
        preds = []
        for name, m in ensemble_models.items():
            preds.append(m.predict(rgb_input if "mobilenet" in name else gray_input, verbose=0))
        result_emotion = emotion_labels[np.argmax(np.mean(preds, axis=0))]
        break

    return render_template("result.html", data={
        "distribution": {result_emotion: 100},
        "analysis": f"Detected emotion from uploaded image: {result_emotion}"
    })


# ---------------- AI RESPONSE ----------------

@app.route("/ai_response", methods=["POST"])
def ai_response():
    user_text = request.json["message"].lower()
    if "hello" in user_text or "hi" in user_text:
        reply = "Hello. I'm glad you're here. How are you feeling today?"
    elif "happy" in user_text:
        reply = "That's wonderful to hear. What made you feel happy today?"
    elif "sad" in user_text:
        reply = "I'm sorry you're feeling sad. Would you like to talk about it?"
    elif "angry" in user_text:
        reply = "Feeling angry can be difficult. What caused that feeling?"
    else:
        reply = "I understand. Can you tell me more?"
    return jsonify({"reply": reply})


# ---------------- RESULT ----------------

@app.route("/result")
def result():
    with open("session_result.json") as f:
        data = json.load(f)
    return render_template("result.html", data=data)


# ---------------- DOWNLOAD REPORT ----------------

@app.route("/download_report")
def download_report():
    from generate_report import generate_pdf
    generate_pdf()
    return send_file("emotion_report.pdf", as_attachment=True)


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
