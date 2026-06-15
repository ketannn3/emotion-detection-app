import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
import json
import time


def ensemble_predict(face_gray_norm, ensemble_models):
    """
    Fast ensemble for real-time: single prediction per model, averaged.
    TTA is intentionally skipped here to keep the camera smooth.
    TTA is used only in evaluate_model.py for accuracy measurement.
    """
    # CNN input: 48x48 grayscale
    gray_1ch = np.reshape(face_gray_norm, (1, 48, 48, 1))

    # MobileNet input: 96x96 RGB
    face_96 = cv2.resize(face_gray_norm, (96, 96))
    rgb_3ch = np.reshape(
        np.stack([face_96] * 3, axis=-1),
        (1, 96, 96, 3)
    )

    all_preds = []

    for name, model in ensemble_models.items():
        inp = rgb_3ch if "mobilenet" in name else gray_1ch
        all_preds.append(model.predict(inp, verbose=0))

    # Average across all models — still better than single model
    return np.mean(all_preds, axis=0)


def run_detection(primary_model=None, ensemble_models=None):

    # ── Fallback: load primary model if not passed ────────────────────────────
    if primary_model is None:
        print("No preloaded model — loading fallback...")
        primary_model = load_model("emotion_model_v2.hdf5", compile=False)

    if ensemble_models is None or len(ensemble_models) == 0:
        ensemble_models = {"cnn_v2": primary_model}

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # ── Build heatmap feature model from primary CNN ──────────────────────────
    last_conv = None
    for layer in reversed(primary_model.layers):
        if "conv" in layer.name:
            last_conv = layer.name
            break

    feature_model = tf.keras.models.Model(
        inputs=primary_model.inputs,
        outputs=primary_model.get_layer(last_conv).output
    )

    emotion_history = []
    timeline        = []
    start_time      = time.time()
    frame_count     = 0   # used to throttle MobileNet

    cap = cv2.VideoCapture(0)

    # Force window to foreground
    cv2.namedWindow("AI Emotion Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AI Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("AI Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("AI Emotion Detection", cv2.WND_PROP_TOPMOST, 1)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            # ── Prepare face input ────────────────────────────────────────────
            face_crop = gray[y:y+h, x:x+w]
            face_48   = cv2.resize(face_crop, (48, 48))
            face_norm = face_48 / 255.0

            # ── Ensemble prediction (MobileNet runs every 3rd frame only) ─────
            # Build a per-frame subset: always CNNs, MobileNet every 3rd frame
            active_models = {
                k: v for k, v in ensemble_models.items()
                if "mobilenet" not in k or frame_count % 3 == 0
            }
            # Fallback: if no models in active (shouldn't happen), use all
            if not active_models:
                active_models = ensemble_models

            avg_preds    = ensemble_predict(face_norm, active_models)
            emotion_idx  = np.argmax(avg_preds)
            emotion      = emotion_labels[emotion_idx]
            confidence   = avg_preds[0][emotion_idx] * 100

            emotion_history.append(emotion)
            timeline.append({
                "time":    int(time.time() - start_time),
                "emotion": emotion
            })

            # ── Heatmap (from primary CNN only) ───────────────────────────────
            face_input = np.reshape(face_norm, (1, 48, 48, 1))
            features   = feature_model.predict(face_input, verbose=0)

            heatmap = np.mean(features[0], axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap) + 1e-8
            heatmap  = cv2.resize(heatmap, (w, h))
            heatmap  = np.uint8(255 * heatmap)
            heatmap  = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(frame[y:y+h, x:x+w], 0.6, heatmap, 0.4, 0)
            frame[y:y+h, x:x+w] = overlay

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Show ensemble model count in label
            n = len(ensemble_models)
            label = f"{emotion} ({confidence:.1f}%) [x{n} ensemble]"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("AI Emotion Detection", frame)

        # Quit on Q key or window X button
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        if cv2.getWindowProperty("AI Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Build session result ──────────────────────────────────────────────────
    if len(emotion_history) == 0:
        emotion_history = ["Neutral"]

    emotion_count = Counter(emotion_history)
    total         = len(emotion_history)

    result = {
        e: round((c / total) * 100, 2)
        for e, c in emotion_count.items()
    }

    most_common = emotion_count.most_common(1)[0][0]

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
        "distribution": result,
        "analysis":     analysis,
        "timeline":     timeline
    }

    with open("session_result.json", "w") as f:
        json.dump(data, f)