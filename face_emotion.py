# face_emotion.py
import numpy as np
def get_fer_detector():
    from fer import FER
    return FER(mtcnn=False)

# ── Emotion → Mental Health mapping ────────────────────────
EMOTION_TO_MENTAL = {
    "sad":      {"depression": 0.80, "stress": 0.30},
    "angry":    {"stress": 0.75,     "anxiety": 0.45, "bipolar": 0.30},
    "fear":     {"anxiety": 0.90,    "stress": 0.50},
    "disgust":  {"depression": 0.45, "stress": 0.40},
    "happy":    {"normal": 0.85},
    "neutral":  {"normal": 0.60,     "depression": 0.15},
    "surprise": {"anxiety": 0.35},
}

LABEL_DISPLAY = {
    "depression": "Depression",
    "anxiety":    "Anxiety",
    "stress":     "Stress",
    "bipolar":    "Bipolar",
    "normal":     "Normal",
    "suicidal":   "Suicidal",
}

def get_fer_detector():
    """Load FER detector (cached by caller)."""
    return FER(mtcnn=False)   # mtcnn=False = lighter, faster, works on cloud

def average_emotion_frames(frame_list: list[dict]) -> dict:
    """
    Takes a list of per-frame emotion dicts
    e.g. [{'happy':0.9,'sad':0.05,...}, ...]
    Returns averaged scores.
    """
    if not frame_list:
        return {}
    keys = frame_list[0].keys()
    return {
        k: float(np.mean([f.get(k, 0.0) for f in frame_list]))
        for k in keys
    }

def map_to_mental_health(emotion_scores: dict) -> tuple[str, dict, float]:
    """
    Maps averaged FER emotion scores → mental health prediction + scores.
    Returns (prediction_label, scores_dict, confidence)
    """
    mental_scores: dict[str, float] = {}

    for emotion, score in emotion_scores.items():
        if score < 0.05:          # ignore noise
            continue
        mapping = EMOTION_TO_MENTAL.get(emotion, {})
        for condition, weight in mapping.items():
            mental_scores[condition] = (
                mental_scores.get(condition, 0.0) + score * weight
            )

    if not mental_scores:
        return "normal", {"normal": 1.0}, 1.0

    # Normalize to sum = 1
    total = sum(mental_scores.values())
    scores_norm = {k: round(v / total, 4) for k, v in mental_scores.items()}

    prediction  = max(scores_norm, key=scores_norm.get)
    confidence  = round(scores_norm[prediction], 4)

    return prediction, scores_norm, confidence

def describe_face_scan(emotion_scores: dict, prediction: str) -> str:
    """
    Returns a short natural-language summary for Mello's context.
    """
    dominant = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
    pct      = int(emotion_scores.get(dominant, 0) * 100)
    return (
        f"User's facial scan showed primarily '{dominant}' ({pct}%) "
        f"across the session, mapped to '{prediction}' mental health signal."
    )