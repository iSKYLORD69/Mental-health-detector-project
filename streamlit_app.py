import streamlit as st
import plotly.graph_objects as go
import joblib
import os

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Please run train.py first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ── Header ─────────────────────────────────────────────────
st.title("🧠 Mental Health Detection from Text")
st.caption("NLP-powered tool to identify emotional distress signals in written text.")
st.warning("⚠️ For educational and research purposes only. Not a clinical tool.")

st.divider()

# ── Example buttons ────────────────────────────────────────
st.subheader("Try an example")

col1, col2, col3 = st.columns(3)

examples = {
    "😔 Depression":    "I have been feeling completely empty for weeks. Nothing brings me joy anymore and I struggle to get out of bed every morning.",
    "😰 Anxiety":       "I cannot stop worrying about everything. My heart races constantly and I feel like something terrible is about to happen.",
    "😊 Normal":        "Had a great day today! Went for a walk, cooked a nice meal and caught up with some old friends. Feeling grateful."
}

if col1.button("😔 Depression", use_container_width=True):
    st.session_state.text_input = examples["😔 Depression"]

if col2.button("😰 Anxiety", use_container_width=True):
    st.session_state.text_input = examples["😰 Anxiety"]

if col3.button("😊 Normal", use_container_width=True):
    st.session_state.text_input = examples["😊 Normal"]

st.divider()

# ── Text input ─────────────────────────────────────────────
st.subheader("Or enter your own text")

text_input = st.text_area(
    label="",
    height=160,
    value=st.session_state.get("text_input", ""),
    placeholder="e.g. I have been feeling overwhelmed lately and nothing seems to help..."
)

# ── Analyze ────────────────────────────────────────────────
if st.button("Analyze Text", type="primary", use_container_width=True):

    if not text_input.strip():
        st.error("Please enter some text first.")
        st.stop()

    with st.spinner("Analyzing..."):
        prediction    = model.predict([text_input])[0]
        probabilities = model.predict_proba([text_input])[0]
        labels        = model.classes_

        scores = {
            label: round(float(prob), 4)
            for label, prob in zip(labels, probabilities)
        }
        confidence = round(float(max(probabilities)), 4)

    st.divider()
    st.subheader("Results")

    # ── Metrics ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Category", prediction.replace("_", " ").title())
    col2.metric("Confidence",        f"{confidence * 100:.1f}%")
    col3.metric("Words Analyzed",    len(text_input.split()))

    # ── Bar chart ──────────────────────────────────────────
    sorted_scores = dict(
        sorted(scores.items(), key=lambda x: x[1], reverse=True)
    )

    colors = [
        "#ef4444" if k == prediction else "#94a3b8"
        for k in sorted_scores
    ]

    fig = go.Figure(go.Bar(
        x=[k.replace("_", " ").title() for k in sorted_scores.keys()],
        y=[v * 100 for v in sorted_scores.values()],
        marker_color=colors,
        text=[f"{v * 100:.1f}%" for v in sorted_scores.values()],
        textposition="outside"
    ))

    fig.update_layout(
        title="Confidence Scores by Category",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 110],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Interpretation ─────────────────────────────────────
    st.subheader("Interpretation")

    interpretations = {
        "normal":     ("✅ No significant distress signals detected.",      "success"),
        "depression": ("🔴 Signals associated with depression detected.",   "error"),
        "anxiety":    ("🟠 Signals associated with anxiety detected.",      "warning"),
        "suicidal":   ("🚨 High risk signals detected. Please seek help.",  "error"),
        "stress":     ("🟡 Signals associated with stress detected.",       "warning"),
        "bipolar":    ("🟠 Signals associated with bipolar mood detected.", "warning"),
        "personality disorder": ("🔴 Signals of personality disorder detected.", "error"),
    }

    label_key = prediction.lower()
    message, msg_type = interpretations.get(
        label_key,
        (f"Category detected: {prediction.title()}", "info")
    )

    if msg_type == "success":
        st.success(message)
    elif msg_type == "error":
        st.error(message)
    elif msg_type == "warning":
        st.warning(message)
    else:
        st.info(message)

    # ── Crisis resources ───────────────────────────────────
    if prediction.lower() in ["suicidal", "depression"]:
        st.divider()
        st.subheader("📞 Crisis Resources")
        st.info("""
        **iCall (India):** 9152987821
        **Vandrevala Foundation:** 1860-2662-345 (24/7)
        **AASRA:** 91-22-27546669
        
        If you or someone you know is in crisis, please reach out immediately.
        """)

st.divider()
st.caption("Built with Python · scikit-learn · Streamlit | For educational purposes only.")