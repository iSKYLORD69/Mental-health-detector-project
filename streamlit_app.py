import streamlit as st
import plotly.graph_objects as go
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
import time

load_dotenv()

st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0px);  }
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70%  { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);   }
    }
    .chat-container { animation: slideIn 0.5s ease-out; }
    .crisis-banner {
        animation: slideIn 0.4s ease-out, pulse 2s infinite;
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-radius: 12px;
        padding: 20px;
        color: white;
        margin-bottom: 20px;
    }
    .crisis-number {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 16px;
        font-weight: 500;
    }
    .mood-badge-crisis  { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
    .mood-badge-warning { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
    .mood-badge-normal  { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
    .suggest-box {
        animation: slideIn 0.5s ease-out;
        border-left: 4px solid #6366f1;
        padding: 16px;
        border-radius: 8px;
        margin-top: 16px;
        background: rgba(99,102,241,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Load ML model ──────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Please run train.py first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ── Groq client ────────────────────────────────────────────
@st.cache_resource
def load_groq():
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None
    return Groq(api_key=api_key)

groq_client = load_groq()

# ── Session state ──────────────────────────────────────────
# page controls which screen is shown:
# "analyze"  → main analysis form
# "results"  → results + suggest chatbot button
# "chat"     → chatbot
# "crisis"   → crisis chatbot
for key, default in {
    "page":              "analyze",
    "messages":          [],
    "detected_mood":     None,
    "user_text_context": "",
    "text_input":        "",
    "scores":            {},
    "confidence":        0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Build system prompt ────────────────────────────────────
def build_system_prompt(mood: str, user_context: str) -> str:
    context_line = (
        f"User's text (detected as '{mood}'): \"{user_context[:200]}\"\n\n"
        if user_context else ""
    )
    mood_instructions = {
        "normal":     "User seems okay. Be warm, ask what is on their mind.",
        "depression": "User shows depression signs. Validate feelings first. Be gentle. Suggest small steps.",
        "anxiety":    "User is anxious. Acknowledge it. Offer box breathing. Ask what worries them most.",
        "stress":     "User is stressed. Empathize. Help them prioritize. Suggest a short break.",
        "suicidal":   "CRISIS. User may be suicidal. Be present. Ask if they are safe. Share helplines every response: iCall 9152987821, AASRA 91-22-27546669, Tele MANAS 14416.",
        "bipolar":    "User shows bipolar signs. Be calm and steady. Ask how they feel right now.",
    }
    mood_key     = mood.lower()
    mood_section = mood_instructions.get(mood_key, mood_instructions["normal"])
    return f"""You are Mitra, a warm empathetic mental health support companion.
Be like a caring friend — calm, patient, non-judgmental.
Short conversational responses only. Never diagnose or prescribe.
{context_line}
Current mood: {mood}
Instruction: {mood_section}

If user mentions self harm or suicide always share:
iCall: 9152987821 | AASRA: 91-22-27546669 | Tele MANAS: 14416"""

# ── Get Groq response ──────────────────────────────────────
def get_groq_response(messages: list, mood: str, user_context: str) -> str:
    if groq_client is None:
        return (
            "I'm having trouble connecting right now. 💙\n\n"
            "📞 **iCall:** 9152987821\n"
            "📞 **Tele MANAS:** 14416 *(free · 24/7)*"
        )
    try:
        system_prompt = build_system_prompt(mood, user_context)
        api_messages  = [{"role": "system", "content": system_prompt}]
        for m in messages:
            if m["role"] in ["user", "assistant"]:
                api_messages.append({"role": m["role"], "content": m["content"]})
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=api_messages,
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"DEBUG Groq error: {e}")
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg or "rate" in error_msg:
            return "I need a short breather — please wait a moment and try again. 💙\n\n📞 **iCall:** 9152987821"
        if "api key" in error_msg or "invalid" in error_msg or "401" in error_msg:
            return "There's a configuration issue. Please check the API key."
        return "Something went wrong on my end. Please try again. 💙"

# ── Open chatbot ───────────────────────────────────────────
def open_chatbot(mood: str, user_context: str, crisis: bool = False):
    st.session_state.messages = []
    st.session_state.page     = "crisis" if crisis else "chat"

    if crisis:
        opening = (
            "I'm really glad you're here right now. 💙\n\n"
            "I can see you might be going through something very painful. "
            "You don't have to face this alone.\n\n"
            "Before anything else — are you safe right now?\n\n"
            "📞 **iCall:** 9152987821\n"
            "📞 **AASRA:** 91-22-27546669\n"
            "📞 **Vandrevala Foundation:** 1860-2662-345 *(24/7)*\n"
            "📞 **Tele MANAS:** 14416 *(free · 24/7)*\n\n"
            "I'm here. Take your time."
        )
    else:
        seed = [{
            "role": "user",
            "content": (
                f"The user wrote this text detected as '{mood}':\n"
                f"\"{user_context[:300]}\"\n\n"
                f"Greet them warmly, acknowledge ONE specific thing from their "
                f"text like a friend would, then ask one gentle open question. "
                f"Keep it short and warm."
                if user_context
                else "I just came to chat. Please greet me warmly."
            )
        }]
        opening = get_groq_response(seed, mood, user_context)

    st.session_state.messages.append({"role": "assistant", "content": opening})


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 MindScan")
    st.divider()

    if st.session_state.detected_mood:
        mood = st.session_state.detected_mood.lower()
        if mood == "suicidal":
            badge = f'<span class="mood-badge-crisis">🚨 {st.session_state.detected_mood.title()}</span>'
        elif mood in ["depression", "anxiety", "stress", "bipolar"]:
            badge = f'<span class="mood-badge-warning">⚠️ {st.session_state.detected_mood.title()}</span>'
        else:
            badge = f'<span class="mood-badge-normal">✅ {st.session_state.detected_mood.title()}</span>'
        st.markdown(f"**Last detected mood:**<br>{badge}", unsafe_allow_html=True)
        st.divider()

    page = st.session_state.page

    if page in ["chat", "crisis"]:
        if st.button("🔍 Back to Analysis", use_container_width=True):
            st.session_state.page = "analyze"
            st.rerun()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            open_chatbot(
                st.session_state.detected_mood or "normal",
                st.session_state.user_text_context,
                crisis=(page == "crisis")
            )
            st.rerun()
    elif page == "results":
        if st.button("🔍 New Analysis", use_container_width=True):
            st.session_state.page = "analyze"
            st.rerun()
        if st.button("💬 Open Support Chat", use_container_width=True, key="sidebar_chat"):
            open_chatbot(
                st.session_state.detected_mood,
                st.session_state.user_text_context,
                crisis=False
            )
            st.rerun()
    else:
        if st.session_state.detected_mood:
            if st.button("💬 Open Support Chat", use_container_width=True, key="sidebar_chat_analyze"):
                open_chatbot(
                    st.session_state.detected_mood,
                    st.session_state.user_text_context,
                    crisis=(st.session_state.detected_mood.lower() == "suicidal")
                )
                st.rerun()

    st.divider()
    st.subheader("Model Info 📊")
    st.caption("Training dataset: 53,000+ samples")
    st.caption("Categories: 7 mental health classes")
    st.caption("Model accuracy: 87%")
    st.caption("Algorithm: TF-IDF + Logistic Regression")
    st.caption("This Tool is for Detecting Mental Health.")
    st.caption("For emergencies call **112**")


# ══════════════════════════════════════════════════════════
# PAGE: CRISIS
# ══════════════════════════════════════════════════════════
if st.session_state.page == "crisis":

    st.markdown("""
    <div class="crisis-banner">
        <h3 style="margin:0 0 12px 0;">🚨 You are not alone</h3>
        <p style="margin:0 0 12px 0; opacity:0.9;">
            Please reach out to a crisis helpline right now.
            They are available 24/7 and calls are free.
        </p>
        <div class="crisis-number">📞 iCall &nbsp;&nbsp; <strong>9152987821</strong></div>
        <div class="crisis-number">📞 AASRA &nbsp;&nbsp; <strong>91-22-27546669</strong></div>
        <div class="crisis-number">📞 Vandrevala Foundation &nbsp;&nbsp; <strong>1860-2662-345</strong></div>
        <div class="crisis-number">📞 Tele MANAS &nbsp;&nbsp; <strong>14416</strong> &nbsp;(free · 24/7)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("You are safe here. Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        recovery_signals = [
            "feel better", "feeling better", "i'm okay", "i am okay",
            "thank you", "thanks", "that helped", "i feel safe",
            "i called", "i will call", "much better"
        ]
        user_recovering = any(s in prompt.lower() for s in recovery_signals)

        with st.chat_message("assistant"):
            with st.spinner(""):
                reply = get_groq_response(
                    st.session_state.messages,
                    "suicidal",
                    st.session_state.user_text_context
                )
            if user_recovering:
                st.session_state.page = "chat"
                reply += "\n\n💙 I'm really glad you're feeling a little better. I'm still right here with you."
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "chat":

    mood = st.session_state.detected_mood or "normal"

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.title("💬 Support Chat")
    st.caption(f"Chatting with context: **{mood.title()}** mood detected")
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        recovery_signals = [
            "feel better", "feeling better", "i'm okay", "i am okay",
            "thank you", "that helped", "much better", "i feel good"
        ]
        crisis_signals = [
            "want to die", "end my life", "kill myself",
            "no point living", "better off dead", "suicide"
        ]

        user_recovering = any(s in prompt.lower() for s in recovery_signals)
        user_in_crisis  = any(s in prompt.lower() for s in crisis_signals)

        if user_in_crisis:
            st.session_state.detected_mood = "suicidal"
            open_chatbot("suicidal", st.session_state.user_text_context, crisis=True)
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner(""):
                current_mood = "normal" if user_recovering else mood
                reply = get_groq_response(
                    st.session_state.messages,
                    current_mood,
                    st.session_state.user_text_context
                )
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        if user_recovering:
            st.session_state.detected_mood = "normal"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    prediction = st.session_state.detected_mood
    scores     = st.session_state.scores
    confidence = st.session_state.confidence

    st.title("🔍 Results")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Category", prediction.replace("_", " ").title())
    col2.metric("Confidence",        f"{confidence * 100:.1f}%")
    col3.metric("Words Analyzed",    len(st.session_state.user_text_context.split()))

    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    colors = ["#ef4444" if k == prediction else "#94a3b8" for k in sorted_scores]
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

    interpretations = {
        "normal":               ("✅ No significant distress signals detected.",          "success"),
        "depression":           ("🔴 Signals associated with depression detected.",      "error"),
        "anxiety":              ("🟠 Signals associated with anxiety detected.",         "warning"),
        "suicidal":             ("🚨 High risk signals detected. Please seek help now.", "error"),
        "stress":               ("🟡 Signals associated with stress detected.",          "warning"),
        "bipolar":              ("🟠 Signals associated with bipolar mood detected.",    "warning"),
        "personality disorder": ("🔴 Signals of personality disorder detected.",        "error"),
    }
    msg, msg_type = interpretations.get(
        prediction.lower(), (f"Category detected: {prediction.title()}", "info")
    )
    if msg_type == "success":
        st.success(msg)
    elif msg_type == "error":
        st.error(msg)
    elif msg_type == "warning":
        st.warning(msg)
    else:
        st.info(msg)

    st.divider()
    st.markdown("""
    <div class="suggest-box">
        <strong>💬 Want to talk about this?</strong><br><br>
        Our support chatbot has read your text and is ready to listen.
        It understands what you shared and can help you work through it.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("💬 Open Support Chat", type="primary", use_container_width=True):
            open_chatbot(
                st.session_state.detected_mood,
                st.session_state.user_text_context,
                crisis=False
            )
            st.rerun()
    with col2:
        if st.button("🔍 Analyze New Text", use_container_width=True):
            st.session_state.page = "analyze"
            st.rerun()

    if prediction.lower() in ["depression", "bipolar"]:
        st.divider()
        st.subheader("📞 Crisis Resources")
        st.info("""
        **iCall:** 9152987821
        **AASRA:** 91-22-27546669
        **Vandrevala Foundation:** 1860-2662-345 *(24/7)*
        **Tele MANAS:** 14416 *(free · 24/7)*
        """)


# ══════════════════════════════════════════════════════════
# PAGE: ANALYZE
# ══════════════════════════════════════════════════════════
else:

    st.title("🔍 Mental Health Detection from Text")
    st.caption("NLP-powered tool to identify emotional distress signals in written text.")
    st.warning("A Tool for detecting mental health of a person by analyzing their text.")
    st.divider()

    st.subheader("Try an example")
    col1, col2, col3 = st.columns(3)
    examples = {
        "😔 Depression": "I have been feeling completely empty for weeks. Nothing brings me joy anymore and I struggle to get out of bed every morning.",
        "😰 Anxiety":    "I cannot stop worrying about everything. My heart races constantly and I feel like something terrible is about to happen.",
        "😊 Normal":     "Had a great day today! Went for a walk, cooked a nice meal and caught up with some old friends. Feeling grateful."
    }
    if col1.button("😔 Depression", use_container_width=True):
        st.session_state.text_input = examples["😔 Depression"]
        st.rerun()
    if col2.button("😰 Anxiety", use_container_width=True):
        st.session_state.text_input = examples["😰 Anxiety"]
        st.rerun()
    if col3.button("😊 Normal", use_container_width=True):
        st.session_state.text_input = examples["😊 Normal"]
        st.rerun()

    st.divider()
    st.subheader("Or enter your own text")
    text_input = st.text_area(
        label="",
        height=160,
        value=st.session_state.get("text_input", ""),
        placeholder="e.g. I have been feeling overwhelmed lately..."
    )

    if st.button("Analyze Text", type="primary", use_container_width=True):

        if not text_input.strip():
            st.error("Please enter some text first.")
            st.stop()

        with st.spinner("Analyzing..."):
            prediction    = model.predict([text_input])[0]
            probabilities = model.predict_proba([text_input])[0]
            labels        = model.classes_
            scores        = {
                label: round(float(prob), 4)
                for label, prob in zip(labels, probabilities)
            }
            confidence = round(float(max(probabilities)), 4)

        # Save to session state
        st.session_state.detected_mood     = prediction
        st.session_state.user_text_context = text_input
        st.session_state.scores            = scores
        st.session_state.confidence        = confidence

        # Auto open crisis chat for suicidal
        if prediction.lower() == "suicidal":
            with st.spinner("Connecting you to support..."):
                time.sleep(1.2)
            open_chatbot("suicidal", text_input, crisis=True)
            st.rerun()

        # Go to results page for everything else
        else:
            st.session_state.page = "results"
            st.rerun()

    st.divider()
    st.caption("Built with Python · scikit-learn · Groq AI · Streamlit | For educational purposes only.")
