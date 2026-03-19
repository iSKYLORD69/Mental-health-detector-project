import streamlit as st
import plotly.graph_objects as go
import joblib
import os
import anthropic
from dotenv import load_dotenv
import time

load_dotenv()

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Smooth slide-in animation for chatbot */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0px);  }
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70%  { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);   }
    }
    .chat-container {
        animation: slideIn 0.5s ease-out;
    }
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
    .mood-badge-crisis   { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
    .mood-badge-warning  { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
    .mood-badge-normal   { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
    .suggest-box {
        animation: slideIn 0.5s ease-out;
        border-left: 4px solid #6366f1;
        padding: 16px;
        border-radius: 8px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Please run train.py first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ── Anthropic client ───────────────────────────────────────
@st.cache_resource
def load_client():
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

client = load_client()

# ── Session state defaults ─────────────────────────────────
if "chatbot_open"       not in st.session_state:
    st.session_state.chatbot_open       = False
if "messages"           not in st.session_state:
    st.session_state.messages           = []
if "detected_mood"      not in st.session_state:
    st.session_state.detected_mood      = None
if "user_text_context"  not in st.session_state:
    st.session_state.user_text_context  = ""
if "crisis_mode"        not in st.session_state:
    st.session_state.crisis_mode        = False
if "crisis_prompt_done" not in st.session_state:
    st.session_state.crisis_prompt_done = False
if "text_input"         not in st.session_state:
    st.session_state.text_input         = ""

# ── Helper — build system prompt based on mood ────────────
def build_system_prompt(mood: str, user_context: str) -> str:

    context_line = (
        f"The user previously shared this text which was analyzed and detected as '{mood}':\n"
        f"\"{user_context}\"\n\n"
        f"Use this as context to understand what they might be going through. "
        f"Reference it naturally if relevant — don't just repeat it back to them.\n\n"
        if user_context else ""
    )

    base = f"""{context_line}You are a warm, empathetic mental health support companion named Mitra.
Your personality: calm, patient, non-judgmental, gently encouraging.
You speak like a caring friend — not a robot, not a therapist.
Use simple conversational language. Short paragraphs. No bullet points unless listing coping steps.
Never diagnose. Never prescribe. Always remind gently that professional help is available.

If the user expresses worsening suicidal thoughts at any point, immediately share:
- iCall: 9152987821
- AASRA: 91-22-27546669  
- Vandrevala Foundation: 1860-2662-345
- Tele MANAS: 14416
And strongly encourage them to call right now.
"""

    mood_instructions = {
        "normal": """
The user seems to be doing okay but reached out anyway — that takes courage.
Be warm and welcoming. Ask open-ended questions about their day or what's on their mind.
Help them reflect on what's going well and what could be better.
Suggest simple positive habits if relevant: journaling, gratitude, walks.
""",
        "depression": """
The user is showing signs of depression. They may feel empty, hopeless or disconnected.
Validate their feelings first — never rush to fix or advise.
Use phrases like 'that sounds really heavy' or 'it makes sense you feel that way'.
Gently explore: how long have they felt this way, do they have support around them.
Suggest small achievable steps: getting sunlight, one small task, texting a friend.
Remind them depression is not a character flaw — it is something many people face and recover from.
""",
        "anxiety": """
The user is experiencing anxiety. They may feel overwhelmed, restless or fearful.
Acknowledge how exhausting anxiety can be. Normalize it without minimizing it.
Offer a simple grounding technique early: the 5-4-3-2-1 method or box breathing.
Ask what specifically is worrying them most right now.
Help them separate what is in their control from what isn't.
""",
        "stress": """
The user is stressed — likely from external pressures like work, college or relationships.
Empathize with how overwhelming it can feel when everything piles up.
Ask what is causing the most stress right now.
Help them prioritize and break things into smaller steps.
Suggest a short break, physical movement or talking to someone they trust.
""",
        "suicidal": """
CRITICAL: This user is in crisis. They may be thinking about ending their life.
Your ONLY priority right now is their safety.
Be extremely gentle, present and non-judgmental.
Do NOT give advice or try to problem-solve. Just be with them.
Start by saying you are glad they are here and talking.
Ask if they are safe right now.
Gently but firmly encourage them to call a helpline immediately.
Repeat the numbers if they don't respond to the first prompt.
If they say they won't call, ask if there is one person nearby they can be with right now.
Every response must end with a crisis number until they confirm they are safe.
""",
        "bipolar": """
The user may be experiencing mood swings associated with bipolar patterns.
Be steady and calm in your tone — an anchor for them.
Ask how they are feeling right now in this moment.
Avoid making assumptions about their current phase.
Gently encourage them to stay connected with any mental health professional they may have.
""",
    }

    mood_key     = mood.lower().replace(" ", "_").replace("personality_disorder", "depression")
    mood_section = mood_instructions.get(mood_key, mood_instructions["normal"])

    return base + mood_section


# ── Helper — get Claude response ───────────────────────────
def get_claude_response(messages: list, mood: str, user_context: str) -> str:
    if client is None:
        return "I'm having trouble connecting right now. Please try again in a moment."

    system = build_system_prompt(mood, user_context)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ["user", "assistant"]
    ]

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system,
        messages=api_messages
    )
    return response.content[0].text


# ── Helper — open chatbot with first message ───────────────
def open_chatbot(mood: str, user_context: str, crisis: bool = False):
    st.session_state.chatbot_open      = True
    st.session_state.detected_mood     = mood
    st.session_state.user_text_context = user_context
    st.session_state.crisis_mode       = crisis
    st.session_state.messages          = []

    if crisis:
        opening = (
            "I'm really glad you're here right now. 💙\n\n"
            "I can see you might be going through something very painful. "
            "You don't have to face this alone.\n\n"
            "Before anything else — are you safe right now?\n\n"
            "Please know that help is just a call away:\n"
            "📞 **iCall:** 9152987821\n"
            "📞 **AASRA:** 91-22-27546669\n"
            "📞 **Vandrevala Foundation:** 1860-2662-345 *(24/7)*\n"
            "📞 **Tele MANAS:** 14416 *(free, 24/7)*\n\n"
            "I'm here with you. Take your time."
        )
    else:
        # Generate a context-aware opening using Claude
        seed_messages = [{
            "role": "user",
            "content": (
                f"I just finished analyzing my text and it was detected as '{mood}'. "
                f"The text I wrote was: \"{user_context}\". "
                f"Please greet me warmly and acknowledge what I shared."
                if user_context
                else f"I just came to chat. My mood seems to be '{mood}'."
            )
        }]
        opening = get_claude_response(seed_messages, mood, user_context)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": opening
    })


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

    if st.session_state.chatbot_open:
        if st.button("🔍 Back to Analysis", use_container_width=True):
            st.session_state.chatbot_open = False
            st.rerun()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            mood    = st.session_state.detected_mood or "normal"
            context = st.session_state.user_text_context
            crisis  = st.session_state.crisis_mode
            open_chatbot(mood, context, crisis)
            st.rerun()
    else:
        if st.button("💬 Open Support Chat", use_container_width=True):
            mood    = st.session_state.detected_mood or "normal"
            context = st.session_state.user_text_context
            open_chatbot(mood, context, mood == "suicidal")
            st.rerun()

    st.divider()
    st.caption("⚠️ Not a clinical tool.")
    st.caption("For emergencies call **112**")


# ══════════════════════════════════════════════════════════
# CRISIS MODE — auto open, full screen
# ══════════════════════════════════════════════════════════
if st.session_state.crisis_mode and st.session_state.chatbot_open:

    # Crisis banner
    st.markdown("""
    <div class="crisis-banner">
        <h3 style="margin:0 0 12px 0;">🚨 You are not alone</h3>
        <p style="margin:0 0 12px 0; opacity:0.9;">Please reach out to a crisis helpline right now. They are available 24/7 and calls are free.</p>
        <div class="crisis-number">📞 iCall &nbsp;&nbsp; <strong>9152987821</strong></div>
        <div class="crisis-number">📞 AASRA &nbsp;&nbsp; <strong>91-22-27546669</strong></div>
        <div class="crisis-number">📞 Vandrevala Foundation &nbsp;&nbsp; <strong>1860-2662-345</strong></div>
        <div class="crisis-number">📞 Tele MANAS &nbsp;&nbsp; <strong>14416</strong> &nbsp; (free · 24/7)</div>
    </div>
    """, unsafe_allow_html=True)

    # Chat window
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("You are safe here. Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if user seems to be recovering
        recovery_signals = [
            "feel better", "feeling better", "i'm okay", "i am okay",
            "thank you", "thanks", "that helped", "i feel safe",
            "i called", "i will call", "okay i will"
        ]
        user_recovering = any(s in prompt.lower() for s in recovery_signals)

        with st.chat_message("assistant"):
            with st.spinner(""):
                reply = get_claude_response(
                    st.session_state.messages,
                    "suicidal",
                    st.session_state.user_text_context
                )

            # If user seems better, soften crisis mode gently
            if user_recovering:
                st.session_state.crisis_mode = False
                reply += (
                    "\n\n💙 I'm really glad you're feeling a bit better. "
                    "I'm still here with you — take all the time you need."
                )

            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CHATBOT PAGE — non-crisis
# ══════════════════════════════════════════════════════════
elif st.session_state.chatbot_open:

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

        # Check for recovery or worsening
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

        # Escalate to crisis if needed
        if user_in_crisis:
            st.session_state.crisis_mode   = True
            st.session_state.detected_mood = "suicidal"
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner(""):
                current_mood = "normal" if user_recovering else mood
                reply = get_claude_response(
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
# ANALYSIS PAGE
# ══════════════════════════════════════════════════════════
else:

    st.title("🔍 Mental Health Detection from Text")
    st.caption("NLP-powered tool to identify emotional distress signals in written text.")
    st.warning("⚠️ For educational and research purposes only. Not a clinical tool.")
    st.divider()

    # ── Example buttons ────────────────────────────────────
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

    # ── Text input ─────────────────────────────────────────
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

        # Store for chatbot context
        st.session_state.detected_mood     = prediction
        st.session_state.user_text_context = text_input

        st.divider()
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Category", prediction.replace("_", " ").title())
        col2.metric("Confidence",        f"{confidence * 100:.1f}%")
        col3.metric("Words Analyzed",    len(text_input.split()))

        # ── Bar chart ──────────────────────────────────────
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

        # ── Interpretation ─────────────────────────────────
        st.subheader("Interpretation")
        interpretations = {
            "normal":               ("✅ No significant distress signals detected.",             "success"),
            "depression":           ("🔴 Signals associated with depression detected.",         "error"),
            "anxiety":              ("🟠 Signals associated with anxiety detected.",            "warning"),
            "suicidal":             ("🚨 High risk signals detected. Please seek help now.",    "error"),
            "stress":               ("🟡 Signals associated with stress detected.",             "warning"),
            "bipolar":              ("🟠 Signals associated with bipolar mood detected.",       "warning"),
            "personality disorder": ("🔴 Signals of personality disorder detected.",           "error"),
        }
        label_key        = prediction.lower()
        message, msg_type = interpretations.get(
            label_key, (f"Category detected: {prediction.title()}", "info")
        )
        if msg_type == "success":
            st.success(message)
        elif msg_type == "error":
            st.error(message)
        elif msg_type == "warning":
            st.warning(message)
        else:
            st.info(message)

        # ── Suicidal → auto open chatbot ───────────────────
        if prediction.lower() == "suicidal":
            st.divider()
            with st.spinner("Opening support chat..."):
                time.sleep(1.2)
            open_chatbot("suicidal", text_input, crisis=True)
            st.rerun()

        # ── Other moods → suggest chatbot ──────────────────
        else:
            st.divider()
            st.markdown("""
            <div class="suggest-box">
                <strong>💬 Want to talk about this?</strong><br>
                Our support chatbot has read your text and is ready to listen.
                It understands what you shared and can help you work through it.
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button(
                    "💬 Open Support Chat",
                    type="primary",
                    use_container_width=True
                ):
                    open_chatbot(prediction, text_input, crisis=False)
                    st.rerun()
            with col2:
                if st.button("Maybe later", use_container_width=True):
                    st.info("That's okay. The chat is always available in the sidebar whenever you're ready. 💙")

            # Crisis resources for depression
            if prediction.lower() in ["depression", "suicidal"]:
                st.divider()
                st.subheader("📞 Crisis Resources")
                st.info("""
                **iCall:** 9152987821
                **AASRA:** 91-22-27546669
                **Vandrevala Foundation:** 1860-2662-345 *(24/7)*
                **Tele MANAS:** 14416 *(free · 24/7)*
                """)

    st.divider()
    st.caption("Built with Python · scikit-learn · Claude AI · Streamlit | For educational purposes only.")
