import streamlit as st
import plotly.graph_objects as go
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
import time
import streamlit.components.v1 as components
from nearby import geocode_city, get_nearby_places, build_map_html, build_location_detector_html
from face_scan_component import render_face_scan_tab

load_dotenv()

st.set_page_config(
    page_title="MindScan",
    page_icon="🧠",
    layout="centered"
)

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

.stApp {
    background: #07061a;
    font-family: 'DM Sans', sans-serif;
}
.stApp > div:first-child {
    background:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(124,58,237,0.22) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(56,189,248,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(79,70,229,0.08) 0%, transparent 70%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 900px !important;
}

@keyframes slideUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0);    }
}
@keyframes gradMove {
    0%   { background-position: 0% 50%;   }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%;   }
}
@keyframes bob {
    0%,100% { transform: translateY(0px);  }
    50%     { transform: translateY(-4px); }
}
@keyframes pulseRed {
    0%,100% { border-color: rgba(239,68,68,0.3); }
    50%     { border-color: rgba(239,68,68,0.7); }
}
@keyframes glowDot {
    0%,100% { opacity:1; }
    50%     { opacity:0.3; }
}

/* ── Topbar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    margin-bottom: 28px;
    animation: slideUp 0.4s ease-out;
    backdrop-filter: blur(20px);
}
.topbar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    white-space: nowrap;
}
.topbar-nav {
    display: flex;
    align-items: center;
    gap: 6px;
}
.nav-pill {
    padding: 7px 16px;
    border-radius: 40px;
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
    white-space: nowrap;
}
.nav-pill.active {
    background: rgba(124,58,237,0.22);
    border-color: rgba(124,58,237,0.5);
    color: #c4b5fd;
}
.nav-pill.inactive {
    background: transparent;
    border-color: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.45);
}
.nav-pill.inactive:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.75);
}
.nav-pill.mello-pill {
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(56,189,248,0.2));
    border-color: rgba(124,58,237,0.4);
    color: #c4b5fd;
    animation: bob 3s ease-in-out infinite;
}

/* ── Buttons ── */
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 40px !important;
    color: rgba(255,255,255,0.55) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.03em !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(124,58,237,0.12) !important;
    border-color: rgba(124,58,237,0.35) !important;
    color: #c4b5fd !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #0ea5e9) !important;
    background-size: 200% 200% !important;
    animation: gradMove 3s ease infinite !important;
    border: none !important;
    border-radius: 40px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.5) !important;
}
/* ── Textarea ── */
.stTextArea > label {
    color: rgba(255,255,255,0.3) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-family: 'Syne', sans-serif !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    color: rgba(255,255,255,0.88) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
    transition: border 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 4px rgba(124,58,237,0.1) !important;
}
.stTextArea textarea::placeholder {
    color: rgba(255,255,255,0.2) !important;
}

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 20px !important;
    animation: slideUp 0.5s ease-out !important;
    transition: all 0.2s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(124,58,237,0.3) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 14px !important;
    border-left-width: 3px !important;
    animation: slideUp 0.4s ease-out !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Chat ── */
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 18px !important;
    margin: 8px 0 !important;
    animation: slideUp 0.3s ease-out !important;
}
div[data-testid="stChatMessageContent"] p {
    color: rgba(255,255,255,0.85) !important;
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.7 !important;
}
div[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
}
div[data-testid="stChatInput"] textarea {
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Misc ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 16px 0 !important; }
.stSpinner > div { border-top-color: #7c3aed !important; }
.stCaption p { color: rgba(255,255,255,0.28) !important; font-size:0.75rem !important; }
p, li { color: rgba(255,255,255,0.7); font-family:'DM Sans',sans-serif; line-height:1.75; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; color:white !important; }

/* ── Custom components ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(100deg, #a78bfa 0%, #60a5fa 45%, #34d399 90%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradMove 4s ease infinite;
    line-height: 1.1;
    letter-spacing: -0.02em;
    text-align: center;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.35);
    font-size: 0.92rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-top: 8px;
}
.hero-bar {
    width: 48px; height: 3px;
    background: linear-gradient(90deg, #7c3aed, #38bdf8);
    border-radius: 2px;
    margin: 14px auto 0;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.28);
    margin-bottom: 10px;
    margin-top: 4px;
}
.suggest-box {
    background: linear-gradient(135deg,
        rgba(124,58,237,0.1) 0%,
        rgba(56,189,248,0.06) 100%);
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: 18px;
    padding: 22px 26px;
    animation: slideUp 0.5s ease-out;
}
.crisis-banner {
    background: linear-gradient(135deg,
        rgba(127,29,29,0.88) 0%,
        rgba(153,27,27,0.88) 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 20px;
    animation: slideUp 0.4s ease-out, pulseRed 2s infinite;
}
.crisis-row {
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 11px 16px;
    margin: 7px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: rgba(255,255,255,0.88);
}
.mello-header {
    display: flex;
    align-items: center;
    gap: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px 22px;
    margin-bottom: 18px;
    animation: slideUp 0.4s ease-out;
}
.m-avatar {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #7c3aed, #38bdf8);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    animation: bob 3s ease-in-out infinite;
}
.m-name { font-family:'Syne',sans-serif; font-weight:800; font-size:1.05rem; color:white; }
.m-status { font-size:0.76rem; color:rgba(255,255,255,0.38); margin-top:1px; }
.dot-live {
    display: inline-block;
    width: 6px; height: 6px;
    background: #34d399; border-radius: 50%;
    margin-right: 5px;
    animation: glowDot 1.5s ease-in-out infinite;
}
.pill-red   { display:inline-block; margin:3px; padding:5px 13px; border-radius:20px; font-size:12px; font-weight:500; background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.28); color:#fca5a5; }
.pill-green { display:inline-block; margin:3px; padding:5px 13px; border-radius:20px; font-size:12px; font-weight:500; background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.22); color:#6ee7b7; }
.badge-c { display:inline-block; background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.3); color:#fca5a5; padding:4px 14px; border-radius:20px; font-size:11px; font-family:'Syne',sans-serif; font-weight:700; }
.badge-w { display:inline-block; background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3); color:#fcd34d; padding:4px 14px; border-radius:20px; font-size:11px; font-family:'Syne',sans-serif; font-weight:700; }
.badge-g { display:inline-block; background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3); color:#6ee7b7; padding:4px 14px; border-radius:20px; font-size:11px; font-family:'Syne',sans-serif; font-weight:700; }
.tag { display:inline-block; background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.22); border-radius:6px; padding:3px 10px; font-size:0.7rem; color:#c4b5fd; font-family:'Syne',sans-serif; font-weight:700; letter-spacing:0.05em; margin:3px 2px; }
.info-card { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:14px 18px; margin-bottom:10px; }
.footer-txt { text-align:center; font-size:0.65rem; color:rgba(255,255,255,0.15); font-family:'Syne',sans-serif; letter-spacing:0.14em; text-transform:uppercase; padding:12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    p = "models/model.pkl"
    if not os.path.exists(p):
        st.error("Model not found. Run train.py first.")
        st.stop()
    return joblib.load(p)

model = load_model()

# ── Groq ───────────────────────────────────────────────────
@st.cache_resource
def load_groq():
    key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not key:
        return None
    return Groq(api_key=key)

groq_client = load_groq()

for k, v in {
    "page":                 "analyze",
    "messages":             [],
    "detected_mood":        None,
    "user_text_context":    "",
    "text_input":           "",
    "scores":               {},
    "confidence":           0.0,
    "total_scans":          0,
    "mood_history":         [],
    "user_lat":             None,
    "user_lng":             None,
    "nearby_places":        [],
    "selected_place_idx":   None,
    "nearby_error":         None,
    "scan_source":          "text",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── System prompt ──────────────────────────────────────────
def build_system_prompt(mood, ctx):
    cl = f"User's text ('{mood}'): \"{ctx[:200]}\"\n\n" if ctx else ""
    m  = {
        "normal":     "Warm and welcoming. Ask what is on their mind.",
        "depression": "Validate first. Never rush to fix. Gentle small steps.",
        "anxiety":    "Acknowledge exhaustion. 5-4-3-2-1 grounding. Ask main worry.",
        "stress":     "Empathize with overload. Help prioritize. Suggest one break.",
        "suicidal":   "CRISIS. Ask if safe. Every reply must end with iCall 9152987821.",
        "bipolar":    "Be a calm anchor. Ask how they feel right now. No assumptions.",
    }
    return f"""You are Mello 🫧 — a gentle warm mental health companion.
Speak like a caring friend. Never clinical. Never robotic.
Short warm replies. No bullet lists unless giving coping steps.
Never diagnose. Always remind professional help exists.
{cl}Mood: {mood} | Approach: {m.get(mood.lower(), m['normal'])}
Crisis numbers: iCall 9152987821 | AASRA 91-22-27546669 | Tele MANAS 14416"""


# ── Groq response ──────────────────────────────────────────
def mello_reply(messages, mood, ctx):
    if not groq_client:
        return "I'm having trouble connecting 💙\n\n📞 iCall: 9152987821"
    try:
        sys_p = build_system_prompt(mood, ctx)
        msgs  = [{"role": "system", "content": sys_p}]
        for m in messages:
            if m["role"] in ["user", "assistant"]:
                msgs.append({"role": m["role"], "content": m["content"]})
        r = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=msgs,
            max_tokens=512,
            temperature=0.75
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"Mello error: {e}")
        err = str(e).lower()
        if "429" in err or "quota" in err:
            return "I need a breath — try again in a moment 💙\n\n📞 iCall: 9152987821"
        return "Something went wrong 💙 Please try again."


# ── Open Mello ─────────────────────────────────────────────
def open_mello(mood, ctx, crisis=False):
    st.session_state.messages = []
    st.session_state.page     = "crisis" if crisis else "chat"
    if crisis:
        opening = (
            "Hey, I'm really glad you reached out. 💙\n\n"
            "You don't have to face this alone — I'm right here.\n\n"
            "First — are you safe right now?\n\n"
            "📞 **iCall:** 9152987821\n"
            "📞 **AASRA:** 91-22-27546669\n"
            "📞 **Vandrevala:** 1860-2662-345 *(24/7)*\n"
            "📞 **Tele MANAS:** 14416 *(free · 24/7)*\n\n"
            "I'm here. Take your time. 🫧"
        )
    else:
        seed = [{"role": "user", "content": (
            f"User text detected as '{mood}':\n\"{ctx[:300]}\"\n\n"
            f"Greet as Mello. Acknowledge ONE thing from their text. "
            f"Ask one gentle open question. Keep it short and warm."
            if ctx else
            "I just came to chat. Greet me warmly as Mello."
        )}]
        opening = mello_reply(seed, mood, ctx)
    st.session_state.messages.append({"role": "assistant", "content": opening})


# ══════════════════════════════════════════════════════════
# TOPBAR — always visible
# ══════════════════════════════════════════════════════════
def render_topbar():
    page        = st.session_state.page
    has_results = bool(st.session_state.detected_mood and st.session_state.scores)

    a_analyze = page == "analyze"
    a_results = page == "results"
    a_mello   = page in ["chat", "crisis"]
    a_nearby  = page == "nearby"

    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:12px 20px;
                background:rgba(255,255,255,0.03);
                border:1px solid rgba(255,255,255,0.08);
                border-radius:18px;margin-bottom:8px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                    background:linear-gradient(90deg,#a78bfa,#38bdf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;white-space:nowrap;">
            🧠 MindScan
        </div>
    </div>
    """, unsafe_allow_html=True)

    nc1, nc2, nc3, nc4 = st.columns(4)

    with nc1:
        if st.button(
            "🔍 Analyze Text",
            key="top_analyze",
            use_container_width=True,
            type="primary" if a_analyze else "secondary"
        ):
            st.session_state.page = "analyze"
            st.rerun()

    with nc2:
        btn_label = "📊 Last Results" if has_results else "📊 No results yet"
        if st.button(
            btn_label,
            key="top_results",
            use_container_width=True,
            type="primary" if a_results else "secondary",
            disabled=not has_results
        ):
            st.session_state.page = "results"
            st.rerun()

    with nc3:
        if st.button(
            "🫧 Mello",
            key="top_mello",
            use_container_width=True,
            type="primary" if a_mello else "secondary"
        ):
            if not a_mello:
                mood = st.session_state.detected_mood or "normal"
                open_mello(mood, st.session_state.user_text_context,
                           crisis=(mood == "suicidal"))
                st.rerun()

    with nc4:
        if st.button(
            "🏥 Nearby Help",
            key="top_nearby",
            use_container_width=True,
            type="primary" if a_nearby else "secondary"
        ):
            st.session_state.page = "nearby"
            st.rerun()

    st.write("")


render_topbar()


# ══════════════════════════════════════════════════════════
# PAGE: CRISIS
# ══════════════════════════════════════════════════════════
if st.session_state.page == "crisis":

    st.markdown("""
    <div class="crisis-banner">
        <div style="font-family:'Syne',sans-serif;font-size:1.15rem;
                    font-weight:800;color:white;margin-bottom:10px;">
            🚨 You are not alone
        </div>
        <div style="color:rgba(255,255,255,0.7);font-size:0.87rem;margin-bottom:14px;">
            Real humans are available 24/7 — please reach out right now.
        </div>
        <div class="crisis-row">📞 iCall &nbsp;&nbsp; <strong>9152987821</strong></div>
        <div class="crisis-row">📞 AASRA &nbsp;&nbsp; <strong>91-22-27546669</strong></div>
        <div class="crisis-row">📞 Vandrevala Foundation &nbsp;&nbsp; <strong>1860-2662-345</strong></div>
        <div class="crisis-row">📞 Tele MANAS &nbsp;&nbsp; <strong>14416</strong> (free · 24/7)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mello-header">
        <div class="m-avatar">🫧</div>
        <div>
            <div class="m-name">Mello</div>
            <div class="m-status">
                <span class="dot-live"></span>here with you right now
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("You are safe here. Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        recovery   = ["feel better","feeling better","i'm okay","i am okay",
                      "thank you","that helped","i feel safe","i called","much better"]
        recovering = any(s in prompt.lower() for s in recovery)
        with st.chat_message("assistant"):
            with st.spinner(""):
                reply = mello_reply(st.session_state.messages, "suicidal",
                                    st.session_state.user_text_context)
            if recovering:
                st.session_state.page = "chat"
                reply += "\n\n💙 I'm so glad you're feeling a little better. Still here."
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "chat":

    mood = st.session_state.detected_mood or "normal"

    st.markdown(f"""
    <div class="mello-header">
        <div class="m-avatar">🫧</div>
        <div>
            <div class="m-name">Mello</div>
            <div class="m-status">
                <span class="dot-live"></span>
                talking with you · {mood.title()} context loaded
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        recovery  = ["feel better","feeling better","i'm okay","i am okay",
                     "thank you","that helped","much better","i feel good"]
        crisis_kw = ["want to die","end my life","kill myself",
                     "no point living","better off dead","suicide"]
        recovering = any(s in prompt.lower() for s in recovery)
        in_crisis  = any(s in prompt.lower() for s in crisis_kw)
        if in_crisis:
            st.session_state.detected_mood = "suicidal"
            open_mello("suicidal", st.session_state.user_text_context, crisis=True)
            st.rerun()
        with st.chat_message("assistant"):
            with st.spinner(""):
                reply = mello_reply(
                    st.session_state.messages,
                    "normal" if recovering else mood,
                    st.session_state.user_text_context
                )
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        if recovering:
            st.session_state.detected_mood = "normal"
        st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    prediction = st.session_state.detected_mood
    scores     = st.session_state.scores
    confidence = st.session_state.confidence

    st.markdown("""
    <div style="animation:slideUp 0.5s ease-out;text-align:center;padding:16px 0 8px">
        <div class="hero-title">Your Results</div>
        <div class="hero-sub">Here is what MindScan detected in your text</div>
        <div class="hero-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected",   prediction.replace("_"," ").title())
    c2.metric("Confidence", f"{confidence*100:.1f}%")
    c3.metric("Words",      len(st.session_state.user_text_context.split()))

    if confidence < 0.55:
        st.warning("⚠️ Low confidence — mixed signals detected. Interpret with caution.")

    st.write("")
    ss  = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    fig = go.Figure(go.Bar(
        x=[k.replace("_"," ").title() for k in ss],
        y=[v*100 for v in ss.values()],
        marker=dict(
            color=["#7c3aed" if k==prediction
                   else "rgba(255,255,255,0.08)" for k in ss],
            line=dict(width=0)
        ),
        text=[f"{v*100:.1f}%" for v in ss.values()],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.5)", size=11)
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.5)", family="DM Sans", size=11),
        yaxis=dict(range=[0,120], showgrid=True,
                   gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(color="rgba(255,255,255,0.25)"), title=""),
        xaxis=dict(tickfont=dict(color="rgba(255,255,255,0.45)")),
        margin=dict(t=16, b=8, l=0, r=0),
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)

    try:
        vec  = model.named_steps["tfidf"]
        clf  = model.named_steps["clf"]
        ci   = list(clf.classes_).index(prediction)
        coef = clf.coef_[ci]
        tv   = vec.transform([st.session_state.user_text_context])
        fn   = vec.get_feature_names_out()
        nz   = tv.nonzero()[1]
        ws   = sorted([(fn[i], float(coef[i]*tv[0,i])) for i in nz],
                      key=lambda x: abs(x[1]), reverse=True)[:8]
        if ws:
            st.markdown(
                '<div class="section-label" style="margin-top:16px">Key signals detected</div>',
                unsafe_allow_html=True
            )
            pills = "".join(
                f'<span class="pill-red">{w}</span>'
                if s > 0 else f'<span class="pill-green">{w}</span>'
                for w, s in ws
            )
            st.markdown(f'<div style="animation:slideUp 0.5s ease-out">{pills}</div>',
                        unsafe_allow_html=True)
            st.caption("Red → drove this prediction · Green → worked against it")
    except Exception:
        pass

    st.write("")
    interp = {
        "normal":               ("✅ No significant distress signals detected.",          "success"),
        "depression":           ("🔴 Signals associated with depression detected.",      "error"),
        "anxiety":              ("🟠 Signals associated with anxiety detected.",         "warning"),
        "suicidal":             ("🚨 High risk signals detected. Please seek help now.", "error"),
        "stress":               ("🟡 Stress signals detected.",                          "warning"),
        "bipolar":              ("🟠 Bipolar mood signals detected.",                    "warning"),
        "personality disorder": ("🔴 Personality disorder signals detected.",           "error"),
    }
    msg, mt = interp.get(prediction.lower(),
                         (f"Detected: {prediction.title()}", "info"))
    if mt == "success":   st.success(msg)
    elif mt == "error":   st.error(msg)
    elif mt == "warning": st.warning(msg)
    else:                 st.info(msg)

    st.write("")
    st.markdown("""
    <div class="suggest-box">
        <div style="font-family:'Syne',sans-serif;font-weight:700;
                    font-size:1rem;color:white;">
            🫧 Want to talk to Mello?
        </div>
        <div style="color:rgba(255,255,255,0.5);font-size:0.87rem;margin-top:6px;">
            Mello has already read your text and is ready to listen.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([2,1])
    with c1:
        if st.button("🫧 Talk to Mello", type="primary", use_container_width=True):
            open_mello(prediction, st.session_state.user_text_context, crisis=False)
            st.rerun()
    with c2:
        if st.button("🔍 New Scan", use_container_width=True):
            st.session_state.page = "analyze"
            st.rerun()

    if prediction.lower() in ["depression","bipolar","suicidal"]:
        st.divider()
        st.markdown('<div class="section-label">Crisis resources</div>',
                    unsafe_allow_html=True)
        st.info("**iCall:** 9152987821 · **AASRA:** 91-22-27546669 · "
                "**Tele MANAS:** 14416 *(free 24/7)*")


# ══════════════════════════════════════════════════════════
# PAGE: NEARBY HELP
# ══════════════════════════════════════════════════════════
elif st.session_state.page == "nearby":

    DEFAULT_LAT, DEFAULT_LNG = 28.6139, 77.2090   # New Delhi

    # ── Guarantee the map ALWAYS has a location ─────────────
    if st.session_state.user_lat is None:
        st.session_state.user_lat = DEFAULT_LAT
        st.session_state.user_lng = DEFAULT_LNG

    if "_nearby_loc_key" not in st.session_state:
        st.session_state._nearby_loc_key = None

    # ── Hero ────────────────────────────────────────────────
    st.markdown("""
    <div style="animation:slideUp 0.5s ease-out;
                text-align:center;padding:16px 0 8px">
        <div class="hero-title">Nearby Help</div>
        <div class="hero-sub">
            Find hospitals, clinics and doctors near you
        </div>
        <div class="hero-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ── AI Recommendation banner ────────────────────────────
    mood = st.session_state.detected_mood
    if mood and mood.lower() != "normal":

        mood_advice = {
            "depression": (
                "🔵 Based on your scan, we recommend consulting a **psychiatrist or psychologist**. "
                "Look for hospitals with a dedicated **mental health or psychiatry department**."
            ),
            "anxiety": (
                "🟠 Your scan suggests anxiety signals. A **clinical psychologist or counselor** "
                "can help. Look for clinics offering **therapy or CBT sessions**."
            ),
            "suicidal": (
                "🚨 Please reach out immediately. Call **iCall: 9152987821** or "
                "**Tele MANAS: 14416** right now. Search for the nearest "
                "**government hospital with a psychiatric emergency unit**."
            ),
            "stress": (
                "🟡 Stress can be managed with professional support. Look for "
                "**wellness clinics, counselors, or general physicians** nearby."
            ),
            "bipolar": (
                "🟣 Bipolar disorder needs specialist care. Search for a "
                "**psychiatrist or a hospital with a neurology/psychiatry department**."
            ),
        }

        advice = mood_advice.get(
            mood.lower(),
            "We recommend consulting a mental health professional nearby."
        )

        color = "#ef4444" if mood.lower() == "suicidal" else "#7c3aed"

        st.markdown(f"""
        <div style="background:rgba(124,58,237,0.08);
                    border-left:4px solid {color};
                    border-radius:0 14px 14px 0;
                    padding:16px 20px;margin-bottom:8px;
                    animation:slideUp 0.5s ease-out;">
            <div style="font-family:'Syne',sans-serif;font-weight:700;
                        font-size:0.82rem;color:white;margin-bottom:6px;">
                🫧 Mello's Recommendation — based on your {mood.title()} scan
            </div>
            <div style="font-size:0.85rem;color:rgba(255,255,255,0.7);
                        line-height:1.7;">
                {advice}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if mood.lower() in ["suicidal", "depression"]:
            st.markdown("""
            <div style="background:rgba(239,68,68,0.08);
                        border:1px solid rgba(239,68,68,0.25);
                        border-radius:14px;padding:16px 20px;
                        margin-bottom:12px;">
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            font-size:0.85rem;color:#fca5a5;margin-bottom:10px;">
                    📞 Immediate Crisis Support
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                    <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                                padding:10px 14px;font-size:0.8rem;
                                color:rgba(255,255,255,0.75);">
                        <strong style="color:white">iCall</strong><br>9152987821
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                                padding:10px 14px;font-size:0.8rem;
                                color:rgba(255,255,255,0.75);">
                        <strong style="color:white">AASRA</strong><br>91-22-27546669
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                                padding:10px 14px;font-size:0.8rem;
                                color:rgba(255,255,255,0.75);">
                        <strong style="color:white">Tele MANAS</strong><br>14416 (free · 24/7)
                    </div>
                    <div style="background:rgba(255,255,255,0.05);border-radius:10px;
                                padding:10px 14px;font-size:0.8rem;
                                color:rgba(255,255,255,0.75);">
                        <strong style="color:white">Vandrevala</strong><br>1860-2662-345
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Location controls (collapsed by default so map is above fold) ──
    with st.expander("📍 Change Location", expanded=False):
        tab1, tab2 = st.tabs(["🌐 Auto-detect", "✏️ Manual input"])

        with tab1:
            st.markdown("""
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.45);
                        margin-bottom:12px;font-family:'DM Sans',sans-serif;">
                Click the button below and allow location access
                when your browser asks.
            </div>
            """, unsafe_allow_html=True)

            components.html(build_location_detector_html(), height=110)

            st.markdown("""
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.3);
                        margin-top:8px;font-family:'DM Sans',sans-serif;">
                If auto-detect doesn't work, paste your coordinates
                in the Manual Input tab.
            </div>
            """, unsafe_allow_html=True)

            pasted = st.text_input(
                label="Paste detected coordinates here",
                placeholder="e.g. 28.635308,77.224960",
                key="paste_coords",
                help="After clicking Allow Location Access, copy the coordinates and paste here",
            )
            if st.button("✅ Use these coordinates", key="use_pasted",
                         use_container_width=True):
                try:
                    parts = pasted.strip().split(",")
                    p_lat = float(parts[0].strip())
                    p_lng = float(parts[1].strip())
                    st.session_state.user_lat = p_lat
                    st.session_state.user_lng = p_lng
                    st.session_state.nearby_places = []
                    st.session_state.selected_place_idx = None
                    st.session_state.nearby_error = None
                    st.success("✅ Location set!")
                    st.rerun()
                except Exception:
                    st.error("Invalid format. Use: latitude,longitude — e.g. 28.6139,77.2090")

        with tab2:
            city_col, btn_col = st.columns([3, 1])
            with city_col:
                city_input = st.text_input(
                    label="city",
                    label_visibility="collapsed",
                    placeholder="Type city or area — e.g. Connaught Place Delhi",
                    key="city_search",
                )
            with btn_col:
                find_btn = st.button("🔍 Find", use_container_width=True,
                                     type="primary", key="find_city_btn")

            if find_btn and city_input.strip():
                with st.spinner(f"Finding {city_input}..."):
                    found_lat, found_lng, display = geocode_city(city_input)
                if found_lat:
                    st.session_state.user_lat = found_lat
                    st.session_state.user_lng = found_lng
                    st.session_state.nearby_places = []
                    st.session_state.selected_place_idx = None
                    st.session_state.nearby_error = None
                    st.success(f"✅ {display[:65]}...")
                    st.rerun()
                else:
                    st.error("Not found. Try a different name.")

            st.write("")
            mc1, mc2 = st.columns(2)
            with mc1:
                manual_lat = st.number_input(
                    "Latitude",
                    value=float(st.session_state.user_lat),
                    format="%.4f", key="manual_lat",
                )
            with mc2:
                manual_lng = st.number_input(
                    "Longitude",
                    value=float(st.session_state.user_lng),
                    format="%.4f", key="manual_lng",
                )
            if st.button("Use these coordinates",
                         use_container_width=True, key="use_manual"):
                st.session_state.user_lat = manual_lat
                st.session_state.user_lng = manual_lng
                st.session_state.nearby_places = []
                st.session_state.selected_place_idx = None
                st.session_state.nearby_error = None
                st.rerun()

            st.caption("Find your coordinates: open maps.google.com → right click your location → copy coordinates.")

    # ── Current location pill ───────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(124,58,237,0.08);
                border:1px solid rgba(124,58,237,0.2);
                border-radius:10px;padding:10px 16px;margin-top:8px;
                font-size:0.8rem;color:rgba(255,255,255,0.6);
                font-family:'DM Sans',sans-serif;">
        📍 Location set to &nbsp;
        <strong style="color:#c4b5fd">
            {st.session_state.user_lat:.4f}, {st.session_state.user_lng:.4f}
        </strong>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ── Search controls ─────────────────────────────────────
    st.markdown('<div class="section-label">What to find</div>',
                unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        btn_hosp = st.button("🏥 Hospitals", use_container_width=True)
    with sc2:
        btn_doc  = st.button("👨‍⚕️ Clinics & Doctors", use_container_width=True)
    with sc3:
        btn_both = st.button("🔍 Search All", use_container_width=True,
                             type="primary")

    radius_km = st.slider("Search radius", 1, 20, 5, format="%d km")

    # ── Decide whether to search ────────────────────────────
    search_type = None
    if btn_hosp: search_type = "hospital"
    if btn_doc:  search_type = "doctor"
    if btn_both: search_type = "both"

    # Auto-search whenever the location key changes (first visit + on coord change)
    loc_key = f"{st.session_state.user_lat:.4f},{st.session_state.user_lng:.4f}"
    auto_search = (
        search_type is None
        and st.session_state._nearby_loc_key != loc_key
    )
    if auto_search:
        search_type = "both"
        st.session_state._nearby_loc_key = loc_key

    # ── Run search — never raises, never hides the map ──────
    if search_type and st.session_state.user_lat is not None:
        with st.spinner(f"Searching within {radius_km} km..."):
            try:
                found = get_nearby_places(
                    st.session_state.user_lat,
                    st.session_state.user_lng,
                    search_type,
                    radius_km * 1000,
                )
                st.session_state.nearby_places = found
                st.session_state.nearby_error = None
                if not auto_search:
                    st.session_state.selected_place_idx = None
            except Exception as e:
                st.session_state.nearby_places = []
                st.session_state.nearby_error = str(e)

    # ── Map — ALWAYS rendered from here down ────────────────
    lat = st.session_state.user_lat
    lng = st.session_state.user_lng
    places  = st.session_state.nearby_places or []
    sel_idx = st.session_state.get("selected_place_idx", None)

    st.write("")
    st.markdown('<div class="section-label">🗺️ Map</div>',
                unsafe_allow_html=True)

    # Status banner above the map (informational, not fatal)
    if st.session_state.nearby_error:
        st.warning(
            f"⚠️ Couldn't reach OpenStreetMap servers "
            f"({st.session_state.nearby_error}). "
            f"The map below still shows your location."
        )
    elif places:
        st.success(f"✅ {len(places)} place(s) found within {radius_km} km")
    else:
        st.info("📍 Showing your location. Press a search button above to find help nearby.")

    # Render map — wrapped so any failure still leaves the page usable
    try:
        map_html = build_map_html(lat, lng, places, sel_idx)
        components.html(map_html, height=440)
    except Exception as e:
        st.error(f"Map could not render, but the list below is still usable. ({e})")

    # ── Result cards ────────────────────────────────────────
    if places:
        st.write("")
        st.markdown(
            '<div class="section-label">Tap a card to highlight on map</div>',
            unsafe_allow_html=True,
        )

        for i in range(0, len(places), 2):
            row = st.columns(2)
            for j, col in enumerate(row):
                idx = i + j
                if idx >= len(places):
                    break
                p      = places[idx]
                is_sel = (idx == sel_idx)
                bg     = "rgba(124,58,237,0.15)" if is_sel else "rgba(255,255,255,0.04)"
                border = "rgba(124,58,237,0.5)"  if is_sel else "rgba(255,255,255,0.07)"

                with col:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {border};
                                border-radius:14px;padding:14px 16px;
                                margin-bottom:4px;min-height:110px;">
                        <div style="font-family:'Syne',sans-serif;font-weight:700;
                                    font-size:0.85rem;color:white;margin-bottom:4px;">
                            {p['name']}
                        </div>
                        <div style="font-size:0.72rem;
                                    color:rgba(255,255,255,0.4);line-height:1.7;">
                            🏷️ {p['type']}<br>
                            📍 {p['dist_km']} km away<br>
                            {('📞 ' + p['phone']) if p['phone'] else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("📍 Show on map",
                                 key=f"sel_{idx}",
                                 use_container_width=True):
                        st.session_state.selected_place_idx = idx
                        st.rerun()

        # Directions for selected place
        if sel_idx is not None and sel_idx < len(places):
            sel = places[sel_idx]
            st.write("")
            st.markdown('<div class="section-label">Get directions</div>',
                        unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            gmap = (f"https://www.google.com/maps/dir/?api=1"
                    f"&origin={lat},{lng}"
                    f"&destination={sel['lat']},{sel['lng']}")
            osm  = (f"https://www.openstreetmap.org/directions?"
                    f"from={lat},{lng}&to={sel['lat']},{sel['lng']}")
            with d1:
                st.markdown(f"""
                <a href="{gmap}" target="_blank"
                   style="display:block;
                          background:linear-gradient(135deg,#7c3aed,#4f46e5);
                          color:white;text-decoration:none;border-radius:12px;
                          padding:13px;text-align:center;
                          font-family:'Syne',sans-serif;
                          font-weight:700;font-size:0.85rem;">
                    🗺️ Google Maps
                </a>
                """, unsafe_allow_html=True)
            with d2:
                st.markdown(f"""
                <a href="{osm}" target="_blank"
                   style="display:block;
                          background:rgba(255,255,255,0.06);
                          border:1px solid rgba(255,255,255,0.12);
                          color:rgba(255,255,255,0.8);text-decoration:none;
                          border-radius:12px;padding:13px;text-align:center;
                          font-family:'Syne',sans-serif;
                          font-weight:600;font-size:0.85rem;">
                    🌍 OpenStreetMap
                </a>
                """, unsafe_allow_html=True)

    st.write("")
    st.divider()
    st.error(
        "🚨 Emergency: **112** · "
        "iCall: **9152987821** · "
        "Tele MANAS: **14416** *(free 24/7)*"
    )


# ══════════════════════════════════════════════════════════
# PAGE: ANALYZE
# ══════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div style="animation:slideUp 0.6s ease-out;text-align:center;padding:20px 0 8px">
        <div class="hero-title">MindScan</div>
        <div class="hero-sub">
            Detect emotional distress signals in text · NLP + AI
        </div>
        <div class="hero-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.warning("A Tool for detecting mental health of a person by analyzing their text.")
    st.write("")

    tab_text, tab_face = st.tabs(["✍️  Analyze Text", "📷  Live Face Scan"])

    with tab_text:
        st.markdown('<div class="section-label">Or write your own</div>',
                    unsafe_allow_html=True)
        text_input = st.text_area(
            label="",
            height=160,
            value=st.session_state.get("text_input", ""),
            placeholder="How have you been feeling lately? Write freely — this is a safe space...",
        )
        st.write("")
        if st.button("✦ Analyze Text", type="primary", use_container_width=True):
            if not text_input.strip():
                st.error("Please enter some text first.")
                st.stop()
            with st.spinner("Scanning for signals..."):
                crisis_keywords = ["suicide","suicidal","kill myself","end my life","want to die"]

                if any(keyword in text_input.lower() for keyword in crisis_keywords):
                    pred = "suicidal"
                    proba = model.predict_proba([text_input])[0]
                else:
                    pred = model.predict([text_input])[0]
                    proba = model.predict_proba([text_input])[0]
                lbls  = model.classes_
                sc    = {l: round(float(p),4) for l,p in zip(lbls,proba)}
                conf  = round(float(max(proba)),4)
            st.session_state.detected_mood     = pred
            st.session_state.user_text_context = text_input
            st.session_state.scores            = sc
            st.session_state.confidence        = conf
            st.session_state.scan_source       = "text"
            st.session_state.total_scans      += 1
            st.session_state.mood_history.append(pred)
            if pred.lower() == "suicidal":
                with st.spinner("Connecting you to Mello..."):
                    time.sleep(1.2)
                open_mello("suicidal", text_input, crisis=True)
            else:
                st.session_state.page = "results"
            st.rerun()

    with tab_face:
        render_face_scan_tab()
    st.write("")

    if st.session_state.total_scans > 0:
        st.divider()
        st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Scans done", st.session_state.total_scans)
        if st.session_state.detected_mood:
            s2.metric("Last result", st.session_state.detected_mood.title())
        s3.metric("Confidence", f"{st.session_state.confidence*100:.0f}%"
                  if st.session_state.confidence else "—")

    st.markdown(
        '<div class="footer-txt" style="margin-top:28px">'
        'Built with Python · Scikit-learn · Groq · Streamlit'
        '</div>',
        unsafe_allow_html=True
    )
