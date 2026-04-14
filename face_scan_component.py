# face_scan_component.py
import streamlit as st
import av
import threading
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from face_emotion import get_fer_detector, average_emotion_frames, map_to_mental_health, describe_face_scan

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

class EmotionVideoProcessor:
    def __init__(self):
        self.detector    = get_fer_detector()
        self.frame_data  = []          # list of emotion dicts
        self.lock        = threading.Lock()
        self.scanning    = False
        self.last_result = None        # last per-frame emotion dict

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = self.detector.detect_emotions(img)

        if result:
            emotions = result[0]["emotions"]
            self.last_result = emotions
            if self.scanning:
                with self.lock:
                    self.frame_data.append(emotions)

            # Draw bounding box + dominant emotion
            import cv2
            x, y, w, h = result[0]["box"]
            dominant   = max(emotions, key=emotions.get)
            score      = emotions[dominant]
            color_map  = {
                "happy":   (52,  211, 153),
                "sad":     (96,  165, 250),
                "angry":   (239, 68,  68 ),
                "fear":    (251, 191, 36 ),
                "disgust": (167, 139, 250),
                "neutral": (156, 163, 175),
                "surprise":(251, 146, 60 ),
            }
            color = color_map.get(dominant, (255, 255, 255))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{dominant.upper()}  {score:.0%}"
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def start_scan(self):
        with self.lock:
            self.frame_data = []
            self.scanning   = True

    def stop_scan(self) -> list:
        with self.lock:
            self.scanning = False
            return list(self.frame_data)


def render_face_scan_tab():
    """
    Call this inside your Analyze page tab.
    Sets st.session_state keys:
        detected_mood, scores, confidence,
        user_text_context, scan_source
    on completion.
    """

    st.markdown("""
    <div style="background:rgba(124,58,237,0.07);border:1px solid rgba(124,58,237,0.18);
                border-radius:16px;padding:16px 20px;margin-bottom:16px;
                font-family:'DM Sans',sans-serif;font-size:0.85rem;
                color:rgba(255,255,255,0.55);line-height:1.8;">
        🎥 <strong style="color:rgba(255,255,255,0.8)">Live Face Scan</strong> —
        your camera detects facial emotions in real time.<br>
        Click <strong style="color:#c4b5fd">Start Scan</strong>, hold for 5 seconds,
        then click <strong style="color:#c4b5fd">Analyze</strong>.
        No video is stored or sent anywhere.
    </div>
    """, unsafe_allow_html=True)

    # Init session keys
    for k, v in {
        "face_scanning":   False,
        "face_frames":     [],
        "face_scan_done":  False,
        "face_emotions":   {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # WebRTC streamer
    ctx = webrtc_streamer(
        key              = "face_emotion",
        mode             = WebRtcMode.SENDRECV,
        rtc_configuration= RTC_CONFIG,
        video_processor_factory = EmotionVideoProcessor,
        media_stream_constraints = {"video": True, "audio": False},
        async_processing = True,
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔴 Start 5-sec Scan", type="primary",
                     use_container_width=True, key="face_start"):
            if ctx.video_processor:
                ctx.video_processor.start_scan()
                st.session_state.face_scanning  = True
                st.session_state.face_scan_done = False
                st.session_state.face_frames    = []
                st.info("📸 Scanning… stay still for 5 seconds.")
                time.sleep(5)
                frames = ctx.video_processor.stop_scan()
                st.session_state.face_frames    = frames
                st.session_state.face_scanning  = False
                st.session_state.face_scan_done = True
                st.rerun()
            else:
                st.warning("Camera not ready. Allow camera access first.")

    with col2:
        analyze_disabled = not st.session_state.face_scan_done
        if st.button("✦ Analyze Face Scan", type="primary",
                     use_container_width=True, key="face_analyze",
                     disabled=analyze_disabled):
            frames = st.session_state.face_frames
            if not frames:
                st.error("No face detected during scan. Try better lighting.")
                st.stop()

            avg_emotions = average_emotion_frames(frames)
            prediction, scores, confidence = map_to_mental_health(avg_emotions)
            description  = describe_face_scan(avg_emotions, prediction)

            # Write to session state — same keys as text scan
            st.session_state.detected_mood      = prediction
            st.session_state.scores             = scores
            st.session_state.confidence         = confidence
            st.session_state.user_text_context  = description
            st.session_state.face_emotions      = avg_emotions
            st.session_state.scan_source        = "face"   # ← NEW flag
            st.session_state.total_scans       += 1
            st.session_state.mood_history.append(prediction)

            if prediction == "suicidal":
                from streamlit_app import open_mello
                open_mello("suicidal", description, crisis=True)
            else:
                st.session_state.page = "results"
            st.rerun()

    # Live emotion bar while scanning
    if ctx.video_processor and ctx.video_processor.last_result:
        emotions = ctx.video_processor.last_result
        st.write("")
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:0.65rem;'
                    'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                    'color:rgba(255,255,255,0.28);margin-bottom:8px;">Live feed</div>',
                    unsafe_allow_html=True)
        for emotion, score in sorted(emotions.items(), key=lambda x: -x[1]):
            if score > 0.02:
                st.progress(min(score, 1.0),
                            text=f"{emotion.capitalize()}  {score:.0%}")

    if st.session_state.face_scan_done and st.session_state.face_frames:
        avg = average_emotion_frames(st.session_state.face_frames)
        st.write("")
        st.success(f"✅ Scan captured {len(st.session_state.face_frames)} frames. "
                   f"Click **Analyze Face Scan** above.")
        st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:0.65rem;'
                    'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
                    'color:rgba(255,255,255,0.28);margin-bottom:8px;margin-top:12px;">'
                    'Averaged scan results</div>', unsafe_allow_html=True)
        for emotion, score in sorted(avg.items(), key=lambda x: -x[1]):
            if score > 0.02:
                st.progress(min(score, 1.0),
                            text=f"{emotion.capitalize()}  {score:.0%}")