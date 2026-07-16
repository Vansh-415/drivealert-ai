"""
====================================================================
DriveAlert AI - Driver Drowsiness Detection Web App
====================================================================
A CNN-based system that detects driver drowsiness from eye state
(open/closed), using a robust DNN face detector to locate the eye
region, then a trained neural network to classify it.

ONE page, THREE ways to test:
  1. Live Camera (Continuous) -- real-time, no clicking needed
  2. Take Photo -- reliable single-snapshot camera test
  3. Upload Image -- upload any photo

HOW TO RUN LOCALLY:
    streamlit run app.py
====================================================================
"""

import json
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

IMG_SIZE = 96
CONF_THRESHOLD = 0.6

st.set_page_config(page_title="DriveAlert AI - Drowsiness Detection", page_icon="🚗", layout="centered")

# ---------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
    .main { background-color: #0e1117; }
    .hero-wrap { text-align: center; padding: 1.2rem 0 0.6rem 0; }
    .hero-icon { font-size: 2.6rem; }
    .title-text {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #6C5CE7, #00cec9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0.3rem 0 0.1rem 0;
    }
    .subtitle-text { color: #9aa0a6; font-size: 0.92rem; margin-bottom: 1.6rem; font-weight: 400; }
    .divider {
        height: 3px; width: 90px; margin: 0.6rem auto 1.6rem auto; border-radius: 10px;
        background: linear-gradient(90deg, #6C5CE7, #00cec9);
    }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; gap: 12px; }
    .stTabs [data-baseweb="tab"] { font-size: 0.98rem; font-weight: 600; padding: 8px 18px; border-radius: 10px 10px 0 0; }
    .result-card {
        border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: 0.8rem;
        font-size: 1.15rem; font-weight: 700; text-align: center; letter-spacing: 0.3px;
    }
    .card-drowsy { background: rgba(255, 71, 87, 0.12); border: 1.5px solid rgba(255, 71, 87, 0.55); color: #ff6b6b; }
    .card-alert { background: rgba(46, 213, 115, 0.10); border: 1.5px solid rgba(46, 213, 115, 0.5); color: #55efc4; }
    .card-warn { background: rgba(255, 190, 60, 0.10); border: 1.5px solid rgba(255, 190, 60, 0.5); color: #ffc048; font-weight: 500; font-size: 0.98rem; }
    footer, #MainMenu { visibility: hidden; }
    .footer-note { text-align: center; color: #666c74; font-size: 0.78rem; margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">🚗💤</div>
    <div class="title-text">DriveAlert AI</div>
    <div class="subtitle-text">Real-time driver drowsiness detection powered by a neural network</div>
    <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# LOAD MODEL + LABELS + ROBUST DNN FACE DETECTOR
# ---------------------------------------------------------------
@st.cache_resource
def load_everything():
    model = load_model('drowsiness_model.h5')
    with open('class_indices.json') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    face_net = cv2.dnn.readNetFromCaffe(
        'face_detector/deploy.prototxt',
        'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    )
    return model, idx_to_class, face_net

model, idx_to_class, face_net = load_everything()


def detect_and_predict(img_rgb):
    """Returns (label_text, region_used_for_prediction, face_box, eye_box)."""
    h, w = img_rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_rgb, (300, 300)), 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf, best_box = 0, None
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > CONF_THRESHOLD and confidence > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)
            best_conf = confidence

    if best_box is None:
        return "No face detected", img_rgb, None, None

    x1, y1, x2, y2 = best_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_w, face_h = x2 - x1, y2 - y1

    if face_w <= 0 or face_h <= 0:
        return "No face detected", img_rgb, None, None

    # Eyes reliably sit in the upper-middle band of the face box
    eye_y1 = y1 + int(face_h * 0.18)
    eye_y2 = y1 + int(face_h * 0.55)
    eye_region = img_rgb[eye_y1:eye_y2, x1:x2]

    if eye_region.size == 0:
        eye_region = img_rgb[y1:y2, x1:x2]
        eye_y1, eye_y2 = y1, y2

    eye_resized = cv2.resize(eye_region, (IMG_SIZE, IMG_SIZE))
    eye_array = np.expand_dims(eye_resized / 255.0, axis=0)

    prediction = model.predict(eye_array, verbose=0)[0][0]
    idx = 1 if prediction > 0.5 else 0
    label = idx_to_class[idx]
    result = "DROWSY!" if label == "Drowsy" else "ALERT"

    face_box_out = (x1, y1, face_w, face_h)
    eye_box_out = (x1, eye_y1, face_w, eye_y2 - eye_y1)
    return result, eye_region, face_box_out, eye_box_out


def show_result(img_to_display, label_text, eye_crop):
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_to_display, caption="Input", use_container_width=True)
    with col2:
        st.image(eye_crop, caption="Region analyzed", use_container_width=True)

    if label_text == "DROWSY!":
        st.markdown('<div class="result-card card-drowsy">🚨 ALERT — DROWSINESS DETECTED!</div>', unsafe_allow_html=True)
    elif label_text == "ALERT":
        st.markdown('<div class="result-card card-alert">✅ AWAKE / NORMAL</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="result-card card-warn">No face detected — try a clearer, '
            'front-facing, well-lit photo, closer to the camera.</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------
# CONTINUOUS live tab (only defined if streamlit-webrtc/av are available)
# ---------------------------------------------------------------
if WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class DrowsinessProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.last_label = "Detecting..."
            self.last_color = (255, 255, 0)
            self.last_face_box = None
            self.last_eye_box = None

        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % 3 == 0:   # run detection every 3rd frame for smooth video
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                label_text, _, face_box, eye_box = detect_and_predict(img_rgb)
                self.last_label = label_text
                self.last_color = (0, 0, 255) if label_text == "DROWSY!" else \
                                   (0, 200, 0) if label_text == "ALERT" else (255, 255, 0)
                self.last_face_box = face_box
                self.last_eye_box = eye_box

            if self.last_face_box is not None:
                (fx, fy, fw, fh) = self.last_face_box
                cv2.rectangle(img_bgr, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
            if self.last_eye_box is not None:
                (bx, by, bw, bh) = self.last_eye_box
                cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), self.last_color, 2)

            cv2.putText(img_bgr, self.last_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.last_color, 3)
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# ---------------------------------------------------------------
# PAGE LAYOUT (Live Camera tab only shown if webrtc/av loaded successfully)
# ---------------------------------------------------------------
if WEBRTC_AVAILABLE:
    tab1, tab2, tab3 = st.tabs(["🔴 Live Camera (Continuous)", "📷 Take Photo", "📁 Upload Image"])
else:
    tab2, tab3 = st.tabs(["📷 Take Photo", "📁 Upload Image"])
    tab1 = None
    st.info("Live continuous camera mode isn't available in this environment (the 'av'/webrtc "
            "package didn't install), but 'Take Photo' and 'Upload Image' work fully and reliably below.")

if tab1 is not None:
    with tab1:
        st.write("Click **Start**, allow camera access, and wait ~2-3 seconds. "
                 "Shows a live DROWSY / ALERT label continuously — no clicking needed after Start.")
        webrtc_streamer(
            key="drowsiness-live",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )
        st.caption("If the video stays blank: your firewall may be blocking the connection. "
                   "Allow Python through your firewall (Private + Public), then restart the app. "
                   "You can always use the 'Take Photo' tab as a reliable alternative.")

with tab2:
    st.write("Click below, allow camera access, and take a photo. Take another any time to re-test.")
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        img = Image.open(camera_photo).convert('RGB')
        label_text, eye_crop, _, _ = detect_and_predict(np.array(img))
        show_result(img, label_text, eye_crop)
    else:
        st.info("Click 'Take Photo' above to test the model.")

with tab3:
    uploaded_file = st.file_uploader("Upload a driver face image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        label_text, eye_crop, _, _ = detect_and_predict(np.array(img))
        show_result(img, label_text, eye_crop)
    else:
        st.info("Upload an image above to test the model.")


st.markdown(
    '<div class="footer-note">DriveAlert AI &middot; CNN-based Drowsiness Detection</div>',
    unsafe_allow_html=True
)
