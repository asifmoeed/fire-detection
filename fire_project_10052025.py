import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import time
import threading

# Streamlit page config
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Real-Time Fire Detection")
st.markdown("""
    This system detects fire, smoke, and flames using YOLOv8.
    Adjust the confidence threshold in the sidebar to control sensitivity.
""")

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()
FIRE_CLASSES = [25, 26, 27]  # Update as per your trained model

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    camera_option = st.radio("Camera Source:", ("Webcam", "IP Camera"))
    ip_cam_url = ""
    if camera_option == "IP Camera":
        ip_cam_url = st.text_input("Enter Camera URL:", "http://192.168.X.X:8080/video")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.01)
    st.subheader("Alert Options")
    enable_alerts = st.checkbox("Enable Alerts", True)
    alert_cooldown = st.number_input("Alert Cooldown (seconds)", 60, 300, 60)
    stop_button = st.button("🛑 Stop Detection")

# Set session state flags
if "stop_detection" not in st.session_state:
    st.session_state["stop_detection"] = False

if stop_button:
    st.session_state["stop_detection"] = True

# Camera initialization
def init_camera():
    if camera_option == "Webcam":
        cap = cv2.VideoCapture(0)
    elif camera_option == "IP Camera" and ip_cam_url:
        cap = cv2.VideoCapture(ip_cam_url)
    else:
        return None

    if not cap.isOpened():
        st.error("Failed to initialize camera!")
        return None
    return cap

cap = init_camera()
video_placeholder = st.empty()
status_text = st.empty()
alert_history = st.expander("Alert History", expanded=False)

# Alert function
def send_alert():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    alert_history.write(f"🚨 Alert at {timestamp} - Fire detected!")
    st.session_state.alert_triggered = True

# Initialize session state for alerts
st.session_state.setdefault("alert_triggered", False)

# Main loop
if cap and model and not st.session_state["stop_detection"]:
    last_alert_time = 0

    while cap.isOpened() and not st.session_state["stop_detection"]:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("Camera feed lost. Reconnecting...")
            time.sleep(2)
            cap = init_camera()
            if not cap:
                break
            continue

        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, channels="RGB")

        fire_detected = any(int(box.cls) in FIRE_CLASSES for box in results[0].boxes)

        if fire_detected:
            status_text.error("🚨 FIRE DETECTED! Please evacuate!")
            current_time = time.time()
            if enable_alerts and (current_time - last_alert_time) > alert_cooldown:
                threading.Thread(target=send_alert, daemon=True).start()
                last_alert_time = current_time
        else:
            status_text.success("✅ No fire detected")

        time.sleep(0.05)

    cap.release()
    st.success("🛑 Detection stopped.")

else:
    st.warning("Please configure a valid camera source")

# Play browser sound alert
if st.session_state.get("alert_triggered", False):
    st.markdown("""
        <audio autoplay>
            <source src="https://soundbible.com/mp3/Fire Truck-SoundBible.com-556371869.mp3" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)
    st.session_state.alert_triggered = False
