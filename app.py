import pathlib
# Fix error when loading model saved in Windows on Linux
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import streamlit as st
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

# Disable Streamlit file watcher to prevent inotify limit errors
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Add yolov5 directory to path
sys.path.append(str(Path(__file__).resolve().parent / "yolov5"))

from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load model
@st.cache_resource
def load_model():
    model = attempt_load("yolov5s_bottle6.pt")
    model.eval()
    return model

model = load_model()

# Restore PosixPath after model is loaded
pathlib.PosixPath = temp

# Resize and pad image
def letterbox(im, new_shape=640, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

# Draw bounding box
def draw_box(img, xyxy, label=None, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_w, text_h = text_size
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10
        cv2.rectangle(img, (text_x, text_y - text_h), (text_x + text_w, text_y), color, -1)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# Inference
def detect(image):
    original_img = np.array(image)
    img = original_img.copy()
    img_resized, ratio, (dw, dh) = letterbox(img, 640)
    img_resized = img_resized.transpose((2, 0, 1))
    img_resized = np.ascontiguousarray(img_resized)
    img_resized = torch.from_numpy(img_resized).float()
    img_resized /= 255.0
    if img_resized.ndimension() == 3:
        img_resized = img_resized.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_resized)[0]
        pred = non_max_suppression(pred, 0.5, 0.45)

    img_result = original_img.copy()
    if pred[0] is not None:
        for *xyxy, conf, cls in pred[0]:
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = xyxy
            x1 = int((x1 - dw) / ratio[0])
            y1 = int((y1 - dh) / ratio[1])
            x2 = int((x2 - dw) / ratio[0])
            y2 = int((y2 - dh) / ratio[1])
            class_name = model.names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            color = (0, 255, 0) if 'normal' in class_name.lower() else (255, 0, 0)
            draw_box(img_result, (x1, y1, x2, y2), label=label, color=color)

    return img_result

# Streamlit interface
st.set_page_config(page_title="Bottle Defect Detection", layout="centered", page_icon="🥴")
menu = st.sidebar.selectbox("Select Page", ["Home", "Upload Image", "Webcam Real-time"])

if menu == "Home":
    st.markdown("""
        <h1 style='color: #2e86c1; text-align: center;'>Plastic Bottle Defect Detection</h1>
        <p style='text-align: center; font-size: 18px;'>AI-powered system to detect defective bottles</p>
        <hr>
        <ul>
            <li><b>Green Box</b>: Normal bottle</li>
            <li><b>Red Box</b>: Defective bottle</li>
        </ul>
        <p style='text-align: center;'>Developed by: <b>Sulthan Dhafir Rafief</b></p>
    """, unsafe_allow_html=True)

elif menu == "Upload Image":
    st.header("📸 Upload Image")
    uploaded_file = st.file_uploader("Upload a bottle image", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1.image(image, caption="Original Image", use_container_width=True)
        if col1.button("🔍 Detect"):
            result = detect(image)
            col2.image(result, caption="Detection Result", use_container_width=True)

elif menu == "Webcam Real-time":
    st.header("🎥 Real-time Camera")
    run = st.checkbox("Enable Camera")
    frame_window = st.image([], use_container_width=True)

    if 'camera' not in st.session_state:
        st.session_state.camera = None

    if run:
        st.session_state.camera = cv2.VideoCapture(0)
        while run:
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.warning("Camera not available.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detect(Image.fromarray(frame_rgb))
            frame_window.image(result, use_container_width=True)
    else:
        if st.session_state.camera:
            st.session_state.camera.release()
            st.session_state.camera = None
