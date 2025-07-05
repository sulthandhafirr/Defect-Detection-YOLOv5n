import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import requests
from io import BytesIO

# Load model ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("yolov5s_bottle6.onnx")

session = load_model()

# Class names (ubah sesuai datasetmu)
CLASS_NAMES = ['defect', 'normal']

# Letterbox resize
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]  # current shape [height, width]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# Preprocess image
def preprocess(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img, ratio, (dw, dh) = letterbox(img, new_shape=640)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    img /= 255.0
    return img, ratio, dw, dh

# Post-process ONNX output
# --- Tambahan fungsi bantu ---
def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms_numpy(boxes, iou_threshold=0.45):
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    final_boxes = []
    while boxes:
        chosen = boxes.pop(0)
        final_boxes.append(chosen)
        boxes = [
            box for box in boxes
            if iou(chosen["bbox"], box["bbox"]) < iou_threshold
        ]
    return final_boxes

# --- Fungsi utama postprocess ---
def postprocess(prediction, img_shape, ratio, dw, dh, conf_thres=0.5, iou_thres=0.45):
    boxes = []
    pred = prediction[0]  # shape (25200, 85)
    for det in pred:
        obj_conf = det[4]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        class_conf = class_probs[class_id]
        conf = obj_conf * class_conf
        if conf < conf_thres:
            continue
        cx, cy, w, h = det[0], det[1], det[2], det[3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        x1 = int((x1 - dw) / ratio)
        y1 = int((y1 - dh) / ratio)
        x2 = int((x2 - dw) / ratio)
        y2 = int((y2 - dh) / ratio)
        boxes.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(conf),
            "class": int(class_id)
        })

    return nms_numpy(boxes, iou_threshold=iou_thres)

# Draw detection boxes
def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        conf = box["conf"]
        cls = box["class"]
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        color = (0, 255, 0) if CLASS_NAMES[cls] == "normal" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Inference pipeline
def detect(image):
    img_np = np.array(image)
    input_tensor, ratio, dw, dh = preprocess(image)
    outputs = session.run(None, {"images": input_tensor})[0]
    boxes = postprocess([outputs[0]], img_np.shape, ratio, dw, dh)
    result_img = draw_boxes(img_np.copy(), boxes)
    return result_img

# Hide text input in selectbox (biar tidak bisa diketik)
# Inject CSS to disable typing but keep dropdown clickable



# --- Streamlit UI ---
st.set_page_config(page_title="Bottle Defect Detection", layout="centered", page_icon="🧴")
menu = st.sidebar.selectbox("Select Page", ["Home", "Upload Image", "Webcam Real-time"])

if menu == "Home":
    st.markdown("""
        <style>
            .home-title {
                color: #0066cc;
                text-align: center;
                margin-bottom: 5px;
            }
            .home-subtitle {
                text-align: center;
                font-size: 18px;
                color: inherit;
                margin-bottom: 30px;
            }
            .home-section-title {
                color: #0066cc;
                margin-top: 25px;
            }
            .home-text, .home-list {
                font-size: 16px;
                color: inherit;
            }
            .home-alert {
                font-size: 16px;
                background-color: rgba(255, 0, 0, 0.1);
                color: currentColor;
                padding: 12px;
                border-radius: 6px;
                border-left: 5px solid #ff4d4d;
            }
            .footer {
                text-align: center;
                font-size: 14px;
                color: inherit;
                margin-top: 30px;
            }
        </style>

        <h1 class='home-title'>Plastic Bottle Defect Detection</h1>
        <p class='home-subtitle'>
            Automated quality control system powered by computer vision
        </p>

        <hr>

        <h3 class='home-section-title'>Description</h3>
        <p class='home-text'>
            An AI-powered system that classifies bottles as <b>normal</b> or <b>defective</b> in real-time using <b>YOLOv5s</b>.
            Features dual input modes (image upload + live camera) with confidence-based filtering. Built with Python and deployed via Streamlit.
        </p>

        <h3 class='home-section-title'>Detection Visualization</h3>
        <ul class='home-list'>
            <li>🔴 <b>Red Bounding Box</b>: Defective bottle</li>
            <li>🟢 <b>Green Bounding Box</b>: Normal bottle</li>
        </ul>

        <h3 class='home-section-title'>Confidence Threshold</h3>
        <ul class='home-list'>
            <li>Only displays detections with <b>> 50%</b> confidence</li>
        </ul>

        <h3 class='home-section-title'>Real-time Camera Notice</h3>
        <p class='home-alert'>
            ⚠️ The real-time detection feature requires access to your local webcam.<br>
            To use this feature, please run the app locally using the <code>app.py</code> file from the GitHub repository.
        </p>

        <h3 class='home-section-title'>Tech Stack</h3>
        <ul class='home-list'>
            <li><b>AI Model:</b> YOLOv5s (PyTorch)</li>
            <li><b>Computer Vision:</b> OpenCV, Albumentations</li>
            <li><b>Data Processing:</b> NumPy, Pandas</li>
            <li><b>Deployment:</b> Streamlit</li>
        </ul>

        <hr>
        
        <p style='text-align: center; margin-top: 20px;'>
            <a href='https://github.com/sulthandhafirr/Defect-Detection-YOLOv5n' target='_blank' style='margin-right: 10px;'>
                <img src='https://img.shields.io/badge/View_Code-GitHub-181717?logo=github&style=for-the-badge'>
            </a>
            <a href='https://colab.research.google.com/drive/19hr5-IpF_GWZ8Z1VLdZWras24lDQd_9k?usp=sharing' target='_blank'>
                <img src='https://img.shields.io/badge/Train_Model-Colab-F9AB00?logo=googlecolab&style=for-the-badge'>
            </a>
        </p>

        <p class='footer'>
            Developed by: <b>Sulthan Dhafir Rafief</b>
        </p>
    """, unsafe_allow_html=True)

# Upload image page
elif menu == "Upload Image":
    st.header("Upload Image")

    uploaded_file = st.file_uploader("Upload a bottle image", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1.image(image, caption="Uploaded Image", use_container_width=True)
        if col1.button("🔍 Detect Uploaded"):
            result = detect(image)
            col2.image(result, caption="Detection Result", use_container_width=True)

    st.markdown("---")
    st.subheader("Or Try Sample Image from GitHub")

    # Daftar gambar dalam folder sample/ di GitHub
    sample_images = [
        "sample1.png",
        "sample2.png",
        "sample3.png",
        "sample4.png",
        "sample5.png",
        "sample6.jpg"
    ]

    selected_sample = st.selectbox("Choose a sample image", sample_images)

    github_raw_base = "https://github.com/sulthandhafirr/Plastic-Bottle-Defect-Detection/main/sample"
    github_raw_url = f"{github_raw_base}/{selected_sample}"

    try:
        import requests
        from io import BytesIO

        response = requests.get(github_raw_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        col1.image(image, caption=f"Sample: {selected_sample}", use_container_width=True)

        if col1.button("🔍 Detect Sample"):
            result = detect(image)
            col2.image(result, caption="Detection Result", use_container_width=True)
    except Exception as e:
        st.warning("Failed to load sample image from GitHub.")
        st.text(str(e))
            
# Webcam page
elif menu == "Webcam Real-time":
    st.header("Real-time Camera")
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
