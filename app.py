import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load model ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession("yolov5s_bottle6.onnx")

session = load_model()

# Class names
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

# IOU + NMS for postprocessing
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
        boxes = [box for box in boxes if iou(chosen["bbox"], box["bbox"]) < iou_threshold]
    return final_boxes

# Postprocess model output
def postprocess(prediction, img_shape, ratio, dw, dh, conf_thres=0.5, iou_thres=0.45):
    boxes = []
    pred = prediction[0]
    for det in pred:
        obj_conf = det[4]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        class_conf = class_probs[class_id]
        conf = obj_conf * class_conf
        if conf < conf_thres:
            continue
        cx, cy, w, h = det[0], det[1], det[2], det[3]
        x1 = int((cx - w / 2 - dw) / ratio)
        y1 = int((cy - h / 2 - dh) / ratio)
        x2 = int((cx + w / 2 - dw) / ratio)
        y2 = int((cy + h / 2 - dh) / ratio)
        boxes.append({"bbox": (x1, y1, x2, y2), "conf": float(conf), "class": int(class_id)})
    return nms_numpy(boxes, iou_threshold=iou_thres)

# Draw bounding boxes
def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        conf = box["conf"]
        cls = box["class"]
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        color = (0, 255, 0) if CLASS_NAMES[cls] == "normal" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Inference pipeline
def detect(image):
    img_np = np.array(image)
    input_tensor, ratio, dw, dh = preprocess(image)
    outputs = session.run(None, {"images": input_tensor})[0]
    boxes = postprocess([outputs[0]], img_np.shape, ratio, dw, dh)
    result_img = draw_boxes(img_np.copy(), boxes)
    return result_img

# Streamlit UI
st.set_page_config(page_title="Bottle Defect Detection", layout="centered", page_icon="🧴")
menu = st.sidebar.selectbox("Select Page", ["Home", "Upload Image", "Webcam Real-time"], label_visibility="visible")

if menu == "Home":
    st.title("Plastic Bottle Defect Detection")
    st.markdown("""
    This app detects defects in plastic bottles using a YOLOv5s ONNX model.
    - 🟢 Normal = Green Box
    - 🔴 Defect = Red Box
    """)

elif menu == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)
        if st.button("🔍 Detect"):
            result = detect(image)
            st.image(result, caption="Detection Result", use_container_width=True)

elif menu == "Webcam Real-time":
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Convert ke PIL Image dulu untuk `detect()` kamu
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = detect(pil_img)
            
            # Pastikan hasil detect() berupa ndarray BGR agar bisa ditampilkan
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    st.header("📷 Real-time Detection")
    
    webrtc_streamer(
        key="realtime",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "style": {"width": "100%", "height": "480px"},  # Kamera besar
        },
    )
