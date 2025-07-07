# Plastic Bottle YOLOv5n Defect Detection

This project provides a simple AI-based solution to detect defects in plastic bottles using the YOLOv5s object detection model. It is designed to help automate the quality control process in manufacturing, allowing faster and more accurate inspections.

# Dataset

Download and extract the dataset before training:

**📁 bottledataset6.zip**

This file is included in the repository. It contains two classes:
- `normal`: Clean, defect-free plastic bottles.
- `defect`: Bottles with visual defects such as cracks or dents.

# Steps to train:
The training process is done using the notebook file: YOLOv5train.ipynb

You can open this notebook in Google Colab or locally using Jupyter Notebook.
1. Open YOLOv5train.ipynb
2. Run each cell step by step.
3. Make sure the dataset path in the notebook points to your extracted bottledataset6/ folder.


## How to Run `app.py` Locally

Follow these steps to run the app on your local machine:

### 1. Download Files

Download the following files and place them in the same folder:
- `app.py`
- `yolov5s_bottledataset6.onnx`
- (optional) `requirements.txt` if using `pip install -r requirements.txt`


### 2. Install Dependencies

Make sure Python 3.8+ is installed. Then run:

```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install streamlit opencv-python pillow numpy onnxruntime
```

### 3. Run the App
In the terminal, navigate to the project folder and run:

```bash
streamlit run app.py
```
or 
```bash
http://localhost:8501
```

# Credits
- Dataset sources: Roboflow, Kaggle, Google Images
- Object Detection: YOLOv5 by Ultralytics
- Conversion to ONNX: PyTorch → ONNX export
- Web Interface: Streamlit

---
Developed by: Sulthan Dhafir Rafief
Demo: https://plastic-bottle-defect-detection.streamlit.app/
