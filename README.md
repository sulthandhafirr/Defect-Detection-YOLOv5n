# YOLOv5-Defect-Detection

This project provides a simple AI-based solution to detect defects in plastic bottles using the YOLOv5s object detection model. It is designed to help automate the quality control process in manufacturing, allowing faster and more accurate inspections.

# 📦 Dataset

Download and extract the dataset before training:

**📁 bottledataset6.zip**

This file is included in the repository. It contains two classes:
- `normal`: Clean, defect-free plastic bottles.
- `defect`: Bottles with visual defects such as cracks or dents.

**🏋️‍♀️ Training the Model**
The training process is done using the notebook file: YOLOv5train.ipynb
You can open this notebook in Google Colab or locally using Jupyter Notebook.

# 📌 Steps to train:
1. Open YOLOv5train.ipynb
2. Run each cell step by step.
3. Make sure the dataset path in the notebook points to your extracted bottledataset6/ folder.

# Credits
- Dataset sources: Roboflow, Kaggle, Google Images
- Object Detection: YOLOv5 by Ultralytics
