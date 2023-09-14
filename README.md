# Defect Detection System for Production Line
## Overview
This project aims to automate the defect detection process on a production line using machine learning models and the ArduCam B0371 (IMX519) camera. It leverages two different neural network architectures: Faster R-CNN with MobileNet3 for object detection and ResNet18 for defect classification. The entire algorithm is optimized for GPU acceleration and operates on the PyTorch framework, achieving defect detection in less than a second.

## Components
* **ArduCam B0371 (IMX519) Camera** - The high-quality camera sensor provides essential imagery for the system, ensuring precise defect detection.

* **Faster R-CNN on MobileNet3** - Faster R-CNN is an advanced object detection model, and MobileNet3 is a lightweight neural network architecture optimized for GPU acceleration. This optimization enables real-time processing on the production line, with defect detection occurring in less than a second.

* **ResNet18 Classifier: ResNet18** - is employed for defect classification after detection. This component identifies the specific type of defect detected.

## Purpose
This project exemplifies the application of machine learning in the industry, offering automated, effective, and accurate defect detection. The rapid defect detection, taking less than a second, further enhances production efficiency and quality.
