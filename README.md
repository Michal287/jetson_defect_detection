# Defect Detection System on Jetson Nano 4GB
## Overview
This project aims to automate the defect detection process on a production line using machine learning models and the ArduCam B0371 (IMX519) camera. It used two different neural network architectures: Faster R-CNN with MobileNet3 for object detection and ResNet18 for defect classification. The entire algorithm is optimized for GPU acceleration and operates on the PyTorch framework, achieving defect detection in less than a second.

## Compatibility
* **Jetson Nano 4GB** - This system is specifically designed to run on the Jetson Nano 4GB.

* **JetPack 4.6** - The project is compatible with JetPack 4.6.

## Components

* **Faster R-CNN on MobileNet3** - Faster R-CNN is an advanced object detection model, and MobileNet3 is a lightweight neural network architecture. 

* **ResNet18 Classifier: ResNet18** - is employed for defect classification after detection. This component identifies the specific type of defect detected.

* **PLC Integration** - The project is connected to PLC for  communication and integration with the production line control system.

* **ArduCam B0371 (IMX519) Camera** - 16MP camera sensor.

## Purpose
This project offering automated, effective, and accurate defect detection.

## Project Files
* ```main.py``` - Is the master file of this algorithm.

* ```models.py``` - Contains wrapped classes for simplified implementation of machine learning algorithms.

* ```connector.py``` - Includes a class for communication with the PLC.
