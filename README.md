# YOLOv11 Object Tracking and Medical Video Analysis
This repository presents a real-time object tracking system built with YOLOv11, designed for medical video analysis and adaptable to broader computer vision applications (e.g., surveillance, industrial inspection, motion analysis).
The project integrates Ultralytics YOLOv11, OpenCV, and Google Colab to process raw endoscopic or clinical videos, track objects frame-by-frame, visualize detections, and automatically save annotated frames and final videos for quantitative analysis.
# Project Overview

The main goal of this work is to detect and track small, dynamic objects in videos — for instance, polyps in endoscopic recordings — and evaluate detection accuracy across multiple performance metrics.
The workflow includes:
Dataset preparation via Roboflow (annotation, versioning, augmentation)
Model training using YOLOv11n with fine-tuned augmentations
Validation and quantitative evaluation (mAP, Precision, Recall, F1)
Real-time object tracking on medical videos
Saving results (annotated frames + output video in /report/)
