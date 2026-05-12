
# PokéLeader — core/face_check.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# Model path for MediaPipe Tasks API
MODEL_PATH = 'blaze_face_short_range.tflite'

def check_and_crop_face(image_array):
    """
    Accepts a numpy array (RGB, from Gradio).
    Returns a cropped face numpy array (RGB, 256x256)
    or None if no face detected or quality too low.
    """
    if not os.path.exists(MODEL_PATH):
        return None, f"Face detection model missing. Please ensure {MODEL_PATH} is in the project root."

    h, w = image_array.shape[:2]

    # Initialize MediaPipe Face Detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    
    with vision.FaceDetector.create_from_options(options) as detector:
        # MediaPipe expects Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        results = detector.detect(mp_image)

    if not results.detections:
        return None, "No face detected. Please upload a clear front-facing photo."

    # Use the first (most confident) detection
    detection = results.detections[0]
    bbox = detection.bounding_box

    # MediaPipe Tasks API returns absolute coordinates for bounding_box
    x = bbox.origin_x
    y = bbox.origin_y
    bw = bbox.width
    bh = bbox.height

    # Add padding around the face
    pad = int(max(bw, bh) * 0.3)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    face_crop = image_array[y1:y2, x1:x2]

    # Quality check — face too small
    if face_crop.shape[0] < 100 or face_crop.shape[1] < 100:
        return None, "Face too small. Please upload a closer photo."

    # Resize to standard size for IP-Adapter
    face_resized = cv2.resize(face_crop, (256, 256))
    return face_resized, "ok"
