# PokéLeader — core/face_check.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageEnhance
import os

# Model path for MediaPipe Tasks API
MODEL_PATH = 'blaze_face_short_range.tflite'

def preprocess_image(image_array):
    """
    Enhance image quality before face detection.
    Helps with low quality, dark, or blurry webcam images.
    """
    pil_img = Image.fromarray(image_array)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.3)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    return np.array(pil_img)


def detect_face(image_array, confidence):
    """
    Run MediaPipe face detection on an image array.
    Returns detections or None.
    """
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=confidence
    )

    with vision.FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        results = detector.detect(mp_image)

    return results.detections if results.detections else None


def check_and_crop_face(image_array):
    """
    Accepts a numpy array (RGB, from Gradio).
    Returns a cropped face numpy array (RGB, 256x256)
    or None if no face detected or quality too low.
    """
    if image_array is None:
        return None, "No image provided."

    if not os.path.exists(MODEL_PATH):
        return None, f"Face detection model missing. Please ensure {MODEL_PATH} is in the project root."

    h, w = image_array.shape[:2]
    enhanced = preprocess_image(image_array)

    # Attempt 1: original image, normal confidence
    detections = detect_face(image_array, confidence=0.5)

    # Attempt 2: enhanced image, lower confidence
    if not detections:
        detections = detect_face(enhanced, confidence=0.3)

    # Attempt 3: enhanced image, very low confidence
    if not detections:
        detections = detect_face(enhanced, confidence=0.2)

    if not detections:
        return None, "No face detected. Try better lighting, move closer, or face the camera directly."

    # Use the most confident detection
    # MediaPipe Tasks API uses categories[0].score, not score[0]
    detection = max(detections, key=lambda d: d.categories[0].score if d.categories else 0)
    bbox = detection.bounding_box

    # MediaPipe Tasks API returns absolute coordinates
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
    if face_crop.shape[0] < 60 or face_crop.shape[1] < 60:
        return None, "Face too small. Please move closer to the camera."

    # Resize to standard size for IP-Adapter
    face_resized = cv2.resize(face_crop, (256, 256))
    return face_resized, "ok"