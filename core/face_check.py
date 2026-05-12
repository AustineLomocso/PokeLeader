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
    if image_array is None:
        return None, "No image provided."

    if not os.path.exists(MODEL_PATH):
        return None, f"Face detection model missing. Please ensure {MODEL_PATH} is in the project root."

    # FIX 1: Strip Alpha Channel if webcam sent RGBA
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    h, w = image_array.shape[:2]
    enhanced = preprocess_image(image_array)

    # Attempt detection
    detections = detect_face(image_array, confidence=0.5)
    used_enhanced = False

    if not detections:
        detections = detect_face(enhanced, confidence=0.3)
        used_enhanced = True
    if not detections:
        detections = detect_face(enhanced, confidence=0.2)
        used_enhanced = True

    if not detections:
        return None, "No face detected. Try better lighting, move closer, or face the camera directly."

    # FIX 2: Use the enhanced image for the crop if it was needed for detection
    source_image = enhanced if used_enhanced else image_array

    detection = max(detections, key=lambda d: d.categories[0].score if d.categories else 0)
    bbox = detection.bounding_box

    # FIX 3: Force a perfect square bounding box to prevent aspect ratio squash
    center_x = bbox.origin_x + (bbox.width / 2)
    center_y = bbox.origin_y + (bbox.height / 2)
    
    # Calculate the size based on the largest dimension plus 30% padding
    size = max(bbox.width, bbox.height) * 1.3
    half_size = size / 2

    x1 = max(0, int(center_x - half_size))
    y1 = max(0, int(center_y - half_size))
    x2 = min(w, int(center_x + half_size))
    y2 = min(h, int(center_y + half_size))

    face_crop = source_image[y1:y2, x1:x2]

    # Quality check
    if face_crop.shape[0] < 60 or face_crop.shape[1] < 60:
        return None, "Face too small. Please move closer to the camera."

    face_resized = cv2.resize(face_crop, (256, 256))
    return face_resized, "ok"