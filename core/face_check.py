# PokéLeader — core/face_check.py

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageEnhance

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


def check_and_crop_face(image_array):
    """
    Accepts a numpy array (RGB, from Gradio).
    Returns a cropped face numpy array (RGB, 256x256)
    or None if no face detected or quality too low.
    """
    if image_array is None:
        return None, "No image provided."

    h, w = image_array.shape[:2]

    # Try detection on original first, then enhanced if it fails
    for attempt, img in enumerate([image_array, preprocess_image(image_array)]):
        mp_face = mp.solutions.face_detection

        with mp_face.FaceDetection(
            model_selection=1,          # 1 = full range model, handles far/small faces better
            min_detection_confidence=0.3 if attempt == 0 else 0.2  # more lenient on second try
        ) as detector:
            results = detector.process(img)

        if results.detections:
            break  # face found, stop trying

    if not results.detections:
        return None, "No face detected. Try better lighting, move closer, or face the camera directly."

    # Use the most confident detection
    detection = max(results.detections, key=lambda d: d.score[0])
    bbox = detection.location_data.relative_bounding_box

    # Convert relative coordinates to absolute
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

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