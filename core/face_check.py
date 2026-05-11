# PokéLeader — core/face_check.py

import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection


def check_and_crop_face(image_array):
    """
    Accepts a numpy array (RGB, from Gradio).
    Returns a cropped face numpy array (RGB, 256x256)
    or None if no face detected or quality too low.
    """
    h, w = image_array.shape[:2]

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as detector:
        results = detector.process(image_array)

    if not results.detections:
        return None, "No face detected. Please upload a clear front-facing photo."

    # Use the first (most confident) detection
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    # Convert relative to absolute coordinates
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
    if face_crop.shape[0] < 100 or face_crop.shape[1] < 100:
        return None, "Face too small. Please upload a closer photo."

    # Resize to standard size for IP-Adapter
    face_resized = cv2.resize(face_crop, (256, 256))
    return face_resized, "ok"
