"""Face detection using MTCNN from facenet_pytorch."""

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN


class FaceDetector:
    """Detect and crop faces from images using MTCNN.

    Converts between BGR (OpenCV) and RGB internally so callers
    can pass raw OpenCV frames directly.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(
            keep_all=True,
            select_largest=True,
            min_face_size=40,
            device=self.device,
        )

    def detect(self, image: np.ndarray) -> list:
        """Detect faces in a BGR (OpenCV) image.

        Args:
            image: HxWxC numpy array in BGR colour order.

        Returns:
            List of dicts, each containing:
                - bbox: [x1, y1, x2, y2] as floats
                - confidence: detection probability
                - landmarks: 5x2 numpy array of facial landmarks
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, confidences, landmarks = self.mtcnn.detect(rgb, landmarks=True)

        results = []
        if boxes is None:
            return results

        for i in range(len(boxes)):
            det = {
                "bbox": boxes[i].tolist(),
                "confidence": float(confidences[i]),
                "landmarks": landmarks[i] if landmarks is not None else None,
            }
            results.append(det)

        return results

    def detect_and_crop(
        self, image: np.ndarray, target_size: tuple = (112, 112)
    ) -> list:
        """Detect faces and return cropped, resized face images.

        Args:
            image: HxWxC numpy array in BGR colour order.
            target_size: (width, height) to resize each cropped face.

        Returns:
            List of (cropped_face_RGB, detection_info) tuples.
            cropped_face_RGB is a uint8 numpy array in RGB order.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detect(image)

        results = []
        h, w = rgb.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Clamp coordinates to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = rgb[y1:y2, x1:x2]
            crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            results.append((crop, det))

        return results
