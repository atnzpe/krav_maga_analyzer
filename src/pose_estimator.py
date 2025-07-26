# src/pose_estimator.py

import mediapipe as mp
import cv2
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


class PoseEstimator:
    """
    Estima a pose usando MediaPipe Pose.
    """

    def __init__(self):
        logger.info("Inicializando PoseEstimator com MediaPipe Pose...")
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # --- NOVOS ESTILOS DE DESENHO ---
        self.default_style = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=2, circle_radius=2
        )
        self.correct_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )
        self.incorrect_style = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2
        )

        logger.info("PoseEstimator inicializado.")

    def estimate_pose(self, image: np.ndarray, style=None):
        """
        Estima a pose em um único frame.
        Args:
            image (np.ndarray): O frame de imagem.
            style: O estilo de desenho a ser usado (default, correct, incorrect).
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            draw_spec = self.default_style if style is None else style
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=draw_spec,
            )
        return results, annotated_image

    def get_landmarks_as_list(self, pose_landmarks):
        if not pose_landmarks:
            return None
        return [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in pose_landmarks.landmark
        ]

    def __del__(self):
        if hasattr(self, "pose") and self.pose:
            self.pose.close()
            logger.info("Recursos do MediaPipe Pose liberados.")
