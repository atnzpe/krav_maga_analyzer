import mediapipe as mp
import cv2
import numpy as np
import logging
from src.utils import setup_logging

logger = setup_logging()

class PoseEstimator:
    """
    Gerencia a detecção de pose usando MediaPipe.
    """
    def __init__(self):
        """
        Inicializa o modelo de pose do MediaPipe.
        """
        logger.info("Inicializando PoseEstimator com MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # 0, 1, ou 2. 1 é um bom equilíbrio.
            enable_segmentation=False, # Não precisamos de segmentação de fundo por enquanto
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        logger.info("PoseEstimator inicializado.")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Processa um único frame para detectar landmarks de pose.

        Args:
            frame (np.ndarray): O frame da imagem (formato BGR do OpenCV).

        Returns:
            tuple[np.ndarray, list]: Uma tupla contendo:
                - O frame com os landmarks desenhados.
                - Uma lista de landmarks detectados (objetos NormalizedLandmark),
                  ou uma lista vazia se nenhum for detectado.
        """
        # Converter a imagem BGR para RGB antes de processar com MediaPipe.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar a imagem e detectar a pose.
        results = self.pose.process(image_rgb)

        # Desenhar os landmarks no frame original (BGR).
        annotated_image = frame.copy()
        landmarks_list = []

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            # Armazenar os landmarks detectados
            for landmark in results.pose_landmarks.landmark:
                landmarks_list.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
        else:
            logger.debug("Nenhum landmark de pose detectado neste frame.")

        return annotated_image, landmarks_list

    def __del__(self):
        """
        Libera os recursos do MediaPipe quando o objeto é destruído.
        """
        if self.pose:
            self.pose.close()
            logger.info("Recursos do MediaPipe Pose liberados.")