# src/pose_estimator.py

import mediapipe as mp
import cv2
import numpy as np
from src.utils import get_logger  # Importa get_logger do seu módulo de utilidades

# Inicializa o logger para este módulo
logger = get_logger(__name__)


class PoseEstimator:
    """
    Classe responsável por estimar a pose de indivíduos em frames de vídeo
    usando a solução MediaPipe Pose.
    """

    def __init__(self):
        """
        Inicializa o PoseEstimator, configurando o modelo MediaPipe Pose
        e as utilidades de desenho.
        """
        logger.info("Inicializando PoseEstimator com MediaPipe Pose...")

        # Inicializa o modelo MediaPipe Pose.
        # static_image_mode=False: para processar vídeo.
        # model_complexity=1: complexidade do modelo (0, 1, 2). 1 é um bom equilíbrio.
        # min_detection_confidence: confiança mínima para detecção de pose.
        # min_tracking_confidence: confiança mínima para rastreamento de pose.
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("Modelo MediaPipe Pose inicializado.")

        # Inicializa as utilidades de desenho do MediaPipe para visualização.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        logger.info("Utilidades de desenho do MediaPipe inicializadas.")

    def estimate_pose(self, image: np.ndarray):
        """
        Estima a pose em um único frame de imagem.

        Args:
            image (np.ndarray): O frame de imagem (array NumPy BGR).

        Returns:
            tuple: Uma tupla contendo (results, annotated_image).
                   results: O objeto MediaPipe PoseResults contendo os landmarks detectados.
                   annotated_image: O frame com os landmarks e conexões desenhados.
        """
        # Converter a imagem de BGR para RGB, que é o formato esperado pelo MediaPipe.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Opcional: tornar a imagem não-gravável para melhor desempenho.
        image_rgb.flags.writeable = False

        # Processar a imagem para estimar a pose.
        results = self.pose.process(image_rgb)

        # Tornar a imagem gravável novamente para desenhar as anotações.
        image_rgb.flags.writeable = True
        # Converter de volta para BGR para exibição com OpenCV.
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Desenhar os landmarks da pose na imagem, se detectados.
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return results, annotated_image

    def get_landmarks_as_list(self, pose_landmarks):
        """
        Converte o objeto PoseLandmarks do MediaPipe em uma lista de dicionários
        para facilitar o processamento.

        Args:
            pose_landmarks: O objeto `mp.solutions.pose.PoseLandmarks`.

        Returns:
            list: Uma lista de dicionários, onde cada dicionário representa um landmark
                  com chaves 'x', 'y', 'z', 'visibility'.
        """
        if not pose_landmarks:
            return None
        landmarks_list = []
        for landmark in pose_landmarks.landmark:
            landmarks_list.append(
                {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }
            )
        return landmarks_list

    def __del__(self):
        """
        Libera os recursos do modelo MediaPipe Pose quando o objeto é destruído.
        """
        logger.info("Destruindo PoseEstimator e liberando recursos do MediaPipe Pose.")
        if self.pose:  # Verifica se self.pose foi inicializado
            self.pose.close()
            logger.info("Recursos do MediaPipe Pose liberados.")
