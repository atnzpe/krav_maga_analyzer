# src/pose_estimator.py

import mediapipe as mp
import cv2
import numpy as np
from src.utils import get_logger

# Obtém uma instância do logger para este módulo.
logger = get_logger(__name__)


class PoseEstimator:
    """
    Estima a pose usando MediaPipe Pose e permite desenhar esqueletos com estilos customizados.
    """

    def __init__(self):
        """
        Construtor da classe PoseEstimator.
        Inicializa o modelo MediaPipe Pose e define os diferentes estilos de desenho.
        """
        logger.info("Inicializando PoseEstimator com MediaPipe Pose...")
        # Inicializa o modelo de detecção de pose do MediaPipe.
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Utilitário de desenho do MediaPipe.
        self.mp_drawing = mp.solutions.drawing_utils
        # Mapeamento dos nomes dos landmarks para seus índices.
        self.landmark_indices = {
            name: landmark.value
            for name, landmark in mp.solutions.pose.PoseLandmark.__members__.items()
        }

        # --- ESTILOS DE DESENHO PARA VÍDEO E PDF ---
        # Cor branca padrão para o esqueleto.
        self.default_style = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=2, circle_radius=2
        )
        # Cor verde para feedback de acerto.
        self.correct_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=3
        )
        # Cor vermelha para feedback de erro.
        self.incorrect_style = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255), thickness=3, circle_radius=3
        )
        # Cor laranja para o lado esquerdo do corpo (no vídeo).
        self.left_side_style = self.mp_drawing.DrawingSpec(
            color=(0, 165, 255), thickness=2, circle_radius=2
        )  # Laranja
        # Cor azul para o lado direito do corpo (no vídeo).
        self.right_side_style = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=2, circle_radius=2
        )  # Azul

        # Mapeamento das conexões do esqueleto para os lados esquerdo e direito.
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.left_connections = {
            conn for conn in self.pose_connections if "LEFT" in str(conn)
        }
        self.right_connections = {
            conn for conn in self.pose_connections if "RIGHT" in str(conn)
        }
        # Conexões centrais (tronco) que não são nem esquerdas nem direitas.
        self.center_connections = (
            self.pose_connections - self.left_connections - self.right_connections
        )

        logger.info(
            "PoseEstimator inicializado com estilos de desenho por lado e por feedback."
        )

    def estimate_pose(self, image: np.ndarray):
        """
        Estima a pose em um único frame, sem desenhar o esqueleto.
        Apenas retorna os resultados da detecção.

        Args:
            image (np.ndarray): O frame de imagem em formato BGR.

        Returns:
            mediapipe.python.solutions.pose.PoseResults: Os resultados da detecção de pose.
        """
        # Converte a imagem de BGR para RGB, o formato esperado pelo MediaPipe.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Melhora o desempenho marcando a imagem como não-gravável.
        image_rgb.flags.writeable = False
        # Processa a imagem para detectar a pose.
        results = self.pose.process(image_rgb)
        # Retorna os resultados.
        return results

    def draw_skeleton_by_side(self, image: np.ndarray, landmarks) -> np.ndarray:
        """
        Desenha o esqueleto na imagem com cores diferentes para cada lado.
        Lado esquerdo em laranja, lado direito em azul.

        Args:
            image (np.ndarray): A imagem onde o esqueleto será desenhado.
            landmarks: Os landmarks de pose detectados pelo MediaPipe.

        Returns:
            np.ndarray: A imagem com o esqueleto colorido desenhado.
        """
        annotated_image = image.copy()
        if landmarks:
            # Desenha as conexões do lado esquerdo em laranja.
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.left_connections,
                landmark_drawing_spec=self.left_side_style,
                connection_drawing_spec=self.left_side_style,
            )
            # Desenha as conexões do lado direito em azul.
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.right_connections,
                landmark_drawing_spec=self.right_side_style,
                connection_drawing_spec=self.right_side_style,
            )
            # Desenha as conexões centrais com o estilo padrão (branco).
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.center_connections,
                landmark_drawing_spec=self.default_style,
                connection_drawing_spec=self.default_style,
            )
        return annotated_image

    def draw_feedback_skeleton(
        self,
        image: np.ndarray,
        landmarks,
        angle_diffs: dict,
        key_angles: dict,
        threshold: float = 15.0,
    ) -> np.ndarray:
        """
        Desenha o esqueleto na imagem destacando acertos (verde) e erros (vermelho)
        com base nas diferenças de ângulo.

        Args:
            image (np.ndarray): A imagem onde o esqueleto será desenhado.
            landmarks: Os landmarks de pose detectados.
            angle_diffs (dict): Dicionário com a diferença de cada ângulo.
            key_angles (dict): Dicionário que mapeia nomes de ângulos para os landmarks que os compõem.
            threshold (float): O limiar de diferença para considerar um ângulo incorreto.

        Returns:
            np.ndarray: A imagem com o esqueleto de feedback desenhado.
        """
        annotated_image = image.copy()
        if not landmarks:
            return annotated_image

        # Converte landmarks para uma lista de dicionários para fácil acesso.
        lm_list = self.get_landmarks_as_list(landmarks)
        if not lm_list:
            return annotated_image

        drawn_connections = set()

        # Itera sobre cada ângulo para decidir a cor.
        for angle_name, diff in angle_diffs.items():
            style = self.correct_style if diff <= threshold else self.incorrect_style

            # Obtém os landmarks que formam o ângulo.
            p1_name, p2_name, p3_name = key_angles[angle_name]

            # Obtém os índices desses landmarks.
            p1_idx = self.landmark_indices[p1_name]
            p2_idx = self.landmark_indices[p2_name]
            p3_idx = self.landmark_indices[p3_name]

            # Define as duas conexões que formam o ângulo (ex: Ombro-Cotovelo, Cotovelo-Pulso).
            connections_to_draw = [
                tuple(sorted((p1_idx, p2_idx))),
                tuple(sorted((p2_idx, p3_idx))),
            ]

            # Desenha cada conexão com a cor apropriada.
            for p_start_idx, p_end_idx in connections_to_draw:
                # Evita desenhar a mesma conexão duas vezes.
                if (p_start_idx, p_end_idx) in drawn_connections:
                    continue

                # Garante que os landmarks têm visibilidade suficiente antes de desenhar.
                if (
                    lm_list[p_start_idx]["visibility"] > 0.5
                    and lm_list[p_end_idx]["visibility"] > 0.5
                ):
                    # Desenha a linha (conexão).
                    cv2.line(
                        annotated_image,
                        (
                            int(lm_list[p_start_idx]["x"] * image.shape[1]),
                            int(lm_list[p_start_idx]["y"] * image.shape[0]),
                        ),
                        (
                            int(lm_list[p_end_idx]["x"] * image.shape[1]),
                            int(lm_list[p_end_idx]["y"] * image.shape[0]),
                        ),
                        style.color,
                        style.thickness,
                    )
                    # Desenha os círculos (landmarks).
                    cv2.circle(
                        annotated_image,
                        (
                            int(lm_list[p_start_idx]["x"] * image.shape[1]),
                            int(lm_list[p_start_idx]["y"] * image.shape[0]),
                        ),
                        style.circle_radius,
                        style.color,
                        -1,
                    )
                    cv2.circle(
                        annotated_image,
                        (
                            int(lm_list[p_end_idx]["x"] * image.shape[1]),
                            int(lm_list[p_end_idx]["y"] * image.shape[0]),
                        ),
                        style.circle_radius,
                        style.color,
                        -1,
                    )

                drawn_connections.add((p_start_idx, p_end_idx))

        return annotated_image

    def get_landmarks_as_list(self, pose_landmarks):
        """Converte o objeto de landmarks do MediaPipe para uma lista de dicionários."""
        if not pose_landmarks:
            return None
        return [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in pose_landmarks.landmark
        ]

    def __del__(self):
        """Destrutor para liberar os recursos do MediaPipe Pose."""
        if hasattr(self, "pose") and self.pose:
            self.pose.close()
            logger.info("Recursos do MediaPipe Pose liberados.")
