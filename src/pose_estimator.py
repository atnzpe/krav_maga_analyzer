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
        # Inicializa o objeto Pose com configurações otimizadas para detecção em vídeo.
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, # Define como False para processar streams de vídeo de forma mais eficiente.
            model_complexity=1,      # 0, 1, ou 2. 1 é um bom equilíbrio entre performance e precisão.
            enable_segmentation=False, # Não precisamos de segmentação de fundo por enquanto, economiza recursos.
            min_detection_confidence=0.5, # Limiar de confiança para detecção inicial da pose.
            min_tracking_confidence=0.5   # Limiar de confiança para rastrear a pose após a detecção.
        )
        self.mp_drawing = mp.solutions.drawing_utils # Utilitários para desenhar landmarks.
        self.mp_drawing_styles = mp.solutions.drawing_styles # Estilos de desenho padrão.
        logger.info("PoseEstimator inicializado.")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Processa um único frame para detectar landmarks de pose.

        Args:
            frame (np.ndarray): O frame da imagem (formato BGR do OpenCV).

        Returns:
            tuple[np.ndarray, list]: Uma tupla contendo:
                - O frame processado com os landmarks desenhados.
                - Os dados dos landmarks (x, y, z, visibility) ou uma lista vazia se nenhum for detectado.
        """
        # Converter a imagem BGR para RGB antes de processar com MediaPipe.
        # MediaPipe espera imagens no formato RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar a imagem e detectar a pose.
        results = self.pose.process(image_rgb)

        # Desenhar os landmarks no frame original (BGR).
        # Cria uma cópia para não modificar o frame original diretamente.
        annotated_image = frame.copy()
        landmarks_list = [] # Lista para armazenar os dados dos landmarks em formato de dicionário.

        if results.pose_landmarks:
            # Desenha as conexões e os landmarks da pose na imagem.
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS, # Define como os landmarks devem ser conectados (esqueleto).
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style() # Estilo de desenho padrão.
            )
            # Armazenar os landmarks detectados em um formato mais acessível.
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
        Adiciona uma verificação para evitar fechar um objeto já nulo/fechado.
        """
        if self.pose:
            try:
                self.pose.close()
                logger.info("Recursos do MediaPipe Pose liberados.")
            except Exception as e:
                # Captura qualquer erro que possa ocorrer durante o fechamento
                # e registra como um aviso em vez de travar a aplicação.
                logger.warning(f"Erro ao fechar recursos do MediaPipe Pose em __del__: {e}")
        else:
            logger.debug("MediaPipe Pose já estava fechado ou não inicializado ao tentar fechar em __del__.")