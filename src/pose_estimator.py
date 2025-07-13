# src/pose_estimator.py

import cv2
import mediapipe as mp
import numpy as np

from src.utils import setup_logging  # Importa a função de setup de logging

# Configura o logger para este módulo específico.
logger = setup_logging()


class PoseEstimator:
    """
    Encapsula a funcionalidade de detecção de pose usando MediaPipe.

    Esta classe é responsável por carregar o modelo de pose do MediaPipe,
    processar frames de vídeo para identificar landmarks corporais
    e opcionalmente desenhar esses landmarks nos frames.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Inicializa o modelo de pose do MediaPipe.

        Args:
            static_image_mode (bool): Se `True`, o modelo detecta a pose em cada imagem de entrada,
                                      útil para imagens estáticas. Se `False`, rastreia a pose,
                                      tornando-o mais rápido para vídeos.
            model_complexity (int): Complexidade do modelo de pose: 0, 1 ou 2.
                                    Maior complexidade = maior precisão, menor velocidade.
            smooth_landmarks (bool): Se `True`, suaviza os landmarks entre os frames para vídeos.
            enable_segmentation (bool): Se `True`, o modelo também produzirá uma máscara de segmentação.
            smooth_segmentation (bool): Se `True`, suaviza a máscara de segmentação entre os frames.
            min_detection_confidence (float): Confiança mínima para a detecção de pose ser considerada bem-sucedida.
            min_tracking_confidence (float): Confiança mínima para o rastreamento de pose ser considerado bem-sucedido.
        """
        logger.info(
            f"Inicializando PoseEstimator com min_detection_confidence={min_detection_confidence}, min_tracking_confidence={min_tracking_confidence}"
        )
        # Inicializa os módulos de desenho e pose do MediaPipe.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Cria uma instância do modelo de pose do MediaPipe.
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info("Modelo MediaPipe Pose inicializado com sucesso.")

    def process_frame(self, image: np.ndarray):
        """
        Processa um único frame (imagem) para detectar landmarks de pose.

        Args:
            image (np.ndarray): O frame da imagem em formato NumPy array (BGR).

        Returns:
            tuple: Uma tupla contendo:
                - annotated_image (np.ndarray): A imagem original com os landmarks e conexões desenhados.
                                               Retorna a imagem original se nenhum landmark for detectado.
                - results (mediapipe.python.solution_base.SolutionOutputs): Objeto contendo os resultados
                                                                           da detecção de pose (landmarks).
                                                                           Retorna None se nenhum landmark for detectado.
        """
        # Aplicação do Zen of Python: "Beautiful is better than ugly."
        # Conversão de BGR para RGB: MediaPipe espera imagens RGB. OpenCV lê em BGR.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Define a imagem como não gravável para otimizar o desempenho.
        image_rgb.flags.writeable = False

        logger.debug("Processando frame para detecção de pose.")
        # Processa a imagem para detectar a pose.
        results = self.pose.process(image_rgb)

        # Torna a imagem gravável novamente para desenhar os landmarks.
        image_rgb.flags.writeable = True
        # Converte de volta para BGR para exibição com OpenCV (ou Streamlit).
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            logger.debug("Landmarks de pose detectados. Desenhando no frame.")
            # Desenha os landmarks da pose na imagem.
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        else:
            logger.debug("Nenhum landmark de pose detectado neste frame.")

        return annotated_image, results

    def close(self):
        """
        Libera os recursos do modelo de pose do MediaPipe.
        É importante chamar este método quando a detecção de pose não for mais necessária.
        """
        logger.info("Fechando o modelo MediaPipe Pose.")
        self.pose.close()


# Exemplo de uso (apenas para demonstração, não será executado na aplicação principal)
if __name__ == "__main__":
    logger.info("Executando exemplo de PoseEstimator.")
    # Crie uma imagem de teste (preta)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Imagem preta 640x480
    cv2.putText(
        dummy_image,
        "No real pose here",
        (100, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    estimator = PoseEstimator()
    annotated_img, pose_results = estimator.process_frame(dummy_image)

    # Exibir a imagem anotada (apenas para teste visual)
    # cv2.imshow("Annotated Dummy Image", annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if pose_results and pose_results.pose_landmarks:
        logger.info(
            f"Número de landmarks detectados: {len(pose_results.pose_landmarks.landmark)}"
        )
    else:
        logger.info("Nenhum landmark detectado na imagem de teste (esperado).")

    estimator.close()
    logger.info("Exemplo de PoseEstimator finalizado.")
