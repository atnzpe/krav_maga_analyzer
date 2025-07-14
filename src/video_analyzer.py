import cv2
import numpy as np
import logging
from src.utils import setup_logging
from src.pose_estimator import PoseEstimator
import io

logger = setup_logging()


class VideoAnalyzer:
    """
    Classe para analisar vídeos, extrair frames e aplicar detecção de pose.
    """

    def __init__(self):
        """
        Inicializa o analisador de vídeo e o estimador de pose.
        """
        logger.info("Inicializando VideoAnalyzer...")
        self.pose_estimator = PoseEstimator()
        logger.info("VideoAnalyzer inicializado.")

    def analyze_video(self, video_source: str | io.BytesIO):
        """
        Processa um vídeo, aplicando a detecção de pose em cada frame.

        Args:
            video_source (str | io.BytesIO): Caminho para o arquivo de vídeo (str)
                                             ou um objeto BytesIO contendo os dados do vídeo.

        Yields:
            tuple[np.ndarray, list]: Uma tupla contendo:
                - O frame processado com os landmarks desenhados.
                - Os dados dos landmarks para o frame.
        """
        if isinstance(video_source, str):
            # Se for um caminho de arquivo, abre diretamente com OpenCV
            cap = cv2.VideoCapture(video_source)
            logger.info(f"Abrindo vídeo do caminho: {video_source}")
        elif isinstance(video_source, io.BytesIO):
            # Se for BytesIO (para Streamlit/Flet uploads), precisa salvar temporariamente
            # ou usar uma abordagem baseada em buffer (mais complexo com cv2.VideoCapture)
            # A forma mais simples para cv2.VideoCapture é salvar para um arquivo temporário
            try:
                # Cria um arquivo temporário
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file.write(video_source.read())
                temp_file_path = temp_file.name
                temp_file.close()
                logger.info(
                    f"Dados de vídeo BytesIO salvos temporariamente em: {temp_file_path}"
                )
                cap = cv2.VideoCapture(temp_file_path)
            except Exception as e:
                logger.error(f"Erro ao criar arquivo temporário para BytesIO: {e}")
                raise
        else:
            raise ValueError(
                "video_source deve ser um caminho de arquivo (str) ou io.BytesIO."
            )

        if not cap.isOpened():
            logger.error(f"Não foi possível abrir o vídeo: {video_source}")
            raise IOError(f"Não foi possível abrir o vídeo: {video_source}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # logger.debug(f"Processando frame {frame_count}...")

            # Aplica a detecção de pose no frame
            annotated_frame, landmarks_data = self.pose_estimator.process_frame(frame)

            yield annotated_frame, landmarks_data

        cap.release()
        logger.info(f"Processamento de vídeo concluído. Total de frames: {frame_count}")

        # Limpa o arquivo temporário se foi criado a partir de BytesIO
        if isinstance(video_source, io.BytesIO) and "temp_file_path" in locals():
            try:
                os.remove(temp_file_path)
                logger.info(f"Arquivo temporário removido: {temp_file_path}")
            except OSError as e:
                logger.warning(
                    f"Erro ao remover arquivo temporário {temp_file_path}: {e}"
                )

    def __del__(self):
        """
        Garante que os recursos do PoseEstimator sejam liberados.
        """
        if self.pose_estimator:
            del self.pose_estimator


# Adiciona temporário para cv2.VideoCapture para lidar com BytesIO
import tempfile
import os
