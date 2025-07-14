# src/video_analyzer.py

import cv2
import numpy as np
import logging
from src.utils import setup_logging
from src.pose_estimator import PoseEstimator
from src.motion_comparator import MotionComparator  # Importa o MotionComparator
import io
import os  # Importa os para manipulação de arquivos
import tempfile  # Adicionar esta importação

logger = setup_logging()


class VideoAnalyzer:
    """
    Classe para analisar vídeos, extrair frames e aplicar detecção de pose,
    e agora, comparar os movimentos de dois vídeos.
    """

    def __init__(self):
        """
        Inicializa o analisador de vídeo, o estimador de pose e o comparador de movimentos.
        """
        logger.info("Inicializando VideoAnalyzer...")
        self.pose_estimator = PoseEstimator()
        self.motion_comparator = MotionComparator()  # Inicializa o MotionComparator

        # Listas para armazenar os landmarks de todos os frames para cada vídeo
        self.aluno_landmarks_history = []
        self.mestre_landmarks_history = []

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
        temp_file_path = None  # Inicializa como None
        if isinstance(video_source, str):
            # Se for um caminho de arquivo, abre diretamente com OpenCV
            cap = cv2.VideoCapture(video_source)
            logger.info(f"Abrindo vídeo do caminho: {video_source}")
        elif isinstance(video_source, io.BytesIO):
            # Se for BytesIO, salva para um arquivo temporário para o OpenCV ler
            logger.info("Recebido BytesIO, salvando para arquivo temporário...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video_source.read())
                temp_file_path = temp_file.name
            cap = cv2.VideoCapture(temp_file_path)
            logger.info(f"Abrindo vídeo de arquivo temporário: {temp_file_path}")
        else:
            logger.error(f"Tipo de video_source não suportado: {type(video_source)}")
            raise ValueError(
                "Tipo de video_source não suportado. Deve ser str ou io.BytesIO."
            )

        if not cap.isOpened():
            logger.error(
                f"Não foi possível abrir o vídeo: {video_source if isinstance(video_source, str) else 'do BytesIO/temp file'}"
            )
            raise IOError(
                f"Não foi possível abrir o vídeo: {video_source if isinstance(video_source, str) else 'do BytesIO/temp file'}"
            )

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # logger.debug(f"Processando frame {frame_count}...")

            # Aplica a detecção de pose no frame
            annotated_frame, landmarks_data = self.pose_estimator.process_frame(frame)

            yield annotated_frame, landmarks_data  # Retorna o frame e os landmarks para o chamador

        cap.release()
        logger.info(f"Processamento de vídeo concluído. Total de frames: {frame_count}")

        # Limpa o arquivo temporário se foi criado a partir de BytesIO
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Arquivo temporário removido: {temp_file_path}")
            except OSError as e:
                logger.warning(
                    f"Erro ao remover arquivo temporário {temp_file_path}: {e}"
                )

    def store_landmarks(self, video_type: str, landmarks_data: list):
        """
        Armazena os dados de landmarks de um frame processado.

        Argumentos:
            video_type (str): O tipo de vídeo ('aluno' ou 'mestre').
            landmarks_data (list): A lista de dicionários de landmarks para um frame.
        """
        if video_type == "aluno":
            self.aluno_landmarks_history.append(landmarks_data)
            logger.debug(
                f"Landmarks do frame do aluno armazenados. Total: {len(self.aluno_landmarks_history)}"
            )
        elif video_type == "mestre":
            self.mestre_landmarks_history.append(landmarks_data)
            logger.debug(
                f"Landmarks do frame do mestre armazenados. Total: {len(self.mestre_landmarks_history)}"
            )
        else:
            logger.warning(
                f"Tipo de vídeo desconhecido '{video_type}'. Landmarks não armazenados."
            )

    def compare_processed_movements(self) -> tuple[list, list]:
        """
        Compara todos os movimentos processados do aluno com os do mestre
        usando o MotionComparator.

        Retorna:
            tuple[list, list]: Uma tupla contendo:
                - lista_comparacao_raw (list): Resultados detalhados da comparação frame a frame.
                - feedback_text (list): Feedback textual gerado pelo MotionComparator.
        """
        logger.info("Iniciando a comparação dos movimentos armazenados...")
        if not self.aluno_landmarks_history or not self.mestre_landmarks_history:
            logger.warning(
                "Não há dados de landmarks suficientes para a comparação (aluno ou mestre estão vazios)."
            )
            return [], [
                "Erro: Não há dados suficientes para comparar os movimentos. Certifique-se de que ambos os vídeos foram processados."
            ]

        raw_comparison, feedback_text = self.motion_comparator.compare_movements(
            self.aluno_landmarks_history, self.mestre_landmarks_history
        )
        logger.info("Comparação de movimentos concluída pelo MotionComparator.")
        return raw_comparison, feedback_text

    def __del__(self):
        """
        Garante que os recursos do PoseEstimator sejam liberados.
        """
        if self.pose_estimator and hasattr(self.pose_estimator, "__del__"):
            self.pose_estimator.__del__()
            logger.info("Recursos do PoseEstimator liberados via VideoAnalyzer.")
