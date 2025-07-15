# src/video_analyzer.py

import cv2
import numpy as np
import logging
from src.utils import setup_logging
from src.pose_estimator import PoseEstimator
from src.motion_comparator import MotionComparator  # Importa o MotionComparator
import io
import os  # Importa os para manipulação de arquivos
import tempfile  # Adicionar esta importação para criar arquivos temporários

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
        temp_file_path = None
        cap = None

        if isinstance(video_source, str):
            # Se for um caminho de arquivo, abre diretamente com OpenCV
            cap = cv2.VideoCapture(video_source)
            logger.info(f"Abrindo vídeo do caminho: {video_source}")
        elif isinstance(video_source, io.BytesIO):
            # Se for BytesIO, cria um arquivo temporário para o OpenCV ler
            try:
                # O parâmetro delete=False permite que o arquivo seja fechado e reaberto pelo cv2
                # e nós o removeremos explicitamente mais tarde.
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_file:
                    temp_file.write(
                        video_source.read()
                    )  # Escreve o conteúdo do BytesIO no arquivo temporário.
                    temp_file_path = temp_file.name
                cap = cv2.VideoCapture(temp_file_path)
                logger.info(
                    f"Abrindo vídeo do BytesIO via arquivo temporário: {temp_file_path}"
                )
            except Exception as e:
                logger.error(
                    f"Erro ao criar arquivo temporário ou abrir vídeo de BytesIO: {e}"
                )
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(
                        temp_file_path
                    )  # Tenta remover o arquivo temporário em caso de erro
                raise IOError(f"Não foi possível processar o vídeo do BytesIO: {e}")
        else:
            raise TypeError("video_source deve ser str ou io.BytesIO")

        if not cap.isOpened():
            logger.error(f"Não foi possível abrir o vídeo: {video_source}")
            # Tenta remover o arquivo temporário se ele foi criado e o cap não abriu
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise IOError(f"Não foi possível abrir o vídeo: {video_source}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Sai do loop se não houver mais frames

            frame_count += 1
            # logger.debug(f"Processando frame {frame_count}...")

            # Aplica a detecção de pose no frame
            annotated_frame, landmarks_data = self.pose_estimator.process_frame(frame)

            yield annotated_frame, landmarks_data  # Retorna o frame processado e os landmarks

        cap.release()  # Libera o objeto VideoCapture
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

    def store_aluno_landmarks(self, landmarks_data: list):
        """Armazena os landmarks do aluno."""
        self.aluno_landmarks_history.append(landmarks_data)

    def store_mestre_landmarks(self, landmarks_data: list):
        """Armazena os landmarks do mestre."""
        self.mestre_landmarks_history.append(landmarks_data)

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

        # Chama o MotionComparator com os históricos completos de landmarks
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
