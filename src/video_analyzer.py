# src/video_analyzer.py

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tempfile
import threading
import io  # Adicionado para lidar com bytes em memória

from src.utils import get_logger  # Importa get_logger do seu módulo de utilidades
from src.pose_estimator import PoseEstimator  # Importa PoseEstimator
from src.motion_comparator import MotionComparator  # Importa MotionComparator

# Inicializa o logger para este módulo
logger = get_logger(__name__)


class VideoAnalyzer:
    """
    Classe responsável por analisar vídeos, detectar poses, comparar movimentos
    e fornecer feedback.
    """

    def __init__(self):
        """
        Inicializa o VideoAnalyzer, configurando o estimador de pose e o comparador de movimento.
        """
        logger.info("Inicializando VideoAnalyzer...")  # Esta linha agora deve funcionar

        # Inicializa o estimador de pose (MediaPipe Pose)
        self.pose_estimator = PoseEstimator()
        logger.info("PoseEstimator inicializado.")

        # Inicializa o comparador de movimento
        self.motion_comparator = MotionComparator()
        logger.info("MotionComparator inicializado.")

        self.cap_aluno = None
        self.cap_mestre = None
        self.video_aluno_path = None
        self.video_mestre_path = None
        self.aluno_landmarks = []
        self.mestre_landmarks = []
        self.comparison_results = []
        self.is_processing = False
        self.processing_thread = None
        self.current_frame_aluno = None
        self.current_frame_mestre = None
        logger.info("Variáveis de estado do VideoAnalyzer configuradas.")

    def load_video_from_bytes(self, video_bytes: bytes, is_aluno: bool):
        """
        Carrega um vídeo a partir de bytes e o salva temporariamente para processamento.

        Args:
            video_bytes (bytes): Os bytes do arquivo de vídeo.
            is_aluno (bool): True se for o vídeo do aluno, False se for do mestre.
        Returns:
            str: O caminho para o arquivo de vídeo temporário.
        Raises:
            Exception: Se houver um erro ao salvar o arquivo.
        """
        try:
            # Cria um arquivo temporário para o vídeo
            # NamedTemporaryFile garante que o arquivo é excluído quando fechado
            # ou quando o programa termina, a menos que delete=False
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(video_bytes)
            temp_file.close()
            video_path = temp_file.name
            logger.info(f"Vídeo temporário salvo em: {video_path}")

            if is_aluno:
                self.video_aluno_path = video_path
                self.cap_aluno = cv2.VideoCapture(video_path)
                if not self.cap_aluno.isOpened():
                    raise Exception(
                        f"Não foi possível abrir o vídeo do aluno em {video_path}"
                    )
                logger.info(f"Vídeo do aluno carregado: {video_path}")
            else:
                self.video_mestre_path = video_path
                self.cap_mestre = cv2.VideoCapture(video_path)
                if not self.cap_mestre.isOpened():
                    raise Exception(
                        f"Não foi possível abrir o vídeo do mestre em {video_path}"
                    )
                logger.info(f"Vídeo do mestre carregado: {video_path}")
            return video_path
        except Exception as e:
            logger.error(f"Erro ao carregar vídeo de bytes: {e}")
            raise

    def process_video(self, video_path: str, is_aluno: bool):
        """
        Processa um vídeo para extrair landmarks de pose.

        Args:
            video_path (str): O caminho para o arquivo de vídeo.
            is_aluno (bool): True se for o vídeo do aluno, False se for do mestre.
        Returns:
            list: Uma lista de landmarks extraídos.
        """
        logger.info(
            f"Iniciando processamento do vídeo: {video_path} (Aluno: {is_aluno})"
        )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Não foi possível abrir o vídeo: {video_path}")
            return []

        landmarks_list = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Redimensionar o frame para acelerar o processamento, se necessário
            # frame = cv2.resize(frame, (640, 480))

            # Estimar pose
            results, annotated_frame = self.pose_estimator.estimate_pose(frame)
            if results.pose_landmarks:
                landmarks_list.append(results.pose_landmarks)
            else:
                landmarks_list.append(
                    None
                )  # Adiciona None se nenhuma pose for detectada

            # Armazenar o frame anotado para exibição posterior, se necessário
            if is_aluno:
                self.current_frame_aluno = annotated_frame
            else:
                self.current_frame_mestre = annotated_frame

            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug(f"Processando frame {frame_count} de {video_path}")

        cap.release()
        logger.info(
            f"Processamento do vídeo {video_path} concluído. Total de frames com landmarks: {len([l for l in landmarks_list if l is not None])}"
        )
        return landmarks_list

    def analyze_and_compare(self):
        """
        Inicia o processo de análise e comparação dos vídeos do aluno e do mestre.
        """
        if not self.video_aluno_path or not self.video_mestre_path:
            logger.warning(
                "Caminhos dos vídeos não definidos. Carregue os vídeos primeiro."
            )
            return False

        if self.is_processing:
            logger.info("Análise já em andamento.")
            return False

        self.is_processing = True
        logger.info("Iniciando a thread de processamento de vídeo.")
        self.processing_thread = threading.Thread(target=self._run_analysis_thread)
        self.processing_thread.start()
        return True

    def _run_analysis_thread(self):
        """
        Método executado em uma thread separada para processar e comparar os vídeos.
        """
        try:
            logger.info("Thread de análise iniciada.")
            self.aluno_landmarks = self.process_video(
                self.video_aluno_path, is_aluno=True
            )
            self.mestre_landmarks = self.process_video(
                self.video_mestre_path, is_aluno=False
            )

            if not self.aluno_landmarks or not self.mestre_landmarks:
                logger.warning(
                    "Não foi possível extrair landmarks de um ou ambos os vídeos. Comparação abortada."
                )
                self.is_processing = False
                return

            logger.info("Iniciando comparação de movimentos...")
            self.comparison_results = self.motion_comparator.compare_poses(
                self.aluno_landmarks, self.mestre_landmarks
            )
            logger.info(
                f"Comparação de movimentos concluída. Resultados: {len(self.comparison_results)} pontos de comparação."
            )

        except Exception as e:
            logger.error(f"Erro durante a análise e comparação na thread: {e}")
        finally:
            self.is_processing = False
            logger.info("Thread de análise finalizada.")

    def get_current_annotated_frames(self):
        """
        Retorna os frames anotados atuais do aluno e do mestre.
        """
        return self.current_frame_aluno, self.current_frame_mestre

    def get_comparison_results(self):
        """
        Retorna os resultados da última comparação.
        """
        return self.comparison_results

    def __del__(self):
        """
        Limpa os recursos quando o objeto VideoAnalyzer é destruído.
        """
        logger.info("Destruindo VideoAnalyzer e liberando recursos.")
        if self.cap_aluno and self.cap_aluno.isOpened():
            self.cap_aluno.release()
            logger.info("Cap_aluno liberado.")
        if self.cap_mestre and self.cap_mestre.isOpened():
            self.cap_mestre.release()
            logger.info("Cap_mestre liberado.")
        if self.video_aluno_path and os.path.exists(self.video_aluno_path):
            os.remove(self.video_aluno_path)
            logger.info(
                f"Arquivo temporário do aluno removido: {self.video_aluno_path}"
            )
        if self.video_mestre_path and os.path.exists(self.video_mestre_path):
            os.remove(self.video_mestre_path)
            logger.info(
                f"Arquivo temporário do mestre removido: {self.video_mestre_path}"
            )

        # Garante que os recursos do PoseEstimator e MotionComparator sejam liberados
        if self.pose_estimator and hasattr(self.pose_estimator, "__del__"):
            self.pose_estimator.__del__()
            logger.info("PoseEstimator recursos liberados.")
        if self.motion_comparator and hasattr(self.motion_comparator, "__del__"):
            self.motion_comparator.__del__()
            logger.info("MotionComparator recursos liberados.")
