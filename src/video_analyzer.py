# src/video_analyzer.py

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tempfile
import threading
import io

from src.utils import get_logger
from src.pose_estimator import PoseEstimator
from src.motion_comparator import MotionComparator

logger = get_logger(__name__)


class VideoAnalyzer:
    """
    Classe responsável por analisar vídeos, detectar poses, comparar movimentos
    e fornecer feedback.
    """

    def __init__(self):
        """
        Inicializa o VideoAnalyzer.
        """
        logger.info("Inicializando VideoAnalyzer...")
        self.pose_estimator = PoseEstimator()
        self.motion_comparator = MotionComparator()

        self.cap_aluno = None
        self.cap_mestre = None
        self.video_aluno_path = None
        self.video_mestre_path = None

        # --- ARMAZENAMENTO DE DADOS DA ANÁLISE ---
        self.aluno_landmarks = []
        self.mestre_landmarks = []
        self.comparison_results = []

        # --- MUDANÇA: Adicionado armazenamento para os frames processados ---
        # Guardaremos todas as imagens aqui para acesso rápido pelo slider e botões.
        self.processed_frames_aluno = []
        self.processed_frames_mestre = []

        self.is_processing = False
        self.processing_thread = None
        logger.info("Variáveis de estado do VideoAnalyzer configuradas.")

    def load_video_from_bytes(self, video_bytes: bytes, is_aluno: bool):
        """
        Carrega um vídeo a partir de bytes e o salva temporariamente para processamento.
        """
        logger.info(
            f"Carregando vídeo a partir de bytes para {'aluno' if is_aluno else 'mestre'}."
        )
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(video_bytes)
            temp_file.close()
            video_path = temp_file.name

            if is_aluno:
                self.video_aluno_path = video_path
                self.cap_aluno = cv2.VideoCapture(video_path)
                if not self.cap_aluno.isOpened():
                    raise Exception(
                        f"Não foi possível abrir o vídeo do aluno em {video_path}"
                    )
            else:
                self.video_mestre_path = video_path
                self.cap_mestre = cv2.VideoCapture(video_path)
                if not self.cap_mestre.isOpened():
                    raise Exception(
                        f"Não foi possível abrir o vídeo do mestre em {video_path}"
                    )

            logger.info(
                f"Vídeo {'aluno' if is_aluno else 'mestre'} carregado com sucesso do caminho: {video_path}"
            )
            return video_path
        except Exception as e:
            logger.error(f"Erro ao carregar vídeo de bytes: {e}")
            raise

    def analyze_and_compare(self):
        """
        Inicia o processo de análise e comparação em uma nova thread.
        """
        if not self.video_aluno_path or not self.video_mestre_path:
            logger.warning("Caminhos dos vídeos não definidos. Análise abortada.")
            return False
        if self.is_processing:
            logger.info("Análise já está em andamento.")
            return False

        self.is_processing = True
        logger.info("Iniciando a thread de processamento de vídeo.")
        self.processing_thread = threading.Thread(target=self._run_analysis_thread)
        self.processing_thread.start()
        return True

    def _run_analysis_thread(self):
        """
        Método principal executado na thread. Processa os vídeos e compara os frames.
        """
        try:
            logger.info("Thread de análise iniciada.")

            # Limpa dados de análises anteriores
            self.aluno_landmarks.clear()
            self.mestre_landmarks.clear()
            self.processed_frames_aluno.clear()
            self.processed_frames_mestre.clear()
            self.comparison_results.clear()

            # Garante que os captures estejam abertos
            if not self.cap_aluno or not self.cap_mestre:
                logger.error("Objetos de captura de vídeo não estão inicializados.")
                return

            num_frames = min(
                int(self.cap_aluno.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(self.cap_mestre.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            logger.info(f"Iniciando processamento e comparação de {num_frames} frames.")

            for i in range(num_frames):
                ret_aluno, frame_aluno = self.cap_aluno.read()
                ret_mestre, frame_mestre = self.cap_mestre.read()

                if not ret_aluno or not ret_mestre:
                    logger.warning(
                        f"Não foi possível ler o frame {i}. Interrompendo a análise."
                    )
                    break

                # Estima a pose para ambos os frames
                results_aluno, annotated_aluno = self.pose_estimator.estimate_pose(
                    frame_aluno
                )
                results_mestre, annotated_mestre = self.pose_estimator.estimate_pose(
                    frame_mestre
                )

                # Armazena os frames anotados
                self.processed_frames_aluno.append(annotated_aluno)
                self.processed_frames_mestre.append(annotated_mestre)

                # Armazena os landmarks
                lm_aluno = self.pose_estimator.get_landmarks_as_list(
                    results_aluno.pose_landmarks
                )
                lm_mestre = self.pose_estimator.get_landmarks_as_list(
                    results_mestre.pose_landmarks
                )
                self.aluno_landmarks.append(lm_aluno)
                self.mestre_landmarks.append(lm_mestre)

                # Compara as poses e armazena o resultado
                score, feedback, _ = self.motion_comparator.compare_poses(
                    lm_aluno, lm_mestre
                )
                self.comparison_results.append({"score": score, "feedback": feedback})

                if i % 50 == 0:
                    logger.info(f"Processados {i}/{num_frames} frames...")

        except Exception as e:
            logger.error(
                f"Erro catastrófico durante a análise na thread: {e}", exc_info=True
            )
        finally:
            self.is_processing = False
            # Libera os captures após o uso
            if self.cap_aluno:
                self.cap_aluno.release()
            if self.cap_mestre:
                self.cap_mestre.release()
            logger.info("Thread de análise finalizada e recursos de vídeo liberados.")

    def __del__(self):
        """
        Limpa os recursos quando o objeto VideoAnalyzer é destruído.
        """
        logger.info("Destruindo VideoAnalyzer e limpando arquivos temporários.")
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
