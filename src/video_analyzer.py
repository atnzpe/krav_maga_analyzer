# src/video_analyzer.py

import cv2
import os
import tempfile
import threading
import logging
import numpy as np
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

        self.aluno_landmarks = []
        self.mestre_landmarks = []
        self.comparison_results = []
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
            else:
                self.video_mestre_path = video_path
                self.cap_mestre = cv2.VideoCapture(video_path)

            logger.info(
                f"Vídeo {'aluno' if is_aluno else 'mestre'} carregado de: {video_path}"
            )
            return video_path
        except Exception as e:
            logger.error(f"Erro ao carregar vídeo de bytes: {e}", exc_info=True)
            raise

    def analyze_and_compare(self, post_analysis_callback, progress_callback=None):
        """
        Inicia a análise em uma nova thread e chama callbacks para progresso e finalização.
        """
        if self.is_processing:
            logger.info("Análise já em andamento.")
            return

        def target():
            self._run_analysis_thread(progress_callback)
            post_analysis_callback()

        self.is_processing = True
        logger.info("Iniciando a thread de processamento de vídeo.")
        self.processing_thread = threading.Thread(target=target, daemon=True)
        self.processing_thread.start()

    def _run_analysis_thread(self, progress_callback=None):
        """
        Método executado na thread. Processa os vídeos, compara os frames e reporta o progresso.
        """
        try:
            logger.info("Thread de análise iniciada.")

            self.aluno_landmarks.clear()
            self.mestre_landmarks.clear()
            self.processed_frames_aluno.clear()
            self.processed_frames_mestre.clear()
            self.comparison_results.clear()

            num_frames = min(
                int(self.cap_aluno.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(self.cap_mestre.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            logger.info(f"Iniciando processamento e comparação de {num_frames} frames.")

            for i in range(num_frames):
                ret_aluno, frame_aluno = self.cap_aluno.read()
                ret_mestre, frame_mestre = self.cap_mestre.read()

                if not ret_aluno or not ret_mestre:
                    break

                results_aluno, annotated_aluno = self.pose_estimator.estimate_pose(
                    frame_aluno
                )
                results_mestre, annotated_mestre = self.pose_estimator.estimate_pose(
                    frame_mestre
                )

                self.processed_frames_aluno.append(annotated_aluno)
                self.processed_frames_mestre.append(annotated_mestre)

                lm_aluno = self.pose_estimator.get_landmarks_as_list(
                    results_aluno.pose_landmarks
                )
                lm_mestre = self.pose_estimator.get_landmarks_as_list(
                    results_mestre.pose_landmarks
                )
                self.aluno_landmarks.append(lm_aluno)
                self.mestre_landmarks.append(lm_mestre)

                score, feedback, _ = self.motion_comparator.compare_poses(
                    lm_aluno, lm_mestre
                )
                self.comparison_results.append({"score": score, "feedback": feedback})

                # --- LÓGICA DE CALLBACK DE PROGRESSO ---
                if progress_callback:
                    percent_complete = (i + 1) / num_frames
                    progress_callback(percent_complete)

        except Exception as e:
            logger.error(f"Erro na thread de análise: {e}", exc_info=True)
        finally:
            self.is_processing = False
            if self.cap_aluno:
                self.cap_aluno.release()
            if self.cap_mestre:
                self.cap_mestre.release()
            logger.info("Thread de análise finalizada.")

    def get_best_frames(self):
        """
        Encontra e retorna os frames (aluno e mestre) correspondentes à maior pontuação.
        """
        logger.info("Buscando os frames com a melhor pontuação para o relatório.")
        if not self.comparison_results:
            return None, None

        try:
            scores = [res["score"] for res in self.comparison_results]
            best_frame_index = np.argmax(scores)

            cap_aluno = cv2.VideoCapture(self.video_aluno_path)
            cap_mestre = cv2.VideoCapture(self.video_mestre_path)

            cap_aluno.set(cv2.CAP_PROP_POS_FRAMES, best_frame_index)
            cap_mestre.set(cv2.CAP_PROP_POS_FRAMES, best_frame_index)

            ret_aluno, frame_aluno = cap_aluno.read()
            ret_mestre, frame_mestre = cap_mestre.read()

            cap_aluno.release()
            cap_mestre.release()

            if ret_aluno and ret_mestre:
                return frame_aluno, frame_mestre
            else:
                return None, None
        except Exception as e:
            logger.error(f"Erro ao recuperar os melhores frames: {e}")
            return None, None

    def get_worst_frames(self):
        """
        Encontra e retorna os frames (aluno e mestre) correspondentes à menor pontuação.
        """
        logger.info("Buscando os frames com a pior pontuação para o relatório.")
        if not self.comparison_results:
            return None, None

        try:
            scores = [res["score"] for res in self.comparison_results]
            worst_frame_index = np.argmin(scores)

            cap_aluno = cv2.VideoCapture(self.video_aluno_path)
            cap_mestre = cv2.VideoCapture(self.video_mestre_path)

            cap_aluno.set(cv2.CAP_PROP_POS_FRAMES, worst_frame_index)
            cap_mestre.set(cv2.CAP_PROP_POS_FRAMES, worst_frame_index)

            ret_aluno, frame_aluno = cap_aluno.read()
            ret_mestre, frame_mestre = cap_mestre.read()

            cap_aluno.release()
            cap_mestre.release()

            if ret_aluno and ret_mestre:
                return frame_aluno, frame_mestre
            else:
                return None, None
        except Exception as e:
            logger.error(f"Erro ao recuperar os piores frames: {e}")
            return None, None

    def __del__(self):
        """Limpa os arquivos temporários."""
        logger.info("Destruindo VideoAnalyzer e limpando arquivos.")
        try:
            if self.video_aluno_path and os.path.exists(self.video_aluno_path):
                os.remove(self.video_aluno_path)
            if self.video_mestre_path and os.path.exists(self.video_mestre_path):
                os.remove(self.video_mestre_path)
        except Exception as e:
            logger.error(f"Erro ao limpar arquivos temporários: {e}")
