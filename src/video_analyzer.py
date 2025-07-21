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

# INÍCIO DO REGISTRO DE LOG: Configurando o logger para este módulo.
logger = get_logger(__name__)


class VideoAnalyzer:
    """
    Classe responsável por analisar vídeos, detectar poses, comparar movimentos
    e fornecer feedback.
    (Nenhuma alteração na docstring da classe)
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
        self.is_processing = False
        self.processing_thread = None
        self.current_frame_aluno = None
        self.current_frame_mestre = None

        # --- NOVA ESTRUTURA DE DADOS ---
        # Armazenará o dicionário de diferenças de ângulo para cada frame.
        self.angle_diffs_history = []

        logger.info("Variáveis de estado do VideoAnalyzer configuradas.")

    def load_video_from_bytes(self, video_bytes: bytes, is_aluno: bool):
        """
        Carrega um vídeo a partir de bytes.
        (Nenhuma alteração neste método)
        """
        try:
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
        (Nenhuma alteração neste método)
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
            results, annotated_frame = self.pose_estimator.estimate_pose(frame)
            if results.pose_landmarks:
                # Transforma os landmarks em uma lista de dicionários para consistência.
                landmarks_as_list = self.pose_estimator.get_landmarks_as_list(
                    results.pose_landmarks
                )
                landmarks_list.append(landmarks_as_list)
            else:
                landmarks_list.append(None)
            if is_aluno:
                self.current_frame_aluno = annotated_frame
            else:
                self.current_frame_mestre = annotated_frame
            frame_count += 1
        cap.release()
        logger.info(f"Processamento do vídeo {video_path} concluído.")
        return landmarks_list

    def analyze_and_compare(self):
        """
        Inicia o processo de análise e comparação.
        (Nenhuma alteração neste método)
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
        Método executado em uma thread para processar e comparar os vídeos.
        """
        try:
            logger.info("Thread de análise iniciada.")
            # Processa ambos os vídeos para obter as listas de landmarks.
            aluno_landmarks_history = self.process_video(
                self.video_aluno_path, is_aluno=True
            )
            mestre_landmarks_history = self.process_video(
                self.video_mestre_path, is_aluno=False
            )

            if not aluno_landmarks_history or not mestre_landmarks_history:
                logger.warning(
                    "Não foi possível extrair landmarks de um ou ambos os vídeos."
                )
                self.is_processing = False
                return

            logger.info("Iniciando comparação de movimentos frame a frame...")

            # Limpa resultados de análises anteriores.
            self.comparison_results.clear()
            self.angle_diffs_history.clear()

            # Itera pelo menor número de frames entre os dois vídeos.
            num_frames = min(
                len(aluno_landmarks_history), len(mestre_landmarks_history)
            )
            for i in range(num_frames):
                aluno_lm = aluno_landmarks_history[i]
                mestre_lm = mestre_landmarks_history[i]

                # --- LÓGICA MODIFICADA ---
                # Agora captura os três valores retornados pelo comparador.
                score, feedback, angle_diffs = self.motion_comparator.compare_poses(
                    aluno_lm, mestre_lm
                )

                # Armazena todos os resultados.
                self.comparison_results.append({"score": score, "feedback": feedback})
                self.angle_diffs_history.append(angle_diffs)

            logger.info(f"Comparação de {num_frames} frames concluída.")

        except Exception as e:
            logger.error(f"Erro durante a análise na thread: {e}", exc_info=True)
        finally:
            self.is_processing = False
            logger.info("Thread de análise finalizada.")

    # get_current_annotated_frames, get_comparison_results, __del__: NENHUMA MUDANÇA
    def get_current_annotated_frames(self):
        return self.current_frame_aluno, self.current_frame_mestre

    def get_comparison_results(self):
        return self.comparison_results

    def __del__(self):
        logger.info("Destruindo VideoAnalyzer e liberando recursos.")
        if self.cap_aluno and self.cap_aluno.isOpened():
            self.cap_aluno.release()
        if self.cap_mestre and self.cap_mestre.isOpened():
            self.cap_mestre.release()
        if self.video_aluno_path and os.path.exists(self.video_aluno_path):
            os.remove(self.video_aluno_path)
        if self.video_mestre_path and os.path.exists(self.video_mestre_path):
            os.remove(self.video_mestre_path)
        if hasattr(self.pose_estimator, "__del__"):
            self.pose_estimator.__del__()
        logger.info("Recursos do VideoAnalyzer liberados.")
