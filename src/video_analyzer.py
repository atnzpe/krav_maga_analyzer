# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------------
#  Krav Maga Motion Analyzer
#  version 1.2.0
#  Copyright (C) 2024,
#  開発者名 [Sujeito Programador]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------
import cv2
import logging
import numpy as np  # Importado para usar argmax.
from threading import Thread
from src.pose_estimator import PoseEstimator
from src.motion_comparator import MotionComparator

# --------------------------------------------------------------------------------------------------
# Configuração do Logging
# --------------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------------------------------------------------------
# Classe de Análise de Vídeo
# --------------------------------------------------------------------------------------------------


class VideoAnalyzer:
    """
    Classe responsável por analisar os vídeos, extrair as poses e comparar os movimentos.
    """

    def __init__(self, video_path_aluno, video_path_mestre):
        """
        Inicializador da classe VideoAnalyzer.
        (Nenhuma mudança no __init__)
        """
        self.video_path_aluno = video_path_aluno
        self.video_path_mestre = video_path_mestre
        self.pose_estimator = PoseEstimator()
        self.motion_comparator = MotionComparator()
        self.frames_aluno = []
        self.frames_mestre = []
        self.landmarks_aluno = []
        self.landmarks_mestre = []
        self.scores = []
        self.feedbacks = []
        self.is_processing = False
        self.processing_thread = None
        logging.info(
            f"VideoAnalyzer inicializado para '{video_path_aluno}' e '{video_path_mestre}'."
        )

    def _process_videos(self):
        """
        Método privado que executa o processamento dos vídeos.
        (Nenhuma mudança neste método)
        """
        self.is_processing = True
        logging.info("Iniciando o processamento dos vídeos.")
        cap_aluno = cv2.VideoCapture(self.video_path_aluno)
        cap_mestre = cv2.VideoCapture(self.video_path_mestre)

        try:
            while cap_aluno.isOpened() and cap_mestre.isOpened():
                ret_aluno, frame_aluno = cap_aluno.read()
                ret_mestre, frame_mestre = cap_mestre.read()
                if not ret_aluno or not ret_mestre:
                    logging.info("Um dos vídeos chegou ao fim.")
                    break

                frame_aluno_processed, landmarks_aluno = (
                    self.pose_estimator.estimate_pose(frame_aluno)
                )
                frame_mestre_processed, landmarks_mestre = (
                    self.pose_estimator.estimate_pose(frame_mestre)
                )

                self.frames_aluno.append(frame_aluno_processed)
                self.frames_mestre.append(frame_mestre_processed)
                self.landmarks_aluno.append(landmarks_aluno)
                self.landmarks_mestre.append(landmarks_mestre)

                score, feedback = self.motion_comparator.compare_poses(
                    landmarks_aluno, landmarks_mestre
                )
                self.scores.append(score)
                self.feedbacks.append(feedback)
        finally:
            cap_aluno.release()
            cap_mestre.release()
            self.is_processing = False
            logging.info("Processamento dos vídeos concluído.")

    def start_analysis(self):
        """
        Inicia a análise dos vídeos em uma nova thread.
        (Nenhuma mudança neste método)
        """
        if self.processing_thread is None or not self.processing_thread.is_alive():
            logging.info("Iniciando a thread de análise.")
            self.processing_thread = Thread(target=self._process_videos)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def get_frame_count(self):
        """
        Retorna o número total de frames processados.
        (Nenhuma mudança neste método)
        """
        return len(self.frames_aluno)

    def get_data_at_frame(self, frame_index):
        """
        Retorna todos os dados para um índice de frame específico.
        (Nenhuma mudança neste método)
        """
        if 0 <= frame_index < self.get_frame_count():
            frame_aluno = self.frames_aluno[frame_index]
            frame_mestre = self.frames_mestre[frame_index]
            score = self.scores[frame_index]
            feedback = self.feedbacks[frame_index]
            logging.debug(f"Recuperando dados para o frame {frame_index}.")
            return frame_aluno, frame_mestre, score, feedback

        logging.warning(f"Índice de frame {frame_index} fora do intervalo.")
        return None, None, 0.0, ""

    # --- NOVO MÉTODO PARA O RELATÓRIO ---
    def get_best_frames(self):
        """
        Encontra e retorna os frames (aluno e mestre) correspondentes à maior pontuação.
        Esses frames serão usados no relatório PDF.

        Returns:
            tuple: (frame_aluno, frame_mestre) ou (None, None) se não houver dados.
        """
        logging.info("Buscando os frames com a melhor pontuação para o relatório.")
        if not self.scores:
            logging.warning("Não há pontuações para determinar os melhores frames.")
            return None, None

        try:
            # np.argmax encontra o índice do valor máximo na lista de pontuações.
            best_frame_index = np.argmax(self.scores)
            logging.info(
                f"Melhor pontuação encontrada no frame de índice: {best_frame_index}."
            )

            # Recupera os frames originais (sem a sobreposição do esqueleto) para o relatório.
            # Para isso, precisamos reler os vídeos, o que é um trade-off pela simplicidade.
            # Uma otimização futura poderia ser armazenar os frames originais.
            cap_aluno = cv2.VideoCapture(self.video_path_aluno)
            cap_mestre = cv2.VideoCapture(self.video_path_mestre)

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
            logging.error(f"Erro ao recuperar os melhores frames: {e}")
            return None, None
