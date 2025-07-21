# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------------
#  Krav Maga Motion Analyzer
#  version 1.1.0
#  Copyright (C) 2024,
#  開発者名 [Sujeito Programador]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------
import cv2  # Biblioteca OpenCV para manipulação de vídeo.
import logging  # Biblioteca para registrar logs de eventos.
from threading import Thread  # Para executar a análise em uma thread separada e não travar a UI.
from src.pose_estimator import PoseEstimator  # Importa o nosso detector de pose.
from src.motion_comparator import MotionComparator  # Importa nosso comparador de movimento.

# --------------------------------------------------------------------------------------------------
# Configuração do Logging
# --------------------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------------------------------------
# Classe de Análise de Vídeo
# --------------------------------------------------------------------------------------------------

class VideoAnalyzer:
    """
    Classe responsável por analisar os vídeos, extrair as poses e comparar os movimentos.
    Orquestra o trabalho do PoseEstimator e do MotionComparator.
    """

    def __init__(self, video_path_aluno, video_path_mestre):
        """
        Inicializador da classe VideoAnalyzer.

        Args:
            video_path_aluno (str): Caminho para o vídeo do aluno.
            video_path_mestre (str): Caminho para o vídeo do mestre.
        """
        # Caminhos para os arquivos de vídeo.
        self.video_path_aluno = video_path_aluno
        self.video_path_mestre = video_path_mestre
        
        # Instanciação das nossas classes de lógica.
        self.pose_estimator = PoseEstimator()
        self.motion_comparator = MotionComparator()
        
        # Estruturas para armazenar os dados processados.
        self.frames_aluno = []  # Armazena os frames (imagens) do vídeo do aluno.
        self.frames_mestre = [] # Armazena os frames do vídeo do mestre.
        self.landmarks_aluno = [] # Armazena os landmarks de cada frame do aluno.
        self.landmarks_mestre = [] # Armazena os landmarks de cada frame do mestre.
        self.scores = [] # NOVA: Armazena a pontuação de similaridade de cada frame.
        self.feedbacks = [] # NOVA: Armazena o feedback textual de cada frame.

        self.is_processing = False  # Flag para controlar o estado do processamento.
        self.processing_thread = None # A thread que fará o trabalho pesado.
        logging.info(f"VideoAnalyzer inicializado para '{video_path_aluno}' e '{video_path_mestre}'.")

    def _process_videos(self):
        """
        Método privado que executa o processamento dos vídeos.
        Este método é executado em uma thread separada.
        """
        self.is_processing = True
        logging.info("Iniciando o processamento dos vídeos.")

        # Abre os arquivos de vídeo usando OpenCV.
        cap_aluno = cv2.VideoCapture(self.video_path_aluno)
        cap_mestre = cv2.VideoCapture(self.video_path_mestre)

        try:
            # Loop principal: continua enquanto ambos os vídeos tiverem frames.
            while cap_aluno.isOpened() and cap_mestre.isOpened():
                # Lê um frame de cada vídeo.
                ret_aluno, frame_aluno = cap_aluno.read()
                ret_mestre, frame_mestre = cap_mestre.read()

                # Se algum dos vídeos terminar, encerra o loop.
                if not ret_aluno or not ret_mestre:
                    logging.info("Um dos vídeos chegou ao fim.")
                    break

                # Estima a pose para cada frame.
                frame_aluno_processed, landmarks_aluno = self.pose_estimator.estimate_pose(frame_aluno)
                frame_mestre_processed, landmarks_mestre = self.pose_estimator.estimate_pose(frame_mestre)
                
                # Armazena os resultados.
                self.frames_aluno.append(frame_aluno_processed)
                self.frames_mestre.append(frame_mestre_processed)
                self.landmarks_aluno.append(landmarks_aluno)
                self.landmarks_mestre.append(landmarks_mestre)

                # --- NOVA FUNCIONALIDADE ---
                # Compara as poses e armazena a pontuação e o feedback.
                score, feedback = self.motion_comparator.compare_poses(landmarks_aluno, landmarks_mestre)
                self.scores.append(score)
                self.feedbacks.append(feedback)
                # -------------------------

        finally:
            # Garante que os arquivos de vídeo sejam liberados, mesmo se ocorrer um erro.
            cap_aluno.release()
            cap_mestre.release()
            self.is_processing = False
            logging.info("Processamento dos vídeos concluído.")

    def start_analysis(self):
        """
        Inicia a análise dos vídeos em uma nova thread.
        """
        if self.processing_thread is None or not self.processing_thread.is_alive():
            logging.info("Iniciando a thread de análise.")
            # Cria e inicia a thread para não bloquear a UI.
            self.processing_thread = Thread(target=self._process_videos)
            self.processing_thread.daemon = True # Permite que a aplicação feche mesmo se a thread estiver rodando.
            self.processing_thread.start()

    def get_frame_count(self):
        """
        Retorna o número total de frames processados (o menor entre os dois vídeos).
        
        Returns:
            int: O número de frames.
        """
        return len(self.frames_aluno)

    def get_data_at_frame(self, frame_index):
        """
        Retorna todos os dados para um índice de frame específico.

        Args:
            frame_index (int): O índice do frame desejado.

        Returns:
            tuple: Uma tupla contendo (frame_aluno, frame_mestre, score, feedback).
                   Retorna (None, None, 0.0, "") se o índice for inválido.
        """
        if 0 <= frame_index < self.get_frame_count():
            # Acessa os dados armazenados nas listas.
            frame_aluno = self.frames_aluno[frame_index]
            frame_mestre = self.frames_mestre[frame_index]
            score = self.scores[frame_index]
            feedback = self.feedbacks[frame_index]
            logging.debug(f"Recuperando dados para o frame {frame_index}.")
            return frame_aluno, frame_mestre, score, feedback
        
        logging.warning(f"Índice de frame {frame_index} fora do intervalo.")
        return None, None, 0.0, ""