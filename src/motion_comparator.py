# src/motion_comparator.py

import logging
import numpy as np
import mediapipe as mp
from src.utils import get_logger, calculate_angle
from sklearn.metrics.pairwise import cosine_similarity

# INÍCIO DO REGISTRO DE LOG: Configurando o logger para este módulo.
logger = get_logger(__name__)


class MotionComparator:
    """
    Classe responsável por comparar os movimentos do aluno com os do mestre.
    (Nenhuma alteração na docstring da classe)
    """

    def __init__(self):
        """
        Inicializa o MotionComparator.
        (Nenhuma alteração no método __init__)
        """
        logger.info("Inicializando MotionComparator...")
        self.KEY_ANGLES = {
            "LEFT_ELBOW_ANGLE": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
            "RIGHT_ELBOW_ANGLE": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            "LEFT_SHOULDER_ANGLE": ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
            "RIGHT_SHOULDER_ANGLE": ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
            "LEFT_KNEE_ANGLE": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
            "RIGHT_KNEE_ANGLE": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
            "LEFT_HIP_ANGLE": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
            "RIGHT_HIP_ANGLE": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
        }
        self.landmark_indices = {
            "NOSE": mp.solutions.pose.PoseLandmark.NOSE.value,
            "LEFT_SHOULDER": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
            "RIGHT_SHOULDER": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
            "LEFT_ELBOW": mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
            "RIGHT_ELBOW": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
            "LEFT_WRIST": mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
            "RIGHT_WRIST": mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
            "LEFT_HIP": mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
            "RIGHT_HIP": mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
            "LEFT_KNEE": mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
            "RIGHT_KNEE": mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
            "LEFT_ANKLE": mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
            "RIGHT_ANKLE": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,
        }
        logger.info(
            f"Ângulos chave para comparação definidos: {list(self.KEY_ANGLES.keys())}"
        )

    def _get_landmark_coords(self, landmarks_data: list, name: str) -> dict:
        """
        Retorna as coordenadas de um landmark pelo nome.
        (Nenhuma alteração neste método)
        """
        idx = self.landmark_indices.get(name)
        if idx is None or not landmarks_data or idx >= len(landmarks_data):
            raise ValueError(
                f"Landmark '{name}' (índice {idx}) não encontrado ou dados inválidos."
            )
        return landmarks_data[idx]

    # --- MÉTODO MODIFICADO ---
    def compare_poses(self, aluno_landmarks, mestre_landmarks):
        """
        Compara as poses do aluno e do mestre frame a frame.

        Args:
            aluno_landmarks (list): Lista de landmarks do aluno para um frame.
            mestre_landmarks (list): Lista de landmarks do mestre para um frame.

        Returns:
            tuple: Uma tupla contendo (pontuação, feedback, dicionário_de_diferenças).
        """
        logger.debug("Iniciando comparação de pose para um único frame.")
        if not aluno_landmarks or not mestre_landmarks:
            logger.warning(
                "Comparação pulada: landmarks de aluno ou mestre ausentes para este frame."
            )
            return 0.0, "Aguardando pose...", {}

        angles_aluno = {}
        angles_mestre = {}
        angle_diffs = {}

        # Calcula os ângulos para aluno e mestre e a diferença entre eles.
        for angle_name, (p1_name, p2_name, p3_name) in self.KEY_ANGLES.items():
            try:
                # Extrai os landmarks necessários
                aluno_p1 = self._get_landmark_coords(aluno_landmarks, p1_name)
                aluno_p2 = self._get_landmark_coords(aluno_landmarks, p2_name)
                aluno_p3 = self._get_landmark_coords(aluno_landmarks, p3_name)
                mestre_p1 = self._get_landmark_coords(mestre_landmarks, p1_name)
                mestre_p2 = self._get_landmark_coords(mestre_landmarks, p2_name)
                mestre_p3 = self._get_landmark_coords(mestre_landmarks, p3_name)

                # Calcula o ângulo para cada um
                angle_aluno = calculate_angle(aluno_p1, aluno_p2, aluno_p3)
                angle_mestre = calculate_angle(mestre_p1, mestre_p2, mestre_p3)

                angles_aluno[angle_name] = angle_aluno
                angles_mestre[angle_name] = angle_mestre
                angle_diffs[angle_name] = abs(angle_aluno - angle_mestre)

            except ValueError as e:
                logger.warning(f"Não foi possível calcular o ângulo {angle_name}: {e}")
                angles_aluno[angle_name] = 0
                angles_mestre[angle_name] = 0
                angle_diffs[angle_name] = 0

        # Converte os dicionários de ângulos em vetores para o cálculo da similaridade
        labels = sorted(angles_aluno.keys())
        vec_aluno = np.array([angles_aluno[k] for k in labels]).reshape(1, -1)
        vec_mestre = np.array([angles_mestre[k] for k in labels]).reshape(1, -1)

        # Calcula a similaridade e a pontuação
        similarity = cosine_similarity(vec_aluno, vec_mestre)[0, 0]
        score = (similarity + 1) / 2 * 100
        logger.info(f"Pontuação de similaridade do frame: {score:.2f}%")

        # Gera o feedback textual
        feedback = self._generate_feedback(angle_diffs, angles_aluno, angles_mestre)

        # RETORNO MODIFICADO: Agora inclui o dicionário de diferenças.
        return score, feedback, angle_diffs

    def _generate_feedback(self, angle_diffs, angles_aluno, angles_mestre):
        """
        Gera um feedback textual baseado na maior diferença angular.
        (Nenhuma alteração neste método)
        """
        if not angle_diffs:
            return "Movimento Perfeito!"

        max_diff_label = max(angle_diffs, key=angle_diffs.get)
        max_diff_value = angle_diffs[max_diff_label]

        if max_diff_value < 10:
            return "Excelente movimento!"

        aluno_angle = angles_aluno[max_diff_label]
        mestre_angle = angles_mestre[max_diff_label]
        readable_label = max_diff_label.replace("_", " ").title()

        feedback = (
            f"Aumente o ângulo do {readable_label}"
            if aluno_angle < mestre_angle
            else f"Diminua o ângulo do {readable_label}"
        )
        logger.info(f"Feedback gerado: '{feedback}' para {readable_label}")
        return feedback
