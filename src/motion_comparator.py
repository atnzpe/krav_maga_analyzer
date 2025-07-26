# src/motion_comparator.py

import logging
import numpy as np
import mediapipe as mp
from src.utils import get_logger, calculate_angle

logger = get_logger(__name__)


class MotionComparator:
    """
    Compara os movimentos do aluno com os do mestre com lógica aprimorada e feedback em português.
    """

    def __init__(self):
        logger.info("Inicializando MotionComparator.")
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
        # --- DICIONÁRIO DE TRADUÇÃO ---
        self.readable_angle_names = {
            "LEFT_ELBOW_ANGLE": "Cotovelo Esquerdo",
            "RIGHT_ELBOW_ANGLE": "Cotovelo Direito",
            "LEFT_SHOULDER_ANGLE": "Ombro Esquerdo",
            "RIGHT_SHOULDER_ANGLE": "Ombro Direito",
            "LEFT_KNEE_ANGLE": "Joelho Esquerdo",
            "RIGHT_KNEE_ANGLE": "Joelho Direito",
            "LEFT_HIP_ANGLE": "Quadril Esquerdo",
            "RIGHT_HIP_ANGLE": "Quadril Direito",
        }
        logger.info(f"Ângulos chave definidos: {list(self.KEY_ANGLES.keys())}")

    def _get_landmark_coords(self, landmarks_data: list, name: str) -> dict:
        idx = self.landmark_indices.get(name)
        if idx is None or not landmarks_data or idx >= len(landmarks_data):
            raise ValueError(f"Landmark '{name}' não encontrado.")
        return landmarks_data[idx]

    def compare_poses(self, aluno_landmarks, mestre_landmarks):
        if not aluno_landmarks or not mestre_landmarks:
            return 0.0, "Aguardando pose...", {}

        angles_aluno, angles_mestre, angle_diffs = {}, {}, {}
        for angle_name, (p1, p2, p3) in self.KEY_ANGLES.items():
            try:
                aluno_p1, aluno_p2, aluno_p3 = (
                    self._get_landmark_coords(aluno_landmarks, p) for p in (p1, p2, p3)
                )
                mestre_p1, mestre_p2, mestre_p3 = (
                    self._get_landmark_coords(mestre_landmarks, p) for p in (p1, p2, p3)
                )

                angles_aluno[angle_name] = calculate_angle(aluno_p1, aluno_p2, aluno_p3)
                angles_mestre[angle_name] = calculate_angle(
                    mestre_p1, mestre_p2, mestre_p3
                )
                angle_diffs[angle_name] = abs(
                    angles_aluno[angle_name] - angles_mestre[angle_name]
                )
            except ValueError:
                (
                    angles_aluno[angle_name],
                    angles_mestre[angle_name],
                    angle_diffs[angle_name],
                ) = (0, 0, 180)

        # --- LÓGICA DE PONTUAÇÃO REFINADA ---
        # A pontuação agora é baseada na média da similaridade de cada ângulo.
        # Similaridade de 100% significa 0 graus de diferença. 0% significa 180 graus.
        similarities = [max(0, 1 - (diff / 180)) for diff in angle_diffs.values()]
        score = np.mean(similarities) * 100 if similarities else 0

        feedback = self._generate_feedback(angle_diffs, angles_aluno, angles_mestre)
        return score, feedback, angle_diffs

    def _generate_feedback(self, angle_diffs, angles_aluno, angles_mestre):
        """Gera feedback consolidado para todos os ângulos com erros significativos."""
        if not angle_diffs:
            return "Movimento Perfeito!"

        error_threshold = 15.0  # Limite de 15 graus para considerar um erro
        errors = []

        for angle_name, diff in angle_diffs.items():
            if diff > error_threshold:
                aluno_angle = angles_aluno[angle_name]
                mestre_angle = angles_mestre[angle_name]
                readable_label = self.readable_angle_names.get(angle_name, angle_name)

                action = "Aumente" if aluno_angle < mestre_angle else "Diminua"
                errors.append(f"{action} o ângulo do {readable_label}")

        if not errors:
            return "Excelente movimento!"

        # Concatena todos os erros em uma única mensagem
        feedback = ". ".join(errors)
        logger.info(f"Feedback gerado: '{feedback}'")
        return feedback
