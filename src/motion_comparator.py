# src/motion_comparator.py

import logging
import numpy as np
from src.utils import setup_logging, calculate_angle  # Importa as funções utilitárias

logger = setup_logging()  # Inicializa o logger para este módulo


class MotionComparator:
    """
    Classe responsável por comparar os movimentos do aluno com os do mestre
    com base nos dados de landmarks de pose.

    Esta classe utiliza os landmarks 3D fornecidos pelo MediaPipe para calcular
    e comparar ângulos corporais chave, fornecendo feedback sobre a execução.
    """

    def __init__(self):
        """
        Inicializa o MotionComparator.
        Define os landmarks e as conexões que serão usados para calcular os ângulos chave
        relevantes para as técnicas de Krav Magá.
        """
        logger.info("Inicializando MotionComparator.")

        # Dicionário que mapeia nomes de partes do corpo para os índices dos landmarks do MediaPipe.
        # Os nomes dos landmarks são definidos pelo MediaPipe (ex: mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        # É crucial usar os índices corretos conforme a documentação do MediaPipe.
        # Estes são os índices numéricos das 33 landmarks do MediaPipe.
        # Para facilitar a leitura, usamos os nomes das enumerações do MediaPipe, mas ao usar
        # o landmarks_data (que é uma lista), acessaremos por índice.
        # Você pode consultar a lista completa em:
        # https://google.github.io/mediapipe/solutions/pose.html#pose-landmarks
        self.LANDMARKS = {
            "NOSE": 0,
            "LEFT_EYE_INNER": 1,
            "LEFT_EYE": 2,
            "LEFT_EYE_OUTER": 3,
            "RIGHT_EYE_INNER": 4,
            "RIGHT_EYE": 5,
            "RIGHT_EYE_OUTER": 6,
            "LEFT_EAR": 7,
            "RIGHT_EAR": 8,
            "MOUTH_LEFT": 9,
            "MOUTH_RIGHT": 10,
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_ELBOW": 13,
            "RIGHT_ELBOW": 14,
            "LEFT_WRIST": 15,
            "RIGHT_WRIST": 16,
            "LEFT_PINKY": 17,
            "RIGHT_PINKY": 18,
            "LEFT_INDEX": 19,
            "RIGHT_INDEX": 20,
            "LEFT_THUMB": 21,
            "RIGHT_THUMB": 22,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_KNEE": 25,
            "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28,
            "LEFT_HEEL": 29,
            "RIGHT_HEEL": 30,
            "LEFT_FOOT_INDEX": 31,
            "RIGHT_FOOT_INDEX": 32,
        }

        # Definição dos ângulos chave a serem monitorados para o Krav Magá.
        # Cada tupla contém os nomes dos três landmarks que formam o ângulo (vértice no meio).
        # Estes ângulos são fundamentais para avaliar a forma e a execução dos movimentos.
        self.KEY_ANGLES = {
            "LEFT_ELBOW_ANGLE": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
            "RIGHT_ELBOW_ANGLE": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            "LEFT_SHOULDER_ANGLE": (
                "LEFT_ELBOW",
                "LEFT_SHOULDER",
                "LEFT_HIP",
            ),  # Ângulo do braço em relação ao tronco
            "RIGHT_SHOULDER_ANGLE": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
            "LEFT_KNEE_ANGLE": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
            "RIGHT_KNEE_ANGLE": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
            "LEFT_HIP_ANGLE": (
                "LEFT_SHOULDER",
                "LEFT_HIP",
                "LEFT_KNEE",
            ),  # Ângulo do tronco/perna
            "RIGHT_HIP_ANGLE": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
        }
        logger.info(
            f"Ângulos chave para comparação definidos: {list(self.KEY_ANGLES.keys())}"
        )

    def _extract_angles_from_frame(self, frame_landmarks: list) -> dict:
        """
        Extrai os ângulos chave de uma lista de landmarks de um único frame.

        Argumentos:
            frame_landmarks (list): Uma lista de dicionários de landmarks para um frame,
                                    onde cada dicionário tem 'x', 'y', 'z', 'visibility'.

        Retorna:
            dict: Um dicionário onde as chaves são os nomes dos ângulos (definidos em KEY_ANGLES)
                  e os valores são os ângulos calculados em graus.
                  Retorna np.nan para ângulos que não puderam ser calculados (ex: landmarks ausentes/invísiveis).
        """
        angles = {}
        if not frame_landmarks:
            logger.debug(
                "Nenhum landmark encontrado no frame para extração de ângulos."
            )
            return {angle_name: np.nan for angle_name in self.KEY_ANGLES.keys()}

        for angle_name, landmark_names in self.KEY_ANGLES.items():
            p1_name, p2_name, p3_name = landmark_names

            # Tenta obter os landmarks pelos seus índices.
            # Se um landmark não existir (lista menor que o índice), ou se a visibilidade for muito baixa,
            # ele é considerado não detectado para este cálculo de ângulo.
            try:
                p1 = frame_landmarks[self.LANDMARKS[p1_name]]
                p2 = frame_landmarks[self.LANDMARKS[p2_name]]
                p3 = frame_landmarks[self.LANDMARKS[p3_name]]

                # Considera o landmark inválido se a visibilidade for muito baixa
                # O threshold de 0.6 é um valor comum, pode ser ajustado.
                if (
                    p1["visibility"] < 0.6
                    or p2["visibility"] < 0.6
                    or p3["visibility"] < 0.6
                ):
                    angles[angle_name] = np.nan
                    logger.debug(
                        f"Visibilidade baixa para {angle_name} em um ou mais pontos. Ignorando cálculo."
                    )
                    continue

                angle = calculate_angle(p1, p2, p3)
                angles[angle_name] = angle
            except (IndexError, KeyError) as e:
                # Caso o landmark não exista na lista (o que não deveria acontecer com MediaPipe 33 landmarks)
                # ou o nome do landmark esteja incorreto
                angles[angle_name] = np.nan
                logger.warning(
                    f"Erro ao extrair landmarks para o ângulo {angle_name}: {e}. Atribuindo NaN."
                )
            except Exception as e:
                # Captura outras exceções inesperadas durante o cálculo do ângulo
                angles[angle_name] = np.nan
                logger.error(
                    f"Erro inesperado ao calcular ângulo {angle_name}: {e}. Atribuindo NaN."
                )

        logger.debug(f"Ângulos extraídos para o frame: {angles}")
        return angles

    def compare_movements(
        self, aluno_data: list, mestre_data: list
    ) -> tuple[list, list]:
        """
        Compara os movimentos do aluno com os do mestre frame a frame.
        Esta é uma comparação inicial, focando em correspondência temporal direta.

        Argumentos:
            aluno_data (list): Lista de landmarks de todos os frames do vídeo do aluno.
                               Cada item da lista é a 'landmarks_data' de um frame.
            mestre_data (list): Lista de landmarks de todos os frames do vídeo do mestre.

        Retorna:
            tuple[list, list]: Uma tupla contendo:
                - lista_comparacao_raw (list): Uma lista de dicionários com os ângulos
                  do aluno e do mestre para cada frame, e as diferenças.
                - feedback_text (list): Uma lista de strings com feedback textual
                  gerado para cada frame ou para o movimento geral.
        """
        logger.info("Iniciando comparação de movimentos entre aluno e mestre.")

        lista_comparacao_raw = []
        feedback_text = []

        # Determina o número de frames a serem comparados.
        # Compara apenas até o menor número de frames entre os dois vídeos.
        min_frames = min(len(aluno_data), len(mestre_data))
        logger.info(f"Comparando {min_frames} frames dos vídeos.")

        for i in range(min_frames):
            logger.debug(f"Comparando Frame {i+1}/{min_frames}...")
            aluno_angles = self._extract_angles_from_frame(aluno_data[i])
            mestre_angles = self._extract_angles_from_frame(mestre_data[i])

            frame_comparison = {
                "frame": i,
                "aluno_angles": aluno_angles,
                "mestre_angles": mestre_angles,
                "differences": {},
            }
            frame_feedback = []

            # Itera sobre os ângulos chave para comparar
            for angle_name in self.KEY_ANGLES.keys():
                aluno_angle = aluno_angles.get(angle_name)
                mestre_angle = mestre_angles.get(angle_name)

                if np.isnan(aluno_angle) or np.isnan(mestre_angle):
                    # Se um dos ângulos não pôde ser calculado, não compara este ângulo neste frame.
                    frame_comparison["differences"][angle_name] = np.nan
                    # Não gera feedback específico para este ângulo neste frame.
                    continue

                diff = abs(aluno_angle - mestre_angle)
                frame_comparison["differences"][angle_name] = diff

                # Lógica simples para gerar feedback. Pode ser expandida e refinada.
                # Thresholds de exemplo, que devem ser ajustados com base na experiência real de Krav Magá.
                if diff > 20:  # Grande diferença
                    feedback_msg = f"Frame {i+1}: Grande diferença no {angle_name.replace('_', ' ').lower()}! Aluno: {aluno_angle:.1f}°, Mestre: {mestre_angle:.1f}°."
                    frame_feedback.append(feedback_msg)
                    logger.info(f"Feedback gerado: {feedback_msg}")
                elif diff > 10:  # Média diferença
                    feedback_msg = f"Frame {i+1}: Pequena diferença no {angle_name.replace('_', ' ').lower()}. Aluno: {aluno_angle:.1f}°, Mestre: {mestre_angle:.1f}°."
                    frame_feedback.append(feedback_msg)
                    logger.info(f"Feedback gerado: {feedback_msg}")
                # Se a diferença for menor que 10, pode ser considerada aceitável e não gera feedback.

            lista_comparacao_raw.append(frame_comparison)
            if frame_feedback:
                feedback_text.extend(
                    frame_feedback
                )  # Adiciona todos os feedbacks do frame

        if not feedback_text:
            feedback_text.append(
                "Nenhuma diferença significativa detectada. Boa execução!"
            )
            logger.info("Nenhuma diferença significativa geral detectada.")

        logger.info("Comparação de movimentos concluída.")
        return lista_comparacao_raw, feedback_text
