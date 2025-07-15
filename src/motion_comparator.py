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
        # https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model
        self.KEY_ANGLES = {
            "LEFT_ELBOW_ANGLE": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
            "RIGHT_ELBOW_ANGLE": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            "LEFT_SHOULDER_ANGLE": ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
            "RIGHT_SHOULDER_ANGLE": ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
            "LEFT_KNEE_ANGLE": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
            "RIGHT_KNEE_ANGLE": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
            "LEFT_HIP_ANGLE": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
            "RIGHT_HIP_ANGLE": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
            # Adicione mais ângulos conforme a necessidade para a análise de Krav Magá
            # Ex: ângulos do tronco, postura, etc.
        }

        # Mapeamento de nomes de landmarks para índices (para facilitar o acesso na lista `landmarks_data`)
        # Esta é uma representação simplificada. No MediaPipe, você usaria mp_pose.PoseLandmark. para obter os valores.
        # Para este contexto, assumimos que landmarks_data é uma lista de dicionários
        # e que a ordem ou o nome são consistentes com a saída do MediaPipe.
        # É fundamental que o `landmarks_data` vindo do `PoseEstimator` seja indexado corretamente.
        # A lista de dicionários `landmarks_list` em `PoseEstimator.process_frame` já é linear e indexável.
        # Então, esta parte do código precisa de um mapeamento correto dos nomes simbólicos
        # para os índices reais da lista de landmarks.
        # Exemplo de como obter os índices corretos se 'landmarks_data' é uma lista com 33 elementos:
        # import mediapipe as mp
        # LEFT_SHOULDER_IDX = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        # ... e assim por diante para cada um dos 33 landmarks.
        # Para simplificar aqui, vamos assumir que `landmarks_data` é uma lista de dicionários
        # onde cada dicionário já tem os valores x,y,z,visibility dos landmarks em uma ordem previsível
        # ou que você terá um mapeamento mais robusto.
        # Para os fins deste exemplo e para o uso com o PoseEstimator, onde landmarks_list
        # é uma lista em ordem, podemos usar os índices numéricos.

        # Importa PoseLandmark para mapear nomes para índices.
        # Isso garante que estamos usando os índices corretos do MediaPipe.
        import mediapipe as mp

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
            # ... adicione outros que forem relevantes
        }

        logger.info("MotionComparator inicializado.")
        logger.info(
            f"Ângulos chave para comparação definidos: {list(self.KEY_ANGLES.keys())}"
        )

    def _get_landmark_coords(self, landmarks_data: list, name: str) -> dict:
        """
        Retorna as coordenadas (x, y, z) e visibilidade de um landmark pelo nome.
        Args:
            landmarks_data (list): A lista de dicionários de landmarks de um frame.
            name (str): O nome do landmark (ex: "LEFT_SHOULDER").
        Returns:
            dict: Um dicionário com 'x', 'y', 'z' e 'visibility' do landmark.
        Raises:
            ValueError: Se o landmark não for encontrado ou os dados forem inválidos.
        """
        idx = self.landmark_indices.get(name)
        if idx is None or idx >= len(landmarks_data):
            # Isso pode acontecer se a detecção de pose falhar completamente em um frame.
            # Retorna um dicionário com valores padrao ou lança um erro, dependendo da robustez desejada.
            # Para comparação, retornar None ou um dict inválido levará a erros, então é melhor levantar.
            raise ValueError(
                f"Landmark '{name}' (índice {idx}) não encontrado ou fora dos limites em landmarks_data de tamanho {len(landmarks_data)}."
            )
        return landmarks_data[idx]

    def compare_movements(
        self,
        aluno_landmarks_history: list[list],
        mestre_landmarks_history: list[list],
    ) -> tuple[list, list]:
        """
        Compara as sequências de movimentos do aluno e do mestre frame a frame.

        Args:
            aluno_landmarks_history (list[list]): Lista de listas de landmarks do aluno,
                                                 onde cada sublista representa um frame.
            mestre_landmarks_history (list[list]): Lista de listas de landmarks do mestre.

        Returns:
            tuple[list, list]: Uma tupla contendo:
                - lista_comparacao_raw (list): Resultados detalhados da comparação frame a frame.
                - feedback_text (list): Feedback textual gerado.
        """
        logger.info("Iniciando a comparação de movimentos...")
        lista_comparacao_raw = []
        feedback_text = []

        # Usar o número mínimo de frames para evitar IndexError se um vídeo for mais curto
        min_frames = min(len(aluno_landmarks_history), len(mestre_landmarks_history))
        logger.info(f"Comparando {min_frames} frames.")

        for i in range(min_frames):
            aluno_frame_landmarks = aluno_landmarks_history[i]
            mestre_frame_landmarks = mestre_landmarks_history[i]

            frame_comparison = {"frame": i + 1, "angles_diff": {}}
            frame_feedback = []

            for angle_name, (p1_name, p2_name, p3_name) in self.KEY_ANGLES.items():
                try:
                    # Obtém as coordenadas dos landmarks para o aluno
                    aluno_p1 = self._get_landmark_coords(aluno_frame_landmarks, p1_name)
                    aluno_p2 = self._get_landmark_coords(aluno_frame_landmarks, p2_name)
                    aluno_p3 = self._get_landmark_coords(aluno_frame_landmarks, p3_name)

                    # Calcula o ângulo para o aluno
                    aluno_angle = calculate_angle(aluno_p1, aluno_p2, aluno_p3)

                    # Obtém as coordenadas dos landmarks para o mestre
                    mestre_p1 = self._get_landmark_coords(
                        mestre_frame_landmarks, p1_name
                    )
                    mestre_p2 = self._get_landmark_coords(
                        mestre_frame_landmarks, p2_name
                    )
                    mestre_p3 = self._get_landmark_coords(
                        mestre_frame_landmarks, p3_name
                    )

                    # Calcula o ângulo para o mestre
                    mestre_angle = calculate_angle(mestre_p1, mestre_p2, mestre_p3)

                    # Calcula a diferença absoluta entre os ângulos
                    diff = abs(aluno_angle - mestre_angle)
                    frame_comparison["angles_diff"][angle_name] = diff

                    # Lógica de feedback: ajustável conforme a necessidade
                    # Esta é uma regra de negócio que pode ser medida e refinada.
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

                except ValueError as ve:
                    # Isso captura erros se um landmark específico não for encontrado,
                    # o que pode acontecer se a detecção de pose falhar para um ponto específico.
                    err_msg = f"Frame {i+1}: Erro ao calcular {angle_name}: {ve}"
                    frame_feedback.append(err_msg)
                    logger.warning(err_msg)
                except Exception as ex:
                    # Captura outras exceções inesperadas
                    err_msg = (
                        f"Frame {i+1}: Erro inesperado ao calcular {angle_name}: {ex}"
                    )
                    frame_feedback.append(err_msg)
                    logger.error(err_msg, exc_info=True)

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
