# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------------
#  Krav Maga Motion Analyzer
#  version 1.1.0
#  Copyright (C) 2024,
#  Gleyson Atanazio [Sujeito Programador]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------

# Importa a classe abstrata de base para criar classes abstratas.
from abc import ABC, abstractmethod
# Importa a biblioteca logging para registrar eventos e mensagens.
import logging
# Importa a biblioteca NumPy para operações numéricas, especialmente com arrays.
import numpy as np
# Importa a função de similaridade de cosseno do scikit-learn para comparar vetores.
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------------------------------
# Configuração do Logging
# --------------------------------------------------------------------------------------------------

# Configura o logging para exibir mensagens a partir do nível INFO.
# O formato da mensagem inclui a hora, o nível do log e a mensagem.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------------------------------------------------------------------------------------
# Classe Base para Comparação de Movimento
# --------------------------------------------------------------------------------------------------

class MotionComparatorBase(ABC):
    """
    Classe base abstrata para comparadores de movimento.
    Define a interface que todos os comparadores concretos devem seguir.
    Isso garante a extensibilidade e a manutenção do código (Zen of Python: "Belo é melhor que feio").
    """
    @abstractmethod
    def compare_poses(self, landmarks_aluno, landmarks_mestre):
        """
        Método abstrato para comparar as poses de um aluno e de um mestre.

        Args:
            landmarks_aluno (list): Lista de landmarks (pontos de referência) do aluno.
            landmarks_mestre (list): Lista de landmarks (pontos de referência) do mestre.

        Returns:
            tuple: Uma tupla contendo a pontuação de similaridade e o feedback textual.
        """
        pass

# --------------------------------------------------------------------------------------------------
# Implementação Concreta do Comparador de Movimento
# --------------------------------------------------------------------------------------------------

class MotionComparator(MotionComparatorBase):
    """
    Classe responsável por comparar os movimentos entre o aluno e o mestre.
    Calcula ângulos das articulações e fornece uma pontuação de similaridade e feedback.
    """

    def __init__(self):
        """
        Inicializador da classe MotionComparator.
        Define os mapeamentos das articulações que serão analisadas.
        """
        # Mapeamento de articulações e os landmarks correspondentes para cálculo dos ângulos.
        # A ordem é [ponto_central, ponto_adjacente_1, ponto_adjacente_2].
        self.angle_definitions = {
            'cotovelo_direito': (14, 12, 16),
            'cotovelo_esquedo': (13, 11, 15),
            'ombro_direito': (12, 14, 24),
            'ombro_esquedo': (11, 13, 23),
            'quadril_direito': (24, 26, 22),
            'quadril_esquedo': (23, 25, 21),
            'joelho_direito': (26, 24, 28),
            'joelho_esquedo': (25, 23, 27)
        }
        logging.info("MotionComparator inicializado com definições de ângulo.")

    def calculate_angle(self, landmarks, p1_idx, p2_idx, p3_idx):
        """
        Calcula o ângulo entre três pontos (landmarks).

        Args:
            landmarks (list): A lista de landmarks da pose.
            p1_idx (int): Índice do ponto que será o vértice do ângulo.
            p2_idx (int): Índice do primeiro ponto adjacente.
            p3_idx (int): Índice do segundo ponto adjacente.

        Returns:
            float: O ângulo calculado em graus. Retorna 0.0 se os landmarks não forem visíveis.
        """
        try:
            # Obtém as coordenadas (x, y) de cada landmark a partir da lista.
            p1 = np.array([landmarks[p1_idx].x, landmarks[p1_idx].y])
            p2 = np.array([landmarks[p2_idx].x, landmarks[p2_idx].y])
            p3 = np.array([landmarks[p3_idx].x, landmarks[p3_idx].y])

            # Cria os vetores a partir do ponto vértice (p1).
            vector1 = p2 - p1
            vector2 = p3 - p1

            # Calcula o produto escalar dos dois vetores.
            dot_product = np.dot(vector1, vector2)
            # Calcula a magnitude (norma) de cada vetor.
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)

            # Calcula o cosseno do ângulo. Para evitar divisão por zero, verifica as normas.
            if norm_vector1 == 0 or norm_vector2 == 0:
                return 0.0
            
            cosine_angle = dot_product / (norm_vector1 * norm_vector2)
            
            # Garante que o valor do cosseno esteja no intervalo [-1, 1] para evitar erros de domínio.
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            # Calcula o ângulo em radianos usando o arco cosseno.
            angle_rad = np.arccos(cosine_angle)
            # Converte o ângulo de radianos para graus.
            angle_deg = np.degrees(angle_rad)

            return angle_deg

        except (IndexError, TypeError) as e:
            # Captura exceções se os landmarks não estiverem disponíveis e registra um erro.
            logging.error(f"Erro ao calcular ângulo para os índices {p1_idx, p2_idx, p3_idx}: {e}")
            return 0.0

    def get_all_angles(self, landmarks):
        """
        Calcula todos os ângulos definidos para uma determinada pose.

        Args:
            landmarks (list): A lista de landmarks da pose.

        Returns:
            dict: Um dicionário onde as chaves são os nomes das articulações e os valores são os ângulos.
        """
        if not landmarks:
            # Se não houver landmarks, retorna um dicionário vazio.
            return {}

        angles = {}
        # Itera sobre as definições de ângulo para calcular cada um.
        for name, (p1, p2, p3) in self.angle_definitions.items():
            angles[name] = self.calculate_angle(landmarks, p1, p2, p3)
        
        logging.debug(f"Ângulos calculados: {angles}")
        return angles

    def compare_poses(self, landmarks_aluno, landmarks_mestre):
        """
        Compara as poses do aluno e do mestre, calculando uma pontuação de similaridade e feedback.

        Args:
            landmarks_aluno (list): Lista de landmarks do aluno.
            landmarks_mestre (list): Lista de landmarks do mestre.

        Returns:
            tuple: (float, str) - Pontuação de similaridade (0 a 100) e feedback textual.
                   Retorna (0.0, "Analisando...") se alguma pose for inválida.
        """
        # Verifica se os landmarks de ambos são válidos.
        if not landmarks_aluno or not landmarks_mestre:
            logging.warning("Comparação de pose pulada: landmarks de aluno ou mestre ausentes.")
            return 0.0, "Analisando..."

        # Calcula todos os ângulos para o aluno e para o mestre.
        angles_aluno = self.get_all_angles(landmarks_aluno)
        angles_mestre = self.get_all_angles(landmarks_mestre)

        # Se algum dos dicionários de ângulos estiver vazio, não é possível comparar.
        if not angles_aluno or not angles_mestre:
            logging.warning("Comparação de pose pulada: não foi possível calcular os ângulos.")
            return 0.0, "Aguardando pose..."

        # Converte os dicionários de ângulos em vetores (arrays NumPy) para o cálculo.
        # A ordem das chaves é garantida para que a comparação seja correta.
        labels = sorted(angles_aluno.keys())
        vec_aluno = np.array([angles_aluno[k] for k in labels]).reshape(1, -1)
        vec_mestre = np.array([angles_mestre[k] for k in labels]).reshape(1, -1)
        
        # Calcula a similaridade de cosseno entre os dois vetores de ângulos.
        # O resultado é uma matriz, então pegamos o primeiro elemento.
        similarity = cosine_similarity(vec_aluno, vec_mestre)[0, 0]
        # Converte a similaridade (que vai de -1 a 1) para uma pontuação de 0 a 100.
        score = (similarity + 1) / 2 * 100
        
        logging.info(f"Similaridade calculada: {similarity:.2f}, Pontuação: {score:.2f}")

        # Gera o feedback textual com base na maior diferença de ângulo.
        feedback = self._generate_feedback(angles_aluno, angles_mestre, labels)
        
        return score, feedback

    def _generate_feedback(self, angles_aluno, angles_mestre, labels):
        """
        Gera um feedback textual baseado na maior diferença angular entre o aluno e o mestre.

        Args:
            angles_aluno (dict): Dicionário de ângulos do aluno.
            angles_mestre (dict): Dicionário de ângulos do mestre.
            labels (list): Lista ordenada de nomes de articulações.

        Returns:
            str: Uma string com a dica de correção, ou uma mensagem de parabéns.
        """
        diffs = {label: abs(angles_aluno[label] - angles_mestre[label]) for label in labels}
        
        # Encontra a articulação com a maior diferença de ângulo.
        if not diffs:
            return "Movimento Perfeito!"
            
        max_diff_label = max(diffs, key=diffs.get)
        max_diff_value = diffs[max_diff_label]

        # Define um limiar para considerar a correção necessária.
        # Se a maior diferença for pequena, o movimento é considerado bom.
        if max_diff_value < 10:  # Limiar de 10 graus
            logging.info("Diferença de ângulo abaixo do limiar. Feedback positivo.")
            return "Excelente movimento!"

        # Compara o ângulo do aluno com o do mestre para dar a dica correta.
        aluno_angle = angles_aluno[max_diff_label]
        mestre_angle = angles_mestre[max_diff_label]
        
        # Formata o nome da articulação para ser mais legível.
        readable_label = max_diff_label.replace('_', ' ').title()
        
        if aluno_angle < mestre_angle:
            feedback = f"Aumente o ângulo do {readable_label}"
        else:
            feedback = f"Diminua o ângulo do {readable_label}"
            
        logging.info(f"Feedback gerado: '{feedback}' para a articulação '{readable_label}' com diferença de {max_diff_value:.2f} graus.")
        return feedback