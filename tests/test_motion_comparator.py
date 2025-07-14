# tests/test_motion_comparator.py

import pytest
import numpy as np
import os
import sys

# IMPORTANTE: Adiciona o diretório raiz do projeto ao sys.path para importações.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.motion_comparator import MotionComparator
from src.utils import setup_logging

logger = setup_logging()  # Configura o logger para os testes


# Fixture para uma instância do MotionComparator
@pytest.fixture
def motion_comparator_instance():
    """Fornece uma instância de MotionComparator para os testes."""
    logger.info("Configurando fixture: motion_comparator_instance")
    return MotionComparator()


# Fixture para gerar dados de landmarks mock (simulados)
def generate_mock_landmarks(
    num_frames: int, num_landmarks: int = 33, base_angle_value: float = 90.0
) -> list:
    """
    Gera uma lista de dados de landmarks mock para simular frames de vídeo,
    com a garantia de que o LEFT_ELBOW_ANGLE terá o valor base_angle_value.

    Argumentos:
        num_frames (int): Número de frames a serem gerados.
        num_landmarks (int): Número de landmarks por frame (padrão MediaPipe = 33).
        base_angle_value (float): O valor do ângulo (em graus) que o LEFT_ELBOW_ANGLE
                                  deve ter para os landmarks gerados.

    Retorna:
        list: Uma lista onde cada item representa um frame, contendo uma lista de dicionários de landmarks.
    """
    mock_data = []

    # Indices para LEFT_SHOULDER (P1), LEFT_ELBOW (P2 - vértice), LEFT_WRIST (P3)
    SHOULDER_IDX = 11
    ELBOW_IDX = 13
    WRIST_IDX = 15

    for f_idx in range(num_frames):
        # Inicializa todos os landmarks com pontos dummy e alta visibilidade
        frame_landmarks = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9}
            for _ in range(num_landmarks)
        ]

        # Definir P2 (LEFT_ELBOW) como o vértice em (0,0,0) para facilitar o cálculo
        frame_landmarks[ELBOW_IDX] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9}

        # Definir P1 (LEFT_SHOULDER) em (0,1,0) para ser o primeiro vetor do ângulo
        frame_landmarks[SHOULDER_IDX] = {
            "x": 0.0,
            "y": 1.0,
            "z": 0.0,
            "visibility": 0.9,
        }

        # Definir P3 (LEFT_WRIST) com base no base_angle_value, usando pontos que já sabemos
        # que funcionam com a função `calculate_angle` testada em `test_utils.py`.
        p3_x, p3_y, p3_z = 0.0, 0.0, 0.0  # Inicializa com valores padrão

        if base_angle_value == 180.0:
            # Simula um braço esticado: P1=(0,1,0), P2=(0,0,0), P3=(0,-1,0)
            p3_x, p3_y, p3_z = 0.0, -1.0, 0.0
        elif base_angle_value == 90.0:
            # Simula um ângulo reto: P1=(0,1,0), P2=(0,0,0), P3=(1,0,0)
            p3_x, p3_y, p3_z = 1.0, 0.0, 0.0
        elif base_angle_value == 45.0:
            # Simula um ângulo agudo: P1=(0,1,0), P2=(0,0,0), P3=(1,1,0)
            p3_x, p3_y, p3_z = 1.0, 1.0, 0.0
        elif base_angle_value == 135.0:
            # Simula um ângulo obtuso: P1=(0,1,0), P2=(0,0,0), P3=(-1,1,0)
            p3_x, p3_y, p3_z = -1.0, 1.0, 0.0
        else:
            # Para outros ângulos não definidos especificamente, retorna 0 para evitar NaN
            # ou calcula um ponto genérico. Para este teste, vamos garantir os casos chave.
            logger.warning(
                f"generate_mock_landmarks: base_angle_value {base_angle_value} não explicitamente tratado para cálculo exato. Ângulo pode não ser preciso."
            )
            p3_x, p3_y, p3_z = (
                0.0,
                0.0,
                0.0,
            )  # Default para algo que dê 0.0 se não for p1=p2 ou p3=p2

        frame_landmarks[WRIST_IDX] = {
            "x": p3_x,
            "y": p3_y,
            "z": p3_z,
            "visibility": 0.9,
        }

        mock_data.append(frame_landmarks)
    return mock_data


class TestMotionComparator:
    """
    Classe de testes para o MotionComparator.
    Verifica se a comparação de movimentos e a extração de feedback funcionam como esperado.
    """

    def test_initialization(self, motion_comparator_instance):
        """
        Testa se o MotionComparator é inicializado corretamente e se os ângulos chave estão definidos.
        """
        logger.info("Executando test_initialization...")
        assert motion_comparator_instance is not None
        assert hasattr(motion_comparator_instance, "KEY_ANGLES")
        assert "LEFT_ELBOW_ANGLE" in motion_comparator_instance.KEY_ANGLES
        logger.info("test_initialization PASSED.")

    def test_extract_angles_from_frame_valid(self, motion_comparator_instance):
        """
        Testa se a extração de ângulos de um frame com landmarks válidos funciona.
        """
        logger.info("Executando test_extract_angles_from_frame_valid...")
        # Simula landmarks para um braço esquerdo esticado (180 graus)
        mock_landmarks = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9}  # Dummy para lm_idx 0
            for _ in range(33)  # 33 landmarks
        ]
        # LEFT_SHOULDER (11), LEFT_ELBOW (13), LEFT_WRIST (15)
        mock_landmarks[motion_comparator_instance.LANDMARKS["LEFT_SHOULDER"]] = {
            "x": 0.0,
            "y": 1.0,
            "z": 0.0,
            "visibility": 0.9,
        }
        mock_landmarks[motion_comparator_instance.LANDMARKS["LEFT_ELBOW"]] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "visibility": 0.9,
        }
        mock_landmarks[motion_comparator_instance.LANDMARKS["LEFT_WRIST"]] = {
            "x": 0.0,
            "y": -1.0,
            "z": 0.0,
            "visibility": 0.9,
        }  # Alinhado

        angles = motion_comparator_instance._extract_angles_from_frame(mock_landmarks)

        # O ângulo esperado para um braço esticado é 180 graus
        assert "LEFT_ELBOW_ANGLE" in angles
        assert angles["LEFT_ELBOW_ANGLE"] == pytest.approx(180.0, abs=0.01)

        # Verifica se outros ângulos (não configurados especificamente) são NaN ou 0.0
        assert np.isnan(angles["RIGHT_ELBOW_ANGLE"]) or angles[
            "RIGHT_ELBOW_ANGLE"
        ] == pytest.approx(0.0)

        logger.info("test_extract_angles_from_frame_valid PASSED.")

    def test_extract_angles_from_frame_no_landmarks(self, motion_comparator_instance):
        """
        Testa a extração de ângulos de um frame sem landmarks (lista vazia).
        Deve retornar NaN para todos os ângulos.
        """
        logger.info("Executando test_extract_angles_from_frame_no_landmarks...")
        angles = motion_comparator_instance._extract_angles_from_frame([])

        for angle_name in motion_comparator_instance.KEY_ANGLES.keys():
            assert np.isnan(angles[angle_name])
        logger.info("test_extract_angles_from_frame_no_landmarks PASSED.")

    def test_extract_angles_from_frame_low_visibility(self, motion_comparator_instance):
        """
        Testa a extração de ângulos quando a visibilidade de um landmark é muito baixa.
        Deve resultar em NaN para o ângulo afetado.
        """
        logger.info("Executando test_extract_angles_from_frame_low_visibility...")
        # Simula landmarks com uma visibilidade muito baixa para o cotovelo esquerdo
        mock_landmarks = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9} for _ in range(33)
        ]
        mock_landmarks[motion_comparator_instance.LANDMARKS["LEFT_SHOULDER"]] = {
            "x": 0.0,
            "y": 1.0,
            "z": 0.0,
            "visibility": 0.9,
        }
        mock_landmarks[motion_comparator_instance.LANDMARKS["LEFT_ELBOW"]] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "visibility": 0.01,
        }  # Visibilidade baixa
        mock_landmarks[motion_comparator_instance.LANDMARKS["LEFT_WRIST"]] = {
            "x": 0.0,
            "y": -1.0,
            "z": 0.0,
            "visibility": 0.9,
        }

        angles = motion_comparator_instance._extract_angles_from_frame(mock_landmarks)

        assert "LEFT_ELBOW_ANGLE" in angles
        assert np.isnan(angles["LEFT_ELBOW_ANGLE"])
        logger.info("test_extract_angles_from_frame_low_visibility PASSED.")

    def test_compare_perfect_match(self, motion_comparator_instance):
        """
        Testa a comparação de movimentos onde aluno e mestre têm ângulos idênticos.
        Esperado: diferenças próximas a zero e feedback positivo.
        """
        logger.info("Executando test_compare_perfect_match...")
        # Cria dados mock onde aluno e mestre têm o mesmo ângulo (180 graus de cotovelo)
        aluno_data = generate_mock_landmarks(num_frames=1, base_angle_value=180.0)
        mestre_data = generate_mock_landmarks(num_frames=1, base_angle_value=180.0)

        raw_comparison, feedback = motion_comparator_instance.compare_movements(
            aluno_data, mestre_data
        )

        assert len(raw_comparison) == 1
        assert raw_comparison[0]["frame"] == 0
        assert "LEFT_ELBOW_ANGLE" in raw_comparison[0]["differences"]
        assert raw_comparison[0]["differences"]["LEFT_ELBOW_ANGLE"] == pytest.approx(
            0.0, abs=0.01
        )

        # Espera um feedback positivo ou ausência de feedback de erro
        assert any("Nenhuma diferença significativa" in msg for msg in feedback)
        logger.info("test_compare_perfect_match PASSED.")

    def test_compare_with_significant_difference(self, motion_comparator_instance):
        """
        Testa a comparação de movimentos com uma diferença significativa de ângulo.
        Esperado: feedback negativo.
        """
        logger.info("Executando test_compare_with_significant_difference...")
        # Aluno com 90 graus, Mestre com 180 graus (grande diferença)
        aluno_data = generate_mock_landmarks(num_frames=1, base_angle_value=90.0)
        mestre_data = generate_mock_landmarks(num_frames=1, base_angle_value=180.0)

        raw_comparison, feedback = motion_comparator_instance.compare_movements(
            aluno_data, mestre_data
        )

        assert len(raw_comparison) == 1
        assert raw_comparison[0]["differences"]["LEFT_ELBOW_ANGLE"] == pytest.approx(
            90.0, abs=0.01
        )

        # Espera um feedback específico sobre a diferença de ângulo
        assert any("Grande diferença no left elbow angle!" in msg for msg in feedback)
        logger.info("test_compare_with_significant_difference PASSED.")

    def test_compare_different_frame_counts(self, motion_comparator_instance):
        """
        Testa a comparação com vídeos de durações diferentes.
        Deve comparar apenas o número de frames do vídeo mais curto.
        """
        logger.info("Executando test_compare_different_frame_counts...")
        aluno_data = generate_mock_landmarks(num_frames=5)
        mestre_data = generate_mock_landmarks(num_frames=3)  # Mestre é mais curto

        raw_comparison, feedback = motion_comparator_instance.compare_movements(
            aluno_data, mestre_data
        )

        assert len(raw_comparison) == 3  # Deve comparar apenas 3 frames
        logger.info("test_compare_different_frame_counts PASSED.")

    def test_compare_with_nan_angles(self, motion_comparator_instance):
        """
        Testa a comparação quando um dos vídeos tem landmarks ausentes (resultando em NaN).
        Esses ângulos devem ser ignorados na comparação.
        """
        logger.info("Executando test_compare_with_nan_angles...")
        aluno_data = generate_mock_landmarks(num_frames=1, base_angle_value=180.0)
        # Mestre com dados vazios para simular landmarks ausentes
        mestre_data = [[]]

        raw_comparison, feedback = motion_comparator_instance.compare_movements(
            aluno_data, mestre_data
        )

        assert len(raw_comparison) == 1
        # Verifica se as diferenças para todos os ângulos são NaN, pois o mestre não tem dados
        for angle_name in motion_comparator_instance.KEY_ANGLES.keys():
            assert np.isnan(raw_comparison[0]["differences"][angle_name])

        # O feedback não deve conter mensagens de erro de comparação específica se for NaN
        # Pode ter a mensagem de 'Nenhuma diferença significativa' se não houver outros erros
        assert (
            any("Nenhuma diferença significativa" in msg for msg in feedback)
            or not feedback
        )
        logger.info("test_compare_with_nan_angles PASSED.")
