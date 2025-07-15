# tests/test_motion_comparator.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import mediapipe as mp # Necessário para os mocks de PoseLandmark

from src.motion_comparator import MotionComparator
from src.utils import get_logger, calculate_angle # Importar para mockar o logger e a função

# Mock do logger para evitar saída real durante os testes e verificar chamadas
@pytest.fixture(autouse=True)
def mock_logger():
    """
    Mocka a função get_logger para controlar o logger nos testes.
    """
    with patch('src.motion_comparator.get_logger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance

# Mock da função calculate_angle para isolar a lógica de comparação
@pytest.fixture
def mock_calculate_angle():
    """
    Mocka a função calculate_angle para retornar valores controlados.
    """
    with patch('src.motion_comparator.calculate_angle') as mock_calc_angle:
        yield mock_calc_angle

def test_motion_comparator_initialization(mock_logger):
    """
    Testa se o MotionComparator é inicializado corretamente e se o logger é usado.
    """
    # Instancia o MotionComparator
    comparator = MotionComparator()

    # Verifica se o logger foi chamado para informar a inicialização
    mock_logger.info.assert_any_call("Inicializando MotionComparator.")
    mock_logger.info.assert_any_call(
        f"Ângulos chave para comparação definidos: {list(comparator.KEY_ANGLES.keys())}"
    )

    # Verifica se KEY_ANGLES e landmark_indices foram definidos
    assert isinstance(comparator.KEY_ANGLES, dict)
    assert len(comparator.KEY_ANGLES) > 0
    assert isinstance(comparator.landmark_indices, dict)
    assert len(comparator.landmark_indices) > 0

    # Verifica se os índices dos landmarks são do tipo correto (int)
    for key, value in comparator.landmark_indices.items():
        assert isinstance(value, int)

def test_get_landmark_coords_valid(mock_logger):
    """
    Testa _get_landmark_coords com dados válidos.
    """
    comparator = MotionComparator()
    # Simula uma lista de landmarks com 33 elementos
    landmarks_data = [{'x': i, 'y': i, 'z': i, 'visibility': 0.9} for i in range(33)]

    # Usa um landmark que sabemos que existe (ex: NOSE)
    nose_coords = comparator._get_landmark_coords(landmarks_data, "NOSE")
    assert nose_coords == {'x': mp.solutions.pose.PoseLandmark.NOSE.value,
                           'y': mp.solutions.pose.PoseLandmark.NOSE.value,
                           'z': mp.solutions.pose.PoseLandmark.NOSE.value,
                           'visibility': 0.9}

def test_get_landmark_coords_invalid_name(mock_logger):
    """
    Testa _get_landmark_coords com nome de landmark inválido.
    """
    comparator = MotionComparator()
    landmarks_data = [{'x': i, 'y': i, 'z': i, 'visibility': 0.9} for i in range(33)]

    with pytest.raises(ValueError, match="Landmark 'INVALID_NAME'"):
        comparator._get_landmark_coords(landmarks_data, "INVALID_NAME")

def test_get_landmark_coords_out_of_bounds_index(mock_logger):
    """
    Testa _get_landmark_coords com índice fora dos limites.
    """
    comparator = MotionComparator()
    # Simula uma lista de landmarks menor que o índice esperado para um landmark
    landmarks_data = [{'x': i, 'y': i, 'z': i, 'visibility': 0.9} for i in range(5)] # Apenas 5 landmarks

    # Tenta acessar um landmark com índice alto (ex: RIGHT_ANKLE é 28)
    with pytest.raises(ValueError, match="Landmark 'RIGHT_ANKLE'"):
        comparator._get_landmark_coords(landmarks_data, "RIGHT_ANKLE")

def test_compare_movements_basic(mock_logger, mock_calculate_angle):
    """
    Testa a função compare_movements com um cenário básico.
    """
    comparator = MotionComparator()

    # Mock de dados de landmarks (simplificados)
    # Cada sublista é um frame, cada dict é um landmark
    # Assumimos que os índices são mapeados corretamente
    aluno_landmarks_history = [
        [{'x': 1, 'y': 1, 'z': 1, 'visibility': 1} for _ in range(33)], # Frame 0
        [{'x': 2, 'y': 2, 'z': 2, 'visibility': 1} for _ in range(33)], # Frame 1
    ]
    mestre_landmarks_history = [
        [{'x': 1.1, 'y': 1.1, 'z': 1.1, 'visibility': 1} for _ in range(33)], # Frame 0
        [{'x': 2.2, 'y': 2.2, 'z': 2.2, 'visibility': 1} for _ in range(33)], # Frame 1
    ]

    # Configura o mock_calculate_angle para retornar valores específicos
    # para simular diferenças
    mock_calculate_angle.side_effect = [
        90, 95, # Aluno e Mestre para LEFT_ELBOW_ANGLE no Frame 0
        45, 40, # Aluno e Mestre para RIGHT_ELBOW_ANGLE no Frame 0
        # ... e assim por diante para todos os ângulos e frames
        90, 90, # Aluno e Mestre para LEFT_ELBOW_ANGLE no Frame 1 (sem diferença)
        45, 60, # Aluno e Mestre para RIGHT_ELBOW_ANGLE no Frame 1 (grande diferença)
        # Mais retornos seriam necessários para cobrir todos os KEY_ANGLES
        # Para simplificar o teste, vamos focar nos primeiros
    ]

    # Para garantir que temos retornos suficientes para todos os ângulos em 2 frames
    num_angles = len(comparator.KEY_ANGLES)
    mock_calculate_angle.side_effect = [
        # Frame 0
        90, 95, # LEFT_ELBOW_ANGLE (diff 5)
        45, 40, # RIGHT_ELBOW_ANGLE (diff 5)
        100, 125, # LEFT_SHOULDER_ANGLE (diff 25) -> Grande diferença
        70, 75, # RIGHT_SHOULDER_ANGLE (diff 5)
        150, 165, # LEFT_KNEE_ANGLE (diff 15) -> Pequena diferença
        170, 172, # RIGHT_KNEE_ANGLE (diff 2)
        80, 85, # LEFT_HIP_ANGLE (diff 5)
        90, 92, # RIGHT_HIP_ANGLE (diff 2)
        # Frame 1
        90, 90, # LEFT_ELBOW_ANGLE (diff 0)
        45, 60, # RIGHT_ELBOW_ANGLE (diff 15) -> Pequena diferença
        100, 105, # LEFT_SHOULDER_ANGLE (diff 5)
        70, 70, # RIGHT_SHOULDER_ANGLE (diff 0)
        150, 155, # LEFT_KNEE_ANGLE (diff 5)
        170, 170, # RIGHT_KNEE_ANGLE (diff 0)
        80, 80, # LEFT_HIP_ANGLE (diff 0)
        90, 90, # RIGHT_HIP_ANGLE (diff 0)
    ]


    raw_results, feedback = comparator.compare_movements(aluno_landmarks_history, mestre_landmarks_history)

    mock_logger.info.assert_any_call("Iniciando a comparação de movimentos...")
    mock_logger.info.assert_any_call("Comparando 2 frames.")
    mock_logger.info.assert_any_call("Comparação de movimentos concluída.")

    assert len(raw_results) == 2
    assert len(feedback) > 0 # Deve haver feedback devido às diferenças simuladas

    # Verifica feedback específico
    assert "Frame 1: Grande diferença no left shoulder angle!" in feedback
    assert "Frame 1: Pequena diferença no left knee angle." in feedback
    assert "Frame 2: Pequena diferença no right elbow angle." in feedback
    assert "Nenhuma diferença significativa detectada. Boa execução!" not in feedback

def test_compare_movements_no_significant_diff(mock_logger, mock_calculate_angle):
    """
    Testa a função compare_movements quando não há diferenças significativas.
    """
    comparator = MotionComparator()

    aluno_landmarks_history = [
        [{'x': 1, 'y': 1, 'z': 1, 'visibility': 1} for _ in range(33)],
    ]
    mestre_landmarks_history = [
        [{'x': 1.0, 'y': 1.0, 'z': 1.0, 'visibility': 1} for _ in range(33)],
    ]

    # Configura o mock_calculate_angle para retornar pouca ou nenhuma diferença
    mock_calculate_angle.side_effect = [
        90, 91, # diff 1
        45, 46, # diff 1
        100, 102, # diff 2
        70, 71, # diff 1
        150, 151, # diff 1
        170, 171, # diff 1
        80, 81, # diff 1
        90, 91, # diff 1
    ]

    raw_results, feedback = comparator.compare_movements(aluno_landmarks_history, mestre_landmarks_history)

    assert len(raw_results) == 1
    assert len(feedback) == 1
    assert "Nenhuma diferença significativa detectada. Boa execução!" in feedback
    mock_logger.info.assert_any_call("Nenhuma diferença significativa geral detectada.")

def test_compare_movements_missing_landmarks_in_frame(mock_logger, mock_calculate_angle):
    """
    Testa a função compare_movements quando um frame não tem landmarks.
    """
    comparator = MotionComparator()

    aluno_landmarks_history = [
        [{'x': 1, 'y': 1, 'z': 1, 'visibility': 1} for _ in range(33)], # Frame 0
        None, # Frame 1 - sem landmarks
        [{'x': 2, 'y': 2, 'z': 2, 'visibility': 1} for _ in range(33)], # Frame 2
    ]
    mestre_landmarks_history = [
        [{'x': 1.1, 'y': 1.1, 'z': 1.1, 'visibility': 1} for _ in range(33)], # Frame 0
        [{'x': 2.2, 'y': 2.2, 'z': 2.2, 'visibility': 1} for _ in range(33)], # Frame 1
        [{'x': 3.3, 'y': 3.3, 'z': 3.3, 'visibility': 1} for _ in range(33)], # Frame 2
    ]

    # Mock calculate_angle para os frames que serão processados
    mock_calculate_angle.side_effect = [
        # Frame 0
        90, 95, # LEFT_ELBOW_ANGLE (diff 5)
        # ... (outros ângulos para Frame 0)
        # Frame 2
        100, 105, # LEFT_ELBOW_ANGLE (diff 5)
        # ... (outros ângulos para Frame 2)
    ] * (len(comparator.KEY_ANGLES) * 2) # Garante retornos suficientes

    raw_results, feedback = comparator.compare_movements(aluno_landmarks_history, mestre_landmarks_history)

    assert len(raw_results) == 3
    assert "Frame 2: Um ou ambos os vídeos não têm landmarks detectados. Pulando este frame." in feedback
    mock_logger.warning.assert_any_call("Frame 2: Um ou ambos os vídeos não têm landmarks detectados. Pulando este frame.")
    mock_logger.info.assert_any_call("Comparando 3 frames.")
    # Verifica que o calculate_angle não foi chamado para o frame com None
    # Isso é mais complexo de verificar diretamente com side_effect, mas o log e o skip confirmam.