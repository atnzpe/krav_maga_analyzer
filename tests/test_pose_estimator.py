# tests/test_pose_estimator.py

import pytest
import numpy as np
import cv2
import mediapipe as mp
from unittest.mock import patch, MagicMock
import logging
from src.pose_estimator import PoseEstimator
from src.utils import (
    setup_logging,
)  # Importa setup_logging para configurar o logger de teste

# Configura o logging para os testes, desativando a saída para o console para não poluir a tela.
setup_logging(
    log_dir="logs_tests",
    log_file_name="test_pose_estimator.log",
    level=logging.CRITICAL,
)


# Fixture para criar uma imagem dummy (preta) para testes.
@pytest.fixture
def dummy_image():
    """
    Cria uma imagem preta de 480x640 pixels para usar como entrada nos testes.
    """
    # np.zeros cria um array preenchido com zeros, que representa uma imagem preta.
    # (altura, largura, canais de cor)
    return np.zeros((480, 640, 3), dtype=np.uint8)


# Fixture para instanciar o PoseEstimator.
@pytest.fixture
def pose_estimator_instance():
    """
    Retorna uma instância de PoseEstimator para os testes.
    Garante que o método close() seja chamado após cada teste para liberar recursos.
    """
    estimator = PoseEstimator()
    yield estimator
    estimator.close()  # Garante que o modelo MediaPipe seja fechado.


# Teste para verificar a inicialização do PoseEstimator.
def test_pose_estimator_initialization(pose_estimator_instance):
    """
    Verifica se o PoseEstimator é inicializado corretamente e se as propriedades
    mp_drawing, mp_drawing_styles e mp_pose são acessíveis.
    """
    assert pose_estimator_instance.mp_drawing is not None
    assert pose_estimator_instance.mp_drawing_styles is not None
    assert pose_estimator_instance.mp_pose is not None
    assert pose_estimator_instance.pose is not None


# Teste para verificar o processamento de um frame sem landmarks detectados.
def test_process_frame_no_landmarks(pose_estimator_instance, dummy_image):
    """
    Testa o método process_frame com uma imagem dummy onde não há landmarks esperados.
    Deve retornar a imagem original (ou uma cópia) e `results.pose_landmarks` como None.
    """
    # Usamos patch para simular o comportamento do MediaPipe,
    # fazendo com que ele retorne None para pose_landmarks.
    with patch.object(
        pose_estimator_instance.pose,
        "process",
        return_value=MagicMock(pose_landmarks=None),
    ) as mock_process:
        annotated_image, results = pose_estimator_instance.process_frame(
            dummy_image.copy()
        )

        # Verifica se o método process do MediaPipe foi chamado.
        mock_process.assert_called_once()
        # Verifica se a imagem anotada é igual à imagem original (sem desenho de landmarks).
        assert np.array_equal(annotated_image, dummy_image)
        # Verifica se os resultados de landmarks são None, conforme simulado.
        assert results.pose_landmarks is None


# Teste para verificar o processamento de um frame com landmarks detectados (simulado).
def test_process_frame_with_landmarks(pose_estimator_instance, dummy_image):
    """
    Testa o método process_frame com uma simulação de landmarks detectados.
    Deve retornar uma imagem anotada (diferente da original) e resultados com landmarks.
    """
    # Cria um mock para simular os landmarks do MediaPipe.
    mock_landmark = MagicMock()
    mock_landmark.x = 0.5
    mock_landmark.y = 0.5
    mock_landmark.z = 0.0
    mock_landmark.visibility = 0.9

    mock_pose_landmarks = MagicMock()
    mock_pose_landmarks.landmark = [
        mock_landmark
    ] * 33  # 33 landmarks do MediaPipe Pose

    mock_results = MagicMock(pose_landmarks=mock_pose_landmarks)

    # Usamos patch para simular o comportamento do MediaPipe para retornar landmarks.
    with patch.object(
        pose_estimator_instance.pose, "process", return_value=mock_results
    ) as mock_process:
        # Usamos patch também para mp_drawing.draw_landmarks, pois não queremos testar a renderização do MP,
        # mas sim que ela seja chamada quando landmarks estão presentes.
        with patch.object(
            pose_estimator_instance.mp_drawing, "draw_landmarks"
        ) as mock_draw_landmarks:
            annotated_image, results = pose_estimator_instance.process_frame(
                dummy_image.copy()
            )

            mock_process.assert_called_once()
            # Verifica se a imagem anotada é diferente da original (indicando que algo foi desenhado).
            # Isso é uma verificação básica, pois 'draw_landmarks' realmente altera a imagem.
            assert not np.array_equal(annotated_image, dummy_image)
            assert results.pose_landmarks is not None
            assert len(results.pose_landmarks.landmark) == 33
            # Verifica se draw_landmarks foi chamado.
            mock_draw_landmarks.assert_called_once_with(
                annotated_image,
                mock_pose_landmarks,
                pose_estimator_instance.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_estimator_instance.mp_drawing_styles.get_default_pose_landmarks_style(),
            )


# Teste para verificar se o método close é chamado corretamente.
def test_close_method(pose_estimator_instance):
    """
    Testa se o método close() do PoseEstimator chama o método close() do modelo MediaPipe.
    """
    with patch.object(pose_estimator_instance.pose, "close") as mock_pose_close:
        pose_estimator_instance.close()
        mock_pose_close.assert_called_once()


# Teste para verificar o destrutor (del)
def test_del_method():
    """
    Testa se o destrutor (__del__) do PoseEstimator chama o close() do modelo MediaPipe.
    Isso é um pouco mais tricky de testar diretamente, pois depende do GC.
    Criamos um mock Pose e garantimos que o close seja chamado quando a referência é perdida.
    """
    mock_pose_model = MagicMock()
    with patch("mediapipe.solutions.pose.Pose", return_value=mock_pose_model):
        estimator = PoseEstimator()
        # Força a coleta de lixo para chamar __del__
        del estimator
        mock_pose_model.close.assert_called_once()
