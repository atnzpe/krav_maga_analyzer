import pytest
import cv2
import numpy as np
import os
import sys

# IMPORTANTE: Adiciona o diretório raiz do projeto ao sys.path para importações
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose_estimator import PoseEstimator


@pytest.fixture
def pose_estimator_instance():
    """Fixture para fornecer uma instância de PoseEstimator para os testes."""
    return PoseEstimator()


@pytest.fixture
def dummy_image():
    """Fixture para fornecer uma imagem de teste simples (quadro branco)."""
    return np.zeros((480, 640, 3), dtype=np.uint8) + 255  # Imagem branca 640x480


def test_pose_estimator_initialization(pose_estimator_instance):
    """
    Testa se o PoseEstimator pode ser inicializado corretamente.
    """
    assert pose_estimator_instance is not None
    assert hasattr(pose_estimator_instance, "mp_pose")
    assert hasattr(pose_estimator_instance, "pose")


def test_process_frame_no_person(pose_estimator_instance, dummy_image):
    """
    Testa o processamento de um frame sem pessoa (apenas fundo branco).
    Deve retornar o frame inalterado e lista de landmarks vazia.
    """
    annotated_frame, landmarks = pose_estimator_instance.process_frame(dummy_image)

    # Verifica se o frame retornado é do mesmo tamanho e tipo
    assert annotated_frame.shape == dummy_image.shape
    assert annotated_frame.dtype == dummy_image.dtype

    # Verifica se os landmarks estão vazios (já que não há pessoa)
    assert len(landmarks) == 0

    # Opcional: Verifica se o frame não foi alterado significativamente (exceto logs/overlay vazios)
    # Comparar arrays numpy diretamente pode ser problemático devido a pequenas diferenças de float
    # Mas para um frame branco sem alterações, pode-se tentar:
    assert np.array_equal(annotated_frame, dummy_image)


# Nota: Testar `process_frame` com uma pessoa real exigiria um mock do MediaPipe
# ou a execução real do modelo, o que tornaria o teste lento e dependente de hardware.
# Geralmente, em testes unitários, mockamos as dependências externas.
# Para este projeto, o foco inicial é na integração e fluxo, então um teste de fumaça é aceitável.
