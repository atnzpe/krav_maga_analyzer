# tests/test_file_upload.py
import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from src.main_flet import pick_file_result_aluno, pick_file_result_mestre
import flet as ft

# Configuração de logging para o teste, se necessário
import logging
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def mock_page_instance():
    """
    Fixture que retorna um mock da instância da página Flet.
    Ajusta o mock para simular o comportamento de run_task e update.
    """
    mock_page = AsyncMock(spec=ft.Page)
    # Mocka run_task para executar a função passada diretamente (para testes síncronos)
    # ou para retornar um awaitable se a função for assíncrona.
    # Para o nosso caso, asyncio.to_thread retorna um awaitable.
    mock_page.run_task.side_effect = lambda func: asyncio.create_task(func())
    mock_page.update = AsyncMock() # Mocka o método update
    return mock_page

@pytest.fixture
def create_dummy_video_file(tmp_path):
    """
    Fixture que cria um arquivo de vídeo dummy para testes.
    """
    def _create_file(filename="test_video.mp4", content=b"dummy video content"):
        file_path = tmp_path / filename
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path
    return _create_file

@pytest.mark.asyncio
async def test_pick_file_result_aluno_success(mock_page_instance, create_dummy_video_file):
    """
    Testa o carregamento bem-sucedido do vídeo do aluno.
    """
    global page_instance
    page_instance = mock_page_instance # Define a instância global para o teste

    # Cria um arquivo dummy
    dummy_file_path = create_dummy_video_file()

    # Cria um mock para FilePickerResultEvent
    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = [
        MagicMock(spec=ft.FilePickerFile, name="aluno_video.mp4", path=str(dummy_file_path))
    ]

    # Chama a função a ser testada
    await pick_file_result_aluno(mock_file_picker_event)

    # Verifica se run_task foi chamado
    mock_page_instance.run_task.assert_called_once()
    # Verifica se update foi chamado para feedback
    mock_page_instance.update.assert_called()

    # Verifica se o feedback de sucesso foi exibido
    # Note: O feedback_manager.update_feedback chama page_instance.update
    # Podemos verificar as chamadas para update ou o estado interno do feedback_manager se ele fosse acessível
    # Para este teste, verificar a chamada de run_task e update é suficiente para a lógica de fluxo.
    # Em um teste de integração, você verificaria a UI diretamente.

@pytest.mark.asyncio
async def test_pick_file_result_aluno_cancel(mock_page_instance):
    """
    Testa o cancelamento da seleção de arquivo do aluno.
    """
    global page_instance
    page_instance = mock_page_instance

    # Cria um mock para FilePickerResultEvent com files vazio (cancelado)
    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = None

    # Chama a função a ser testada
    await pick_file_result_aluno(mock_file_picker_event)

    # Verifica se run_task NÃO foi chamado
    mock_page_instance.run_task.assert_not_called()
    # Verifica se update foi chamado para feedback de cancelamento
    mock_page_instance.update.assert_called_once()

@pytest.mark.asyncio
async def test_pick_file_result_aluno_error_reading(mock_page_instance):
    """
    Testa o cenário de erro durante a leitura do arquivo do aluno.
    """
    global page_instance
    page_instance = mock_page_instance

    # Cria um mock para FilePickerResultEvent com um caminho inválido para forçar erro
    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = [
        MagicMock(spec=ft.FilePickerFile, name="non_existent.mp4", path="/path/to/non_existent_file.mp4")
    ]

    # Chama a função a ser testada
    await pick_file_result_aluno(mock_file_picker_event)

    # Verifica se run_task foi chamado (a tentativa de leitura ainda ocorre)
    mock_page_instance.run_task.assert_called_once()
    # Verifica se update foi chamado para feedback de erro
    mock_page_instance.update.assert_called()

# Os testes para pick_file_result_mestre seriam muito semelhantes,
# apenas mudando a função chamada.
@pytest.mark.asyncio
async def test_pick_file_result_mestre_success(mock_page_instance, create_dummy_video_file):
    """
    Testa o carregamento bem-sucedido do vídeo do mestre.
    """
    global page_instance
    page_instance = mock_page_instance

    dummy_file_path = create_dummy_video_file(filename="mestre_video.mp4")

    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = [
        MagicMock(spec=ft.FilePickerFile, name="mestre_video.mp4", path=str(dummy_file_path))
    ]

    await pick_file_result_mestre(mock_file_picker_event)

    mock_page_instance.run_task.assert_called_once()
    mock_page_instance.update.assert_called()