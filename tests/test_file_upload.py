# tests/test_file_upload.py
import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from src.main_flet import (
    pick_file_result_aluno,
    pick_file_result_mestre,
    feedback_manager,
)  # Importe feedback_manager se for global
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
    mock_page.run_task.side_effect = lambda func: asyncio.create_task(func())
    mock_page.update = AsyncMock()
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


@pytest.fixture(autouse=True)
def setup_feedback_manager(mock_page_instance):
    """
    Fixture para configurar o FeedbackManager antes de cada teste.
    Isso é crucial porque o FeedbackManager agora espera um ft.Text.
    """
    # Cria um mock para o controle ft.Text
    mock_feedback_text_control = MagicMock(spec=ft.Text)
    mock_feedback_text_control.page = (
        mock_page_instance  # Garante que o .page seja acessível
    )
    feedback_manager.set_feedback_control(mock_feedback_text_control)


@pytest.mark.asyncio
async def test_pick_file_result_aluno_success(
    mock_page_instance, create_dummy_video_file
):
    """
    Testa o carregamento bem-sucedido do vídeo do aluno.
    """
    global page_instance  # Acessa a variável global page_instance no main_flet.py
    page_instance = mock_page_instance  # Define a instância global para o teste

    dummy_file_path = create_dummy_video_file()

    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = [
        MagicMock(
            spec=ft.FilePickerFile, name="aluno_video.mp4", path=str(dummy_file_path)
        )
    ]

    await pick_file_result_aluno(mock_file_picker_event)

    mock_page_instance.run_task.assert_called_once()
    mock_page_instance.update.assert_called()
    # Verifica se o método update_feedback do feedback_manager foi chamado com a página
    feedback_manager.update_feedback.assert_any_call(
        mock_page_instance, "Vídeo do aluno carregado com sucesso!"
    )


@pytest.mark.asyncio
async def test_pick_file_result_aluno_cancel(mock_page_instance):
    """
    Testa o cancelamento da seleção de arquivo do aluno.
    """
    global page_instance
    page_instance = mock_page_instance

    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = None

    await pick_file_result_aluno(mock_file_picker_event)

    mock_page_instance.run_task.assert_not_called()
    mock_page_instance.update.assert_called_once()
    feedback_manager.update_feedback.assert_any_call(
        mock_page_instance, "Seleção de vídeo do aluno cancelada."
    )


@pytest.mark.asyncio
async def test_pick_file_result_aluno_error_reading(mock_page_instance):
    """
    Testa o cenário de erro durante a leitura do arquivo do aluno.
    """
    global page_instance
    page_instance = mock_page_instance

    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = [
        MagicMock(
            spec=ft.FilePickerFile,
            name="non_existent.mp4",
            path="/path/to/non_existent_file.mp4",
        )
    ]

    await pick_file_result_aluno(mock_file_picker_event)

    mock_page_instance.run_task.assert_called_once()
    mock_page_instance.update.assert_called()
    # Verifica se o feedback de erro foi chamado
    # Usamos assert_called_with para verificar a mensagem exata com is_error=True
    # Como a mensagem de erro pode variar, podemos ser mais flexíveis ou mockar a exceção
    # Para este teste, verificamos que update_feedback foi chamado com is_error=True
    args, kwargs = feedback_manager.update_feedback.call_args
    assert kwargs.get("is_error") is True


@pytest.mark.asyncio
async def test_pick_file_result_mestre_success(
    mock_page_instance, create_dummy_video_file
):
    """
    Testa o carregamento bem-sucedido do vídeo do mestre.
    """
    global page_instance
    page_instance = mock_page_instance

    dummy_file_path = create_dummy_video_file(filename="mestre_video.mp4")

    mock_file_picker_event = MagicMock(spec=ft.FilePickerResultEvent)
    mock_file_picker_event.files = [
        MagicMock(
            spec=ft.FilePickerFile, name="mestre_video.mp4", path=str(dummy_file_path)
        )
    ]

    await pick_file_result_mestre(mock_file_picker_event)

    mock_page_instance.run_task.assert_called_once()
    mock_page_instance.update.assert_called()
    feedback_manager.update_feedback.assert_any_call(
        mock_page_instance, "Vídeo do mestre carregado com sucesso!"
    )
