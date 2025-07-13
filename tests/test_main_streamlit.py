import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import os
import logging

# Configuração de logging para os testes, para não interferir com o logging da aplicação.
logging.getLogger().setLevel(logging.CRITICAL)


# Teste básico para verificar se a aplicação Streamlit carrega sem erros.
def test_streamlit_app_loads():
    """
    Testa se a aplicação Streamlit carrega e exibe o título principal.
    Este é um teste de fumaça para garantir que o Streamlit está configurado corretamente.
    """
    # Cria uma instância de AppTest para simular a execução da aplicação.
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    # Executa a aplicação.
    at.run()

    # Verifica se a execução foi bem-sucedida (sem erros).
    assert (
        at.exception is None
    ), f"A aplicação Streamlit falhou ao carregar: {at.exception}"
    # Verifica se o título principal esperado está presente na saída.
    assert (
        "Analisador de Movimentos de Krav Maga (Streamlit)" in at.title[0].body
    ), "Título da aplicação Streamlit não encontrado."


# Teste para verificar o comportamento de upload de arquivos.
def test_streamlit_file_upload_message():
    """
    Testa se a mensagem de aviso é exibida corretamente quando os vídeos não são carregados.
    """
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    at.run()

    # Verifica se a mensagem de aviso é exibida quando nenhum vídeo é carregado.
    assert (
        "Por favor, carregue ambos os vídeos para iniciar a análise."
        in at.warning[0].body
    ), "Mensagem de aviso de upload de vídeo não encontrada."


# Teste para verificar a exibição da mensagem de sucesso após o upload (simulado).
def test_streamlit_success_message_after_simulated_upload():
    """
    Testa se a mensagem de sucesso é exibida quando ambos os vídeos são "carregados" (simulados).
    """
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    at.run(
        uploaded_file_manager={
            "aluno_video_uploader": b"dummy_video_data_aluno",
            "mestre_video_uploader": b"dummy_video_data_mestre",
        }
    )

    # Verifica se a mensagem de sucesso é exibida.
    assert (
        "Ambos os vídeos foram carregados com sucesso! 🎉" in at.success[0].body
    ), "Mensagem de sucesso de upload de vídeo não encontrada."


# Teste para verificar se o botão "Analisar Movimentos" aparece após o upload.
def test_streamlit_analyze_button_visibility_after_upload():
    """
    Testa se o botão 'Analisar Movimentos' se torna visível após a simulação de upload dos vídeos.
    """
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    at.run(
        uploaded_file_manager={
            "aluno_video_uploader": b"dummy_video_data_aluno",
            "mestre_video_uploader": b"dummy_video_data_mestre",
        }
    )
    # Verifica se o botão "Analisar Movimentos" existe.
    assert (
        at.button[0].label == "Analisar Movimentos"
    ), "Botão 'Analisar Movimentos' não encontrado."


# Teste para verificar a mensagem de funcionalidade em desenvolvimento após clicar no botão de análise.
def test_streamlit_analysis_in_progress_message():
    """
    Testa se a mensagem de "funcionalidade em desenvolvimento" é exibida
    após clicar no botão 'Analisar Movimentos'.
    """
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    at.run(
        uploaded_file_manager={
            "aluno_video_uploader": b"dummy_video_data_aluno",
            "mestre_video_uploader": b"dummy_video_data_mestre",
        }
    )
    # Clica no botão "Analisar Movimentos".
    at.button[0].click().run()

    # Verifica se a mensagem de informação sobre a análise em desenvolvimento é exibida.
    assert (
        "Iniciando a análise dos vídeos. Isso pode levar alguns minutos, por favor aguarde..."
        in at.info[0].body
    ), "Mensagem de análise em desenvolvimento não encontrada."
