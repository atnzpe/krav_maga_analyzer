# src/main_flet.py

import flet as ft
import logging
import asyncio  # Importa a biblioteca asyncio para operações assíncronas e de thread
import os
import cv2  # Necessário para operações com vídeo, mesmo que de forma indireta aqui.
import numpy as np  # Necessário para manipulação de arrays, comum em processamento de vídeo.
import tempfile  # Para lidar com arquivos temporários
import threading  # Se houver uso de threads em segundo plano (além de asyncio.to_thread)
import time  # Para operações de tempo, como sleeps ou contadores
import io  # Para trabalhar com streams de bytes em memória
import base64  # Para codificar frames de imagem para exibição no Flet

# IMPORTANTE: Adicione estas linhas no topo para resolver ModuleNotFoundError
import sys

# Adiciona o diretório raiz do projeto ao sys.path. Isso é crucial para que as importações
# de módulos locais, como `src.utils` e `src.video_analyzer`, funcionem corretamente
# quando o script é executado de diferentes diretórios.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa as classes e funções necessárias dos módulos locais
from src.utils import (
    setup_logging,  # Função para configurar o sistema de log da aplicação.
    get_logger,  # Função para obter uma instância do logger.
    FeedbackManager,  # AGORA IMPORTADA DO utils.py
    # VideoProcessor, # Mantenha ou remova conforme a necessidade real do seu projeto.
)
from src.video_analyzer import (
    VideoAnalyzer,  # Importa a classe `VideoAnalyzer`, responsável por processar vídeos e detectar poses.
)
from src.motion_comparator import (
    MotionComparator,  # Importa a classe MotionComparator para comparação de movimentos
)

# Configura o logger para este módulo
setup_logging()
logger = get_logger(__name__)  # Obtém o logger para o módulo atual.

# Variáveis globais (ou de sessão) para armazenar os caminhos dos arquivos e dados processados.
# Estas variáveis são acessíveis e modificáveis por qualquer função dentro do módulo.
VIDEO_ALUNO_PATH = None  # Armazena o caminho completo para o arquivo de vídeo do aluno. Inicialmente `None`.
VIDEO_MESTRE_PATH = None  # Armazena o caminho completo para o arquivo de vídeo do mestre. Inicialmente `None`.
PROCESSED_FRAMES_ALUNO = (
    []
)  # Lista para armazenar frames processados do vídeo do aluno.
PROCESSED_FRAMES_MESTRE = (
    []
)  # Lista para armazenar frames processados do vídeo do mestre.
VIDEO_ANALYZER = (
    VideoAnalyzer()
)  # Instância global do VideoAnalyzer para gerenciar o processamento de vídeo e comparação.

# Dicionários para controlar o estado da reprodução dos vídeos.
# Útil para pausar/reproduzir e manter o controle do frame atual.
VIDEO_PLAYER_STATE = {
    "aluno": {"playing": False, "thread": None, "current_frame_index": 0},
    "mestre": {"playing": False, "thread": None, "current_frame_index": 0},
}

# Instância do gerenciador de feedback.
# Será inicializada corretamente dentro da função `main` com o controle `ft.Text`.
feedback_manager = FeedbackManager()  # Inicializa sem o controle de texto aqui.

# Instância da página Flet (será definida na função main)
page_instance = None  # Variável global para armazenar a instância da página Flet.


async def pick_file_result_aluno(e: ft.FilePickerResultEvent):
    """
    Manipula o resultado da seleção de arquivo para o vídeo do aluno.

    Args:
        e (ft.FilePickerResultEvent): Evento de resultado do seletor de arquivos.
    """
    global page_instance
    if e.files:
        selected_file = e.files[0]
        logger.info(f"Arquivo do aluno selecionado: {selected_file.name}")
        feedback_manager.update_feedback(
            page_instance, f"Carregando vídeo do aluno: {selected_file.name}..."
        )
        try:
            # Usa asyncio.to_thread para executar a leitura do arquivo (operação bloqueante)
            # em um thread separado, evitando que a UI congele.
            # page_instance.run_task() espera uma coroutine, e asyncio.to_thread retorna uma.
            file_bytes = await page_instance.run_task(
                lambda: asyncio.to_thread(lambda: open(selected_file.path, "rb").read())
            )
            logger.info("Vídeo do aluno lido com sucesso.")
            feedback_manager.update_feedback(
                page_instance, "Vídeo do aluno carregado com sucesso!"
            )
            # Aqui você pode processar 'file_bytes'
            # Exemplo: Salvar temporariamente ou passar para o processador de vídeo
            # video_processor.load_video_aluno(file_bytes) # Exemplo de uso
        except Exception as ex:
            logger.error(f"Erro ao ler vídeo do aluno: {ex}")
            feedback_manager.update_feedback(
                page_instance, f"Erro ao carregar vídeo do aluno: '{ex}'", is_error=True
            )
    else:
        logger.info("Seleção de arquivo do aluno cancelada.")
        feedback_manager.update_feedback(
            page_instance, "Seleção de vídeo do aluno cancelada."
        )
    page_instance.update()  # Atualiza a página para exibir o feedback


async def pick_file_result_mestre(e: ft.FilePickerResultEvent):
    """
    Manipula o resultado da seleção de arquivo para o vídeo do mestre.

    Args:
        e (ft.FilePickerResultEvent): Evento de resultado do seletor de arquivos.
    """
    global page_instance
    if e.files:
        selected_file = e.files[0]
        logger.info(f"Arquivo do mestre selecionado: {selected_file.name}")
        feedback_manager.update_feedback(
            page_instance, f"Carregando vídeo do mestre: {selected_file.name}..."
        )
        try:
            # Usa asyncio.to_thread para executar a leitura do arquivo (operação bloqueante)
            # em um thread separado, evitando que a UI congele.
            # page_instance.run_task() espera uma coroutine, e asyncio.to_thread retorna uma.
            file_bytes = await page_instance.run_task(
                lambda: asyncio.to_thread(lambda: open(selected_file.path, "rb").read())
            )
            logger.info("Vídeo do mestre lido com sucesso.")
            feedback_manager.update_feedback(
                page_instance, "Vídeo do mestre carregado com sucesso!"
            )
            # Aqui você pode processar 'file_bytes'
            # Exemplo: Salvar temporariamente ou passar para o processador de vídeo
            # video_processor.load_video_mestre(file_bytes) # Exemplo de uso
        except Exception as ex:
            logger.error(f"Erro ao ler vídeo do mestre: {ex}")
            feedback_manager.update_feedback(
                page_instance,
                f"Erro ao carregar vídeo do mestre: '{ex}'",
                is_error=True,
            )
    else:
        logger.info("Seleção de arquivo do mestre cancelada.")
        feedback_manager.update_feedback(
            page_instance, "Seleção de vídeo do mestre cancelada."
        )
    page_instance.update()  # Atualiza a página para exibir o feedback


def main(page: ft.Page):
    """
    Função principal da aplicação Flet.
    Define a interface do usuário e a lógica de interação.
    """
    global page_instance  # Declara que estamos usando a variável global
    page_instance = page  # Atribui a instância da página à variável global

    page.title = "Analisador de Movimentos de Krav Maga"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1200
    page.window_height = 800
    page.bgcolor = ft.colors.BLUE_GREY_50
    page.theme_mode = ft.ThemeMode.LIGHT  # Modo claro
    logger.info("Iniciando a aplicação Flet e configurando a página.")

    # Inicialização do FeedbackManager com o controle de texto da UI
    # Isso garante que o FeedbackManager tenha acesso ao controle de texto do Flet
    # criado dentro da função main.
    feedback_text_container = ft.Text(
        value="Bem-vindo ao Analisador de Movimentos de Krav Maga!",
        color=ft.colors.BLACK,
        size=16,
        weight=ft.FontWeight.NORMAL,
        text_align=ft.TextAlign.CENTER,
        width=800,  # Largura para o container de feedback
        height=100,  # Altura para o container de feedback
        max_lines=5,
        overflow=ft.TextOverflow.ELLIPSIS,
    )
    # Define o controle de feedback na instância global do FeedbackManager
    feedback_manager.set_feedback_control(feedback_text_container)
    logger.info("Controle de feedback na UI associado ao FeedbackManager.")

    # Criar instâncias do FilePicker
    file_picker_aluno = ft.FilePicker(on_result=pick_file_result_aluno)
    file_picker_mestre = ft.FilePicker(on_result=pick_file_result_mestre)

    # Adicionar FilePicker à sobreposição da página
    page.overlay.append(file_picker_aluno)
    page.overlay.append(file_picker_mestre)
    logger.info("FilePickers adicionados à sobreposição da página.")

    # Botões para selecionar arquivos
    btn_select_aluno = ft.ElevatedButton(
        "Selecionar Vídeo do Aluno",
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: file_picker_aluno.pick_files(
            allow_multiple=False, allowed_extensions=["mp4", "avi", "mov", "mkv"]
        ),
    )
    logger.info("Botão 'Selecionar Vídeo do Aluno' criado.")

    btn_select_mestre = ft.ElevatedButton(
        "Selecionar Vídeo do Mestre",
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: file_picker_mestre.pick_files(
            allow_multiple=False, allowed_extensions=["mp4", "avi", "mov", "mkv"]
        ),
    )
    logger.info("Botão 'Selecionar Vídeo do Mestre' criado.")

    # Botão para iniciar análise
    btn_start_analysis = ft.ElevatedButton(
        "Iniciar Análise",
        icon=ft.icons.ANALYTICS,
        on_click=lambda _: feedback_manager.update_feedback(
            page, "Análise iniciada (funcionalidade em desenvolvimento)..."
        ),
        disabled=True,  # Desabilitado até que os vídeos sejam carregados
    )
    logger.info("Botão 'Iniciar Análise' criado (desabilitado).")

    # Layout principal da UI
    page.add(
        ft.Column(
            [
                ft.AppBar(
                    title=ft.Text("Krav Maga Analyzer", color=ft.colors.WHITE),
                    bgcolor=ft.colors.BLUE_GREY_700,
                    center_title=True,
                ),
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        "Vídeo do Aluno",
                                        size=20,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Container(
                                        content=ft.Image(
                                            src="https://via.placeholder.com/320x240?text=Video+Aluno",
                                            fit=ft.ImageFit.CONTAIN,
                                            border_radius=ft.border_radius.all(10),
                                        ),
                                        width=320,
                                        height=240,
                                        bgcolor=ft.colors.GREY_300,
                                        alignment=ft.alignment.center,
                                        border_radius=ft.border_radius.all(10),
                                    ),
                                    btn_select_aluno,
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                spacing=10,
                            ),
                            padding=20,
                            margin=10,
                            bgcolor=ft.colors.WHITE,
                            border_radius=ft.border_radius.all(15),
                            shadow=ft.BoxShadow(
                                spread_radius=1,
                                blur_radius=10,
                                color=ft.colors.BLACK_26,
                                offset=ft.Offset(0, 0),
                            ),
                        ),
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Text(
                                        "Vídeo do Mestre",
                                        size=20,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Container(
                                        content=ft.Image(
                                            src="https://via.placeholder.com/320x240?text=Video+Mestre",
                                            fit=ft.ImageFit.CONTAIN,
                                            border_radius=ft.border_radius.all(10),
                                        ),
                                        width=320,
                                        height=240,
                                        bgcolor=ft.colors.GREY_300,
                                        alignment=ft.alignment.center,
                                        border_radius=ft.border_radius.all(10),
                                    ),
                                    btn_select_mestre,
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                spacing=10,
                            ),
                            padding=20,
                            margin=10,
                            bgcolor=ft.colors.WHITE,
                            border_radius=ft.border_radius.all(15),
                            shadow=ft.BoxShadow(
                                spread_radius=1,
                                blur_radius=10,
                                color=ft.colors.BLACK_26,
                                offset=ft.Offset(0, 0),
                            ),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=30,
                ),
                ft.Container(
                    content=feedback_text_container,  # Usa o controle de texto para feedback
                    alignment=ft.alignment.center,
                    padding=10,
                    margin=ft.margin.only(top=20),
                    width=800,
                    height=100,
                    bgcolor=ft.colors.WHITE,
                    border_radius=ft.border_radius.all(10),
                    shadow=ft.BoxShadow(
                        spread_radius=1,
                        blur_radius=5,
                        color=ft.colors.BLACK_12,
                        offset=ft.Offset(0, 0),
                    ),
                ),
                ft.Container(
                    content=btn_start_analysis,
                    alignment=ft.alignment.center,
                    padding=10,
                    margin=ft.margin.only(top=20),
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        )
    )
    page.update()  # Atualiza a página para exibir todos os elementos da UI
    logger.info("Elementos da UI Flet adicionados e página atualizada.")


if __name__ == "__main__":
    ft.app(target=main)
