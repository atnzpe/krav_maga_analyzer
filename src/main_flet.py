import flet as ft
import logging
import os
import cv2
import numpy as np
import tempfile
import threading
import time
import asyncio  # Importa o módulo asyncio para permitir operações assíncronas, como `asyncio.sleep` para controlar o FPS da reprodução de vídeo.
import io  # Importa io para trabalhar com streams de bytes em memória
import base64  # Importa base64 para codificar frames de imagem para exibição no Flet

# IMPORTANTE: Adicione estas linhas no topo para resolver ModuleNotFoundError
import sys

# Adiciona o diretório raiz do projeto ao sys.path. Isso é crucial para que as importações
# de módulos locais, como `src.utils` e `src.video_analyzer`, funcionem corretamente
# quando o script é executado de diferentes diretórios.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (
    setup_logging,
)  # Importa a função `setup_logging` do módulo `utils` para configurar o sistema de log da aplicação.
from src.video_analyzer import (
    VideoAnalyzer,
)  # Importa a classe `VideoAnalyzer` do módulo `video_analyzer`, responsável por processar vídeos e detectar poses.
from src.motion_comparator import (
    MotionComparator,
)  # Importa a classe MotionComparator para comparação de movimentos

logger = (
    setup_logging()
)  # Inicializa o logger da aplicação, configurado para registrar informações e erros.

# Variáveis globais (ou de sessão) para armazenar os caminhos dos arquivos e dados processados.
# Estas variáveis são acessíveis e modificáveis por qualquer função dentro do módulo.
# Alterado para armazenar o conteúdo do vídeo como BytesIO, não o caminho.
VIDEO_ALUNO_CONTENT: io.BytesIO = (
    None  # Armazena o conteúdo do vídeo do aluno como BytesIO
)
VIDEO_MESTRE_CONTENT: io.BytesIO = (
    None  # Armazena o conteúdo do vídeo do mestre como BytesIO
)

PROCESSED_FRAMES_ALUNO = (
    []
)  # Lista para armazenar frames processados (com landmarks) do vídeo do aluno.
PROCESSED_FRAMES_MESTRE = (
    []
)  # Lista para armazenar frames processados (com landmarks) do vídeo do mestre.

CURRENT_FRAME_ALUNO_INDEX = (
    0  # Índice do frame atualmente exibido para o vídeo do aluno.
)
CURRENT_FRAME_MESTRE_INDEX = (
    0  # Índice do frame atualmente exibido para o vídeo do mestre.
)

IS_PLAYING_ALUNO = (
    False  # Flag booleana para controlar o estado de reprodução do vídeo do aluno.
)
IS_PLAYING_MESTRE = (
    False  # Flag booleana para controlar o estado de reprodução do vídeo do mestre.
)

PLAYBACK_THREAD_ALUNO: asyncio.Task = (
    None  # Referência à tarefa assíncrona de reprodução do vídeo do aluno.
)
PLAYBACK_THREAD_MESTRE: asyncio.Task = (
    None  # Referência à tarefa assíncrona de reprodução do vídeo do mestre.
)

FPS = 30  # Assumindo 30 FPS para reprodução. Idealmente, este valor deveria ser extraído do vídeo original.

# --- Controles da UI (inicializados dentro de main) ---
analyze_button: ft.Control = None
upload_button_aluno: ft.Control = None
upload_button_mestre: ft.Control = None
# Alterado para Container para permitir bgcolor
video_aluno_image_container: ft.Container = None
video_mestre_image_container: ft.Container = None
video_aluno_image: ft.Image = (
    None  # Mantém a referência direta para o Image control dentro do Container
)
video_mestre_image: ft.Image = (
    None  # Mantém a referência direta para o Image control dentro do Container
)
feedback_text_container: ft.Container = None
page_instance: ft.Page = None  # Referência para a instância da página Flet
play_button_aluno: ft.IconButton = (
    None  # Adicionado para uso global na função toggle_play
)
play_button_mestre: ft.IconButton = (
    None  # Adicionado para uso global na função toggle_play
)

# --- Funções de callback para eventos da UI ---


def update_analyze_button_state():
    """
    Atualiza o estado do botão 'Analisar Movimentos'.
    Ele será habilitado somente se ambos os vídeos (aluno e mestre) tiverem sido carregados.
    """
    if VIDEO_ALUNO_CONTENT is not None and VIDEO_MESTRE_CONTENT is not None:
        analyze_button.disabled = False
        update_feedback(
            "Ambos os vídeos foram carregados. Clique em 'Analisar Movimentos' para iniciar."
        )
        logger.info("Botão 'Analisar Movimentos' habilitado.")
    else:
        analyze_button.disabled = True
        update_feedback("Por favor, carregue ambos os vídeos para iniciar a análise.")
        logger.info("Botão 'Analisar Movimentos' desabilitado.")
    page_instance.update()


def update_feedback(message: str, is_error: bool = False, is_success: bool = False):
    """
    Atualiza a caixa de texto de feedback na UI com a mensagem fornecida.
    Args:
        message (str): A mensagem a ser exibida.
        is_error (bool): Se True, a mensagem será formatada como um erro (cor vermelha).
        is_success (bool): Se True, a mensagem será formatada como sucesso (cor verde).
    """
    feedback_text_control = feedback_text_container.content
    if isinstance(feedback_text_control, ft.Text):
        feedback_text_control.value = message
        if is_error:
            feedback_text_control.color = ft.Colors.RED_500
        elif is_success:
            feedback_text_control.color = ft.Colors.GREEN_500
        else:
            feedback_text_control.color = ft.Colors.BLACK  # Cor padrão
    page_instance.update()
    logger.info(f"Feedback atualizado: {message}")


def clear_ui_and_analysis_data():
    """
    Limpa os frames processados e redefine os índices e estados de reprodução.
    Também limpa as imagens exibidas na UI.
    """
    global PROCESSED_FRAMES_ALUNO, PROCESSED_FRAMES_MESTRE, CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX, IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, PLAYBACK_THREAD_ALUNO, PLAYBACK_THREAD_MESTRE

    PROCESSED_FRAMES_ALUNO = []
    PROCESSED_FRAMES_MESTRE = []
    CURRENT_FRAME_ALUNO_INDEX = 0
    CURRENT_FRAME_MESTRE_INDEX = 0
    IS_PLAYING_ALUNO = False
    IS_PLAYING_MESTRE = False

    # Parar threads de reprodução se estiverem ativas
    if PLAYBACK_THREAD_ALUNO:
        PLAYBACK_THREAD_ALUNO.cancel()
        PLAYBACK_THREAD_ALUNO = None
    if PLAYBACK_THREAD_MESTRE:
        PLAYBACK_THREAD_MESTRE.cancel()
        PLAYBACK_THREAD_MESTRE = None

    # Limpar imagens exibidas (agora acessando o Image control dentro do Container)
    video_aluno_image.src_base64 = ""
    video_mestre_image.src_base64 = ""
    video_aluno_image_container.update()  # Atualiza o container que contém a imagem
    video_mestre_image_container.update()  # Atualiza o container que contém a imagem

    update_feedback("Dados de análise e UI limpos.")
    logger.info("Dados de análise e UI limpos.")


def set_ui_analysis_state(is_analyzing: bool):
    """
    Define o estado da UI durante a análise de vídeo.
    Desabilita botões e mostra/esconde indicadores de progresso.
    """
    upload_button_aluno.disabled = is_analyzing
    upload_button_mestre.disabled = is_analyzing
    analyze_button.disabled = is_analyzing
    page_instance.update()


async def pick_file_result_aluno(e: ft.FilePickerResultEvent):
    """
    Callback para quando um arquivo de vídeo do aluno é selecionado via FilePicker.
    Lê o conteúdo do arquivo e armazena em VIDEO_ALUNO_CONTENT.
    """
    global VIDEO_ALUNO_CONTENT
    if e.files:
        selected_file = e.files[0]
        try:
            # Leia o conteúdo do arquivo como bytes
            file_bytes = await page_instance.run_thread_task(selected_file.read_bytes)
            VIDEO_ALUNO_CONTENT = io.BytesIO(file_bytes)
            logger.info(f"Vídeo do Aluno selecionado: {selected_file.name}")
            upload_button_aluno.text = f"Vídeo do Aluno: {selected_file.name}"
            clear_ui_and_analysis_data()  # Limpa dados anteriores ao carregar um novo vídeo
            update_analyze_button_state()
        except Exception as ex:
            logger.error(f"Erro ao ler vídeo do aluno: {ex}", exc_info=True)
            update_feedback(f"Erro ao carregar vídeo do aluno: {ex}", is_error=True)
    page_instance.update()


async def pick_file_result_mestre(e: ft.FilePickerResultEvent):
    """
    Callback para quando um arquivo de vídeo do mestre é selecionado via FilePicker.
    Lê o conteúdo do arquivo e armazena em VIDEO_MESTRE_CONTENT.
    """
    global VIDEO_MESTRE_CONTENT
    if e.files:
        selected_file = e.files[0]
        try:
            # Leia o conteúdo do arquivo como bytes
            file_bytes = await page_instance.run_thread_task(selected_file.read_bytes)
            VIDEO_MESTRE_CONTENT = io.BytesIO(file_bytes)
            logger.info(f"Vídeo do Mestre selecionado: {selected_file.name}")
            upload_button_mestre.text = f"Vídeo do Mestre: {selected_file.name}"
            clear_ui_and_analysis_data()  # Limpa dados anteriores ao carregar um novo vídeo
            update_analyze_button_state()
        except Exception as ex:
            logger.error(f"Erro ao ler vídeo do mestre: {ex}", exc_info=True)
            update_feedback(f"Erro ao carregar vídeo do mestre: {ex}", is_error=True)
    page_instance.update()


async def play_video(player_type: str):
    """
    Reproduz o vídeo processado no player especificado (aluno ou mestre).
    Esta é uma coroutine para ser usada com page.run_task().
    """
    global IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX, PROCESSED_FRAMES_ALUNO, PROCESSED_FRAMES_MESTRE, FPS

    # Mapeia as variáveis globais para o tipo de player
    is_playing_var = globals()[f"IS_PLAYING_{player_type.upper()}"]
    current_frame_index_var = globals()[f"CURRENT_FRAME_{player_type.upper()}_INDEX"]
    processed_frames_var = globals()[f"PROCESSED_FRAMES_{player_type.upper()}"]
    video_image_control = globals()[
        f"video_{player_type}_image"
    ]  # Acessa o Image control
    play_button_control = globals()[f"play_button_{player_type}"]

    if not is_playing_var:
        return  # Sair se não estiver reproduzindo

    # Garante que o índice não exceda o número de frames
    if current_frame_index_var >= len(processed_frames_var):
        current_frame_index_var = 0  # Reinicia a reprodução se já estiver no final
        globals()[f"CURRENT_FRAME_{player_type.upper()}_INDEX"] = 0

    while globals()[
        f"IS_PLAYING_{player_type.upper()}"
    ] and current_frame_index_var < len(processed_frames_var):
        annotated_frame, _ = processed_frames_var[current_frame_index_var]

        # Converte o frame OpenCV (numpy array) para bytes PNG para exibição no Flet
        _, buffer = cv2.imencode(
            ".png", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        )
        video_image_control.src_base64 = base64.b64encode(buffer.tobytes()).decode(
            "utf-8"
        )
        await video_image_control.update_async()  # Atualiza a imagem de forma assíncrona

        globals()[f"CURRENT_FRAME_{player_type.upper()}_INDEX"] += 1
        current_frame_index_var = globals()[
            f"CURRENT_FRAME_{player_type.upper()}_INDEX"
        ]  # Atualiza a variável local

        # Controlar a velocidade de reprodução
        await asyncio.sleep(1 / FPS)  # Pausa para simular o FPS do vídeo

    # Após o loop (vídeo terminou ou foi pausado), reseta o estado
    globals()[f"IS_PLAYING_{player_type.upper()}"] = False
    play_button_control.icon = ft.Icons.PLAY_ARROW  # Define o ícone para play
    await play_button_control.update_async()  # Atualiza o botão


async def toggle_play(e, player_type: str):
    """
    Alterna o estado de reprodução/pausa do vídeo e atualiza o ícone do botão.
    Esta é uma coroutine para ser usada como handler de evento.
    """
    global IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, PLAYBACK_THREAD_ALUNO, PLAYBACK_THREAD_MESTRE, play_button_aluno, play_button_mestre

    play_button_control = globals()[f"play_button_{player_type}"]
    is_playing_global_var_name = f"IS_PLAYING_{player_type.upper()}"
    playback_thread_global_var_name = f"PLAYBACK_THREAD_{player_type.upper()}"

    # Alterna o estado de reprodução
    globals()[is_playing_global_var_name] = not globals()[is_playing_global_var_name]

    if globals()[is_playing_global_var_name]:
        # Se começou a reproduzir
        play_button_control.icon = ft.Icons.PAUSE
        # Inicia a tarefa de reprodução usando page.run_task com a coroutine
        globals()[playback_thread_global_var_name] = page_instance.run_task(
            play_video(player_type)
        )
        logger.info(f"Iniciando reprodução do vídeo {player_type}.")
    else:
        # Se pausou
        play_button_control.icon = ft.Icons.PLAY_ARROW
        # Cancela a tarefa de reprodução se ela existir
        if globals()[playback_thread_global_var_name]:
            globals()[playback_thread_global_var_name].cancel()
            globals()[playback_thread_global_var_name] = None
        logger.info(f"Pausando reprodução do vídeo {player_type}.")

    await play_button_control.update_async()


async def analyze_videos(e):
    """
    Função principal que orquestra a análise dos vídeos do aluno e do mestre.
    Chama o VideoAnalyzer para processar os frames e o MotionComparator para comparar.
    """
    global PROCESSED_FRAMES_ALUNO, PROCESSED_FRAMES_MESTRE, CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX, IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, VIDEO_ALUNO_CONTENT, VIDEO_MESTRE_CONTENT, PLAYBACK_THREAD_ALUNO, PLAYBACK_THREAD_MESTRE

    # Verifica se os conteúdos dos vídeos foram carregados
    if VIDEO_ALUNO_CONTENT is None or VIDEO_MESTRE_CONTENT is None:
        update_feedback(
            "Por favor, carregue ambos os vídeos para iniciar a análise.", is_error=True
        )
        return

    set_ui_analysis_state(True)
    update_feedback(
        "Iniciando a análise dos vídeos. Isso pode levar alguns minutos, por favor aguarde..."
    )

    # Garante que os BytesIO estejam no início para leitura
    VIDEO_ALUNO_CONTENT.seek(0)
    VIDEO_MESTRE_CONTENT.seek(0)

    try:
        # Limpar dados anteriores
        clear_ui_and_analysis_data()
        update_feedback("Limpando dados anteriores...")

        video_analyzer = VideoAnalyzer()

        # Processar Vídeo do Aluno
        update_feedback("Processando vídeo do Aluno...")
        # Use o BytesIO diretamente. O VideoAnalyzer deve saber como lidar com isso.
        for annotated_frame, landmarks_data in video_analyzer.analyze_video(
            VIDEO_ALUNO_CONTENT
        ):
            PROCESSED_FRAMES_ALUNO.append((annotated_frame, landmarks_data))
        logger.info(
            f"Processamento do vídeo do Aluno concluído. Total de frames: {len(PROCESSED_FRAMES_ALUNO)}"
        )
        update_feedback(
            f"Processamento do vídeo do Aluno concluído. Total de frames: {len(PROCESSED_FRAMES_ALUNO)}"
        )

        # Processar Vídeo do Mestre
        update_feedback("Processando vídeo do Mestre...")
        for annotated_frame, landmarks_data in video_analyzer.analyze_video(
            VIDEO_MESTRE_CONTENT
        ):
            PROCESSED_FRAMES_MESTRE.append((annotated_frame, landmarks_data))
        logger.info(
            f"Processamento do vídeo do Mestre concluído. Total de frames: {len(PROCESSED_FRAMES_MESTRE)}"
        )
        update_feedback(
            f"Processamento do vídeo do Mestre concluído. Total de frames: {len(PROCESSED_FRAMES_MESTRE)}"
        )

        # Comparar movimentos (usando os dados de landmarks armazenados no VideoAnalyzer)
        update_feedback("Comparando movimentos...")
        # A classe VideoAnalyzer agora gerencia o armazenamento dos landmarks
        # e tem um método para comparar_movimentos.
        raw_comparison_results, feedback_list = (
            video_analyzer.compare_processed_movements()
        )
        logger.info(
            f"Comparação de movimentos concluída. {len(feedback_list)} itens de feedback gerados."
        )

        # Exibir feedback
        if feedback_list:
            full_feedback = "\n".join(feedback_list)
            update_feedback(
                f"Análise concluída!\n\nFeedback:\n{full_feedback}", is_success=True
            )
        else:
            update_feedback(
                "Análise concluída! Nenhuma diferença significativa detectada. Boa execução!",
                is_success=True,
            )

        # Exibir o primeiro frame processado de cada vídeo
        if PROCESSED_FRAMES_ALUNO:
            first_frame_aluno, _ = PROCESSED_FRAMES_ALUNO[0]
            _, buffer_aluno = cv2.imencode(
                ".png", cv2.cvtColor(first_frame_aluno, cv2.COLOR_BGR2RGB)
            )
            video_aluno_image.src_base64 = base64.b64encode(
                buffer_aluno.tobytes()
            ).decode("utf-8")
        if PROCESSED_FRAMES_MESTRE:
            first_frame_mestre, _ = PROCESSED_FRAMES_MESTRE[0]
            _, buffer_mestre = cv2.imencode(
                ".png", cv2.cvtColor(first_frame_mestre, cv2.COLOR_BGR2RGB)
            )
            video_mestre_image.src_base64 = base64.b64encode(
                buffer_mestre.tobytes()
            ).decode("utf-8")

        # Atualiza a UI para exibir os primeiros frames e o feedback
        page_instance.update()

    except Exception as ex:
        logger.error(f"Erro durante a análise de vídeo: {ex}", exc_info=True)
        update_feedback(f"Ocorreu um erro durante a análise: {ex}", is_error=True)
    finally:
        set_ui_analysis_state(False)


def main(page: ft.Page):
    """
    Função principal que define a interface do usuário (UI) do aplicativo Flet.
    Configura a página, adiciona controles e gerencia a disposição dos elementos.
    """
    global analyze_button, upload_button_aluno, upload_button_mestre, video_aluno_image_container, video_mestre_image_container, feedback_text_container, page_instance, play_button_aluno, play_button_mestre, video_aluno_image, video_mestre_image

    page_instance = page  # Armazena a referência da página

    logger.info("Iniciando a aplicação Flet...")

    page.title = "Analisador de Movimentos de Krav Maga (Flet)"
    page.vertical_alignment = (
        ft.MainAxisAlignment.START
    )  # Alinha o conteúdo ao topo verticalmente.
    page.horizontal_alignment = (
        ft.CrossAxisAlignment.CENTER
    )  # Centraliza o conteúdo horizontalmente.
    page.window_width = 1000
    page.window_height = 800
    page.window_resizable = True
    page.scroll = ft.ScrollMode.ADAPTIVE  # Adiciona scroll se o conteúdo exceder a tela

    # Configurações do FilePicker
    file_picker_aluno = ft.FilePicker(on_result=pick_file_result_aluno)
    file_picker_mestre = ft.FilePicker(on_result=pick_file_result_mestre)
    page.overlay.append(file_picker_aluno)
    page.overlay.append(file_picker_mestre)

    # --- Elementos da UI ---

    # Botões de Upload
    upload_button_aluno = ft.ElevatedButton(
        text="Upload Vídeo do Aluno",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: file_picker_aluno.pick_files(
            allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"]
        ),
        tooltip="Carregar vídeo do aluno para análise.",
    )

    upload_button_mestre = ft.ElevatedButton(
        text="Upload Vídeo do Mestre",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: file_picker_mestre.pick_files(
            allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"]
        ),
        tooltip="Carregar vídeo do mestre para análise.",
    )

    # Botão de Análise
    analyze_button = ft.ElevatedButton(
        text="Analisar Movimentos",
        icon=ft.Icons.ANALYTICS,
        on_click=analyze_videos,
        disabled=True,  # Começa desabilitado até que ambos os vídeos sejam carregados.
        tooltip="Inicia a análise comparativa dos movimentos.",
    )

    # Controles de Imagem para exibição de vídeo
    # Cria o Image control
    video_aluno_image = ft.Image(
        src_base64="",  # Inicialmente vazio
        fit=ft.ImageFit.CONTAIN,
        border_radius=ft.border_radius.all(10),
    )
    # Cria o Container que envolve o Image control e define o background
    video_aluno_image_container = ft.Container(
        content=video_aluno_image,
        width=400,
        height=300,
        bgcolor=ft.Colors.BLACK,  # Fundo preto para as áreas de vídeo
        alignment=ft.alignment.center,  # Centraliza a imagem dentro do container
        border_radius=ft.border_radius.all(10),
    )

    video_mestre_image = ft.Image(
        src_base64="",  # Inicialmente vazio
        fit=ft.ImageFit.CONTAIN,
        border_radius=ft.border_radius.all(10),
    )
    video_mestre_image_container = ft.Container(
        content=video_mestre_image,
        width=400,
        height=300,
        bgcolor=ft.Colors.BLACK,  # Fundo preto para as áreas de vídeo
        alignment=ft.alignment.center,  # Centraliza a imagem dentro do container
        border_radius=ft.border_radius.all(10),
    )

    # Botões de Reprodução
    play_button_aluno = ft.IconButton(
        icon=ft.Icons.PLAY_ARROW,
        on_click=lambda e: page.run_task(
            toggle_play(e, "aluno")
        ),  # Chama toggle_play como uma coroutine
        tooltip="Reproduzir/Pausar vídeo do aluno",
        icon_size=30,
    )
    play_button_mestre = ft.IconButton(
        icon=ft.Icons.PLAY_ARROW,
        on_click=lambda e: page.run_task(
            toggle_play(e, "mestre")
        ),  # Chama toggle_play como uma coroutine
        tooltip="Reproduzir/Pausar vídeo do mestre",
        icon_size=30,
    )

    # Container para feedback textual
    feedback_text_container = ft.Container(
        content=ft.Text(
            "Por favor, carregue ambos os vídeos para iniciar a análise.",
            selectable=True,
            color=ft.Colors.BLACK,
        ),
        width=900,
        height=200,
        bgcolor=ft.Colors.BLUE_GREY_50,
        padding=ft.padding.all(10),
        border_radius=ft.border_radius.all(10),
        alignment=ft.alignment.top_left,
    )

    # Adiciona todos os elementos à página
    page.add(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Image(
                            src="assets/logo_FBKKLN.png",
                            width=100,
                            height=100,
                            fit=ft.ImageFit.CONTAIN,
                        ),
                        ft.Text(
                            "Analisador de Movimentos de Krav Maga",
                            size=28,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_GREY_900,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                ft.Row(
                    [
                        upload_button_aluno,
                        upload_button_mestre,
                        analyze_button,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=20,
                ),
                ft.Row(
                    [
                        # Coluna para Vídeo do Aluno
                        ft.Column(
                            [
                                ft.Text(
                                    "Vídeo do Aluno",
                                    size=18,
                                    weight=ft.FontWeight.NORMAL,
                                ),
                                ft.Stack(  # Usa Stack para sobrepor imagem e slider
                                    [
                                        video_aluno_image_container,  # Usa o Container aqui
                                        # ft.Slider (adicionar se for usar slider de frames)
                                    ],
                                    width=400,
                                    height=300,
                                ),
                                ft.Row(
                                    [
                                        play_button_aluno,
                                        # ft.IconButton(
                                        #     icon=ft.Icons.STOP, # Corrigido para ft.Icons
                                        #     on_click=lambda e: stop_video("aluno"),
                                        #     tooltip="Parar vídeo do aluno",
                                        #     icon_size=30,
                                        # ),
                                    ],
                                    alignment=ft.MainAxisAlignment.CENTER,
                                ),
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            alignment=ft.MainAxisAlignment.START,
                        ),
                        # Coluna para Vídeo do Mestre
                        ft.Column(
                            [
                                ft.Text(
                                    "Vídeo do Mestre",
                                    size=18,
                                    weight=ft.FontWeight.NORMAL,
                                ),
                                ft.Stack(  # Usa Stack para sobrepor imagem e slider
                                    [
                                        video_mestre_image_container,  # Usa o Container aqui
                                        # ft.Slider (adicionar se for usar slider de frames)
                                    ],
                                    width=400,
                                    height=300,
                                ),
                                ft.Row(
                                    [
                                        play_button_mestre,
                                        # ft.IconButton(
                                        #     icon=ft.Icons.STOP, # Corrigido para ft.Icons
                                        #     on_click=lambda e: stop_video("mestre"),
                                        #     tooltip="Parar vídeo do mestre",
                                        #     icon_size=30,
                                        # ),
                                    ],
                                    alignment=ft.MainAxisAlignment.CENTER,
                                ),
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            alignment=ft.MainAxisAlignment.START,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,  # Distribui o espaço igualmente ao redor dos itens na linha.
                    width=900,  # Largura da linha.
                ),
                ft.Text(
                    "Resultados da Análise e Feedback:",
                    size=18,
                    weight=ft.FontWeight.NORMAL,
                ),
                feedback_text_container,  # O container onde o feedback será exibido.
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Centraliza o conteúdo da coluna principal horizontalmente.
            spacing=10,  # Espaçamento entre os elementos da coluna.
            expand=True,  # Permite que a coluna se expanda para preencher o espaço disponível.
        )
    )
    page.update()  # Atualiza a página para renderizar todos os elementos da UI.
    logger.info("Elementos da UI Flet adicionados e página atualizada.")


# Ponto de entrada da aplicação Flet.
if __name__ == "__main__":
    # Inicia a aplicação Flet, apontando para a função 'main'.
    # O Flet cuida do ciclo de vida da aplicação, criando a janela e executando a função `main`.
    ft.app(target=main)
