import flet as ft
import logging
import os
import cv2
import numpy as np
import tempfile
import threading
import time

# IMPORTANTE: Adicione estas linhas no topo para resolver ModuleNotFoundError
import sys

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import setup_logging
from src.video_analyzer import VideoAnalyzer

logger = setup_logging()

# Variáveis globais (ou de sessão) para armazenar os caminhos dos arquivos e dados processados
VIDEO_ALUNO_PATH = None
VIDEO_MESTRE_PATH = None
PROCESSED_FRAMES_ALUNO = []
PROCESSED_FRAMES_MESTRE = []
CURRENT_FRAME_ALUNO_INDEX = 0
CURRENT_FRAME_MESTRE_INDEX = 0
IS_PLAYING_ALUNO = False
IS_PLAYING_MESTRE = False
PLAYBACK_THREAD_ALUNO = None
PLAYBACK_THREAD_MESTRE = None
FPS = 30  # Assumindo 30 FPS para reprodução, idealmente extraído do vídeo original


def main(page: ft.Page):
    logger.info("Iniciando a aplicação Flet...")

    page.title = "Analisador de Movimentos de Krav Maga (Flet)"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window_width = 1200
    page.window_height = 800
    page.scroll = "adaptive"

    logger.info("Configurações da página Flet aplicadas.")

    # Elementos da UI que serão atualizados
    status_text = ft.Text("Status: Aguardando upload de vídeos...", key="status_text")

    img_aluno = ft.Image(src_base64="", width=400, height=300, fit=ft.ImageFit.CONTAIN)
    img_mestre = ft.Image(src_base64="", width=400, height=300, fit=ft.ImageFit.CONTAIN)

    slider_aluno = ft.Slider(
        min=0, max=0, divisions=1, value=0, label="{value}", width=400
    )
    slider_mestre = ft.Slider(
        min=0, max=0, divisions=1, value=0, label="{value}", width=400
    )

    play_button_aluno = ft.ElevatedButton(
        "Play", icon=ft.icons.PLAY_ARROW, on_click=None, disabled=True
    )
    pause_button_aluno = ft.ElevatedButton(
        "Pause", icon=ft.icons.PAUSE, on_click=None, disabled=True
    )
    step_forward_button_aluno = ft.ElevatedButton(
        "+1", icon=ft.icons.ARROW_RIGHT, on_click=None, disabled=True
    )
    step_backward_button_aluno = ft.ElevatedButton(
        "-1", icon=ft.icons.ARROW_LEFT, on_click=None, disabled=True
    )

    play_button_mestre = ft.ElevatedButton(
        "Play", icon=ft.icons.PLAY_ARROW, on_click=None, disabled=True
    )
    pause_button_mestre = ft.ElevatedButton(
        "Pause", icon=ft.icons.PAUSE, on_click=None, disabled=True
    )
    step_forward_button_mestre = ft.ElevatedButton(
        "+1", icon=ft.icons.ARROW_RIGHT, on_click=None, disabled=True
    )
    step_backward_button_mestre = ft.ElevatedButton(
        "-1", icon=ft.icons.ARROW_LEFT, on_click=None, disabled=True
    )

    def frame_to_base64(frame: np.ndarray) -> str:
        if frame is None or frame.size == 0:
            return ""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            _, buffer = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            _, buffer = cv2.imencode(".png", frame)
        return "data:image/png;base64," + np.base64encode(buffer).decode("utf-8")

    def update_video_display():
        global PROCESSED_FRAMES_ALUNO, PROCESSED_FRAMES_MESTRE, CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX

        # Aluno
        if PROCESSED_FRAMES_ALUNO and CURRENT_FRAME_ALUNO_INDEX < len(
            PROCESSED_FRAMES_ALUNO
        ):
            img_aluno.src_base64 = frame_to_base64(
                PROCESSED_FRAMES_ALUNO[CURRENT_FRAME_ALUNO_INDEX]
            )
            slider_aluno.value = CURRENT_FRAME_ALUNO_INDEX
            slider_aluno.max = len(PROCESSED_FRAMES_ALUNO) - 1
            slider_aluno.divisions = (
                len(PROCESSED_FRAMES_ALUNO) - 1
                if len(PROCESSED_FRAMES_ALUNO) > 1
                else 1
            )
            step_forward_button_aluno.disabled = (
                CURRENT_FRAME_ALUNO_INDEX >= len(PROCESSED_FRAMES_ALUNO) - 1
            )
            step_backward_button_aluno.disabled = CURRENT_FRAME_ALUNO_INDEX <= 0
        else:
            img_aluno.src_base64 = ""
            slider_aluno.max = 0
            slider_aluno.value = 0
            step_forward_button_aluno.disabled = True
            step_backward_button_aluno.disabled = True

        # Mestre
        if PROCESSED_FRAMES_MESTRE and CURRENT_FRAME_MESTRE_INDEX < len(
            PROCESSED_FRAMES_MESTRE
        ):
            img_mestre.src_base64 = frame_to_base64(
                PROCESSED_FRAMES_MESTRE[CURRENT_FRAME_MESTRE_INDEX]
            )
            slider_mestre.value = CURRENT_FRAME_MESTRE_INDEX
            slider_mestre.max = len(PROCESSED_FRAMES_MESTRE) - 1
            slider_mestre.divisions = (
                len(PROCESSED_FRAMES_MESTRE) - 1
                if len(PROCESSED_FRAMES_MESTRE) > 1
                else 1
            )
            step_forward_button_mestre.disabled = (
                CURRENT_FRAME_MESTRE_INDEX >= len(PROCESSED_FRAMES_MESTRE) - 1
            )
            step_backward_button_mestre.disabled = CURRENT_FRAME_MESTRE_INDEX <= 0
        else:
            img_mestre.src_base64 = ""
            slider_mestre.max = 0
            slider_mestre.value = 0
            step_forward_button_mestre.disabled = True
            step_backward_button_mestre.disabled = True

        page.update()

    def play_video(video_type: str):
        """
        Inicia a reprodução do vídeo especificado em uma thread separada.
        Args:
            video_type (str): 'aluno' ou 'mestre'.
        """
        global IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX

        if video_type == "aluno":
            IS_PLAYING_ALUNO = True
            frames = PROCESSED_FRAMES_ALUNO
            # current_index = CURRENT_FRAME_ALUNO_INDEX # Não usado diretamente aqui
            play_btn = play_button_aluno
            pause_btn = pause_button_aluno
        elif video_type == "mestre":
            IS_PLAYING_MESTRE = True
            frames = PROCESSED_FRAMES_MESTRE
            # current_index = CURRENT_FRAME_MESTRE_INDEX # Não usado diretamente aqui
            play_btn = play_button_mestre
            pause_btn = pause_button_mestre
        else:
            return

        play_btn.disabled = True
        pause_btn.disabled = False
        page.update()

        logger.info(f"Iniciando reprodução do vídeo: {video_type}")

        while (video_type == "aluno" and IS_PLAYING_ALUNO) or (
            video_type == "mestre" and IS_PLAYING_MESTRE
        ):
            if not frames:
                break

            if video_type == "aluno":
                if CURRENT_FRAME_ALUNO_INDEX >= len(frames):
                    CURRENT_FRAME_ALUNO_INDEX = 0  # Loop the video
                # current_index_local = CURRENT_FRAME_ALUNO_INDEX # Não usado
            else:  # mestre
                if CURRENT_FRAME_MESTRE_INDEX >= len(frames):
                    CURRENT_FRAME_MESTRE_INDEX = 0  # Loop the video
                # current_index_local = CURRENT_FRAME_MESTRE_INDEX # Não usado

            update_video_display()  # Atualiza a UI para o frame atual
            time.sleep(1 / FPS)  # Controla a velocidade de reprodução

            if video_type == "aluno":
                CURRENT_FRAME_ALUNO_INDEX += 1
            else:
                CURRENT_FRAME_MESTRE_INDEX += 1

        # Quando a reprodução para (ou termina)
        if video_type == "aluno":
            IS_PLAYING_ALUNO = False
        else:
            IS_PLAYING_MESTRE = False
        play_btn.disabled = False
        pause_btn.disabled = True
        page.update()
        logger.info(f"Reprodução do vídeo {video_type} finalizada.")

    def start_playback_aluno(e):
        global PLAYBACK_THREAD_ALUNO
        if not PLAYBACK_THREAD_ALUNO or not PLAYBACK_THREAD_ALUNO.is_alive():
            PLAYBACK_THREAD_ALUNO = threading.Thread(
                target=play_video, args=("aluno",), daemon=True
            )
            PLAYBACK_THREAD_ALUNO.start()
            logger.info("Thread de reprodução do vídeo do Aluno iniciada.")

    def stop_playback_aluno(e):
        global IS_PLAYING_ALUNO
        IS_PLAYING_ALUNO = False
        logger.info("Reprodução do vídeo do Aluno pausada.")

    def start_playback_mestre(e):
        global PLAYBACK_THREAD_MESTRE
        if not PLAYBACK_THREAD_MESTRE or not PLAYBACK_THREAD_MESTRE.is_alive():
            PLAYBACK_THREAD_MESTRE = threading.Thread(
                target=play_video, args=("mestre",), daemon=True
            )
            PLAYBACK_THREAD_MESTRE.start()
            logger.info("Thread de reprodução do vídeo do Mestre iniciada.")

    def stop_playback_mestre(e):
        global IS_PLAYING_MESTRE
        IS_PLAYING_MESTRE = False
        logger.info("Reprodução do vídeo do Mestre pausada.")

    # Atribui as funções aos botões
    play_button_aluno.on_click = start_playback_aluno
    pause_button_aluno.on_click = stop_playback_aluno
    play_button_mestre.on_click = start_playback_mestre
    pause_button_mestre.on_click = stop_playback_mestre

    def step_frame(video_type: str, direction: int):
        """
        Avança ou retrocede um frame no vídeo especificado.
        Args:
            video_type (str): 'aluno' ou 'mestre'.
            direction (int): 1 para avançar, -1 para retroceder.
        """
        global CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX, IS_PLAYING_ALUNO, IS_PLAYING_MESTRE

        if video_type == "aluno":
            frames = PROCESSED_FRAMES_ALUNO
            if IS_PLAYING_ALUNO:
                stop_playback_aluno(None)  # Pausa se estiver tocando
            if frames:
                CURRENT_FRAME_ALUNO_INDEX = max(
                    0, min(len(frames) - 1, CURRENT_FRAME_ALUNO_INDEX + direction)
                )
                logger.debug(f"Aluno: Step para frame {CURRENT_FRAME_ALUNO_INDEX}")
        elif video_type == "mestre":
            frames = PROCESSED_FRAMES_MESTRE
            if IS_PLAYING_MESTRE:
                stop_playback_mestre(None)  # Pausa se estiver tocando
            if frames:
                CURRENT_FRAME_MESTRE_INDEX = max(
                    0, min(len(frames) - 1, CURRENT_FRAME_MESTRE_INDEX + direction)
                )
                logger.debug(f"Mestre: Step para frame {CURRENT_FRAME_MESTRE_INDEX}")
        update_video_display()

    step_forward_button_aluno.on_click = lambda _: step_frame("aluno", 1)
    step_backward_button_aluno.on_click = lambda _: step_frame("aluno", -1)
    step_forward_button_mestre.on_click = lambda _: step_frame("mestre", 1)
    step_backward_button_mestre.on_click = lambda _: step_frame("mestre", -1)

    def process_and_display_videos():
        global VIDEO_ALUNO_PATH, VIDEO_MESTRE_PATH, PROCESSED_FRAMES_ALUNO, PROCESSED_FRAMES_MESTRE
        global CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX

        # Desabilita botões de play/pause e navegação enquanto processa
        play_button_aluno.disabled = True
        pause_button_aluno.disabled = True
        step_forward_button_aluno.disabled = True
        step_backward_button_aluno.disabled = True
        play_button_mestre.disabled = True
        pause_button_mestre.disabled = True
        step_forward_button_mestre.disabled = True
        step_backward_button_mestre.disabled = True
        page.update()

        if not VIDEO_ALUNO_PATH or not VIDEO_MESTRE_PATH:
            status_text.value = (
                "Por favor, carregue ambos os vídeos para iniciar a análise."
            )
            page.update()
            return

        status_text.value = "Iniciando a análise dos vídeos. Isso pode levar alguns minutos, por favor aguarde..."
        page.update()
        logger.info("Botão 'Analisar Movimentos' clicado. Iniciando análise.")

        analyzer = VideoAnalyzer()

        PROCESSED_FRAMES_ALUNO = []
        PROCESSED_FRAMES_MESTRE = []
        CURRENT_FRAME_ALUNO_INDEX = 0
        CURRENT_FRAME_MESTRE_INDEX = 0

        try:
            # Processamento do vídeo do Aluno
            logger.info(f"Processando vídeo do Aluno: {VIDEO_ALUNO_PATH}")
            with open(VIDEO_ALUNO_PATH, "rb") as f_aluno:
                for i, (frame, l_data) in enumerate(analyzer.analyze_video(f_aluno)):
                    PROCESSED_FRAMES_ALUNO.append(frame)
                    status_text.value = f"Processando vídeo do Aluno: Frame {i}..."
                    if i % 10 == 0:
                        page.update()
            logger.info("Processamento do vídeo do Aluno concluído.")

            # Processamento do vídeo do Mestre
            logger.info(f"Processando vídeo do Mestre: {VIDEO_MESTRE_PATH}")
            with open(VIDEO_MESTRE_PATH, "rb") as f_mestre:
                for i, (frame, l_data) in enumerate(analyzer.analyze_video(f_mestre)):
                    PROCESSED_FRAMES_MESTRE.append(frame)
                    status_text.value = f"Processando vídeo do Mestre: Frame {i}..."
                    if i % 10 == 0:
                        page.update()
            logger.info("Processamento do vídeo do Mestre concluído.")

            status_text.value = "Ambos os vídeos processados! Exibindo resultados..."
            page.update()

            update_video_display()

            play_button_aluno.disabled = False
            pause_button_aluno.disabled = True
            step_forward_button_aluno.disabled = False
            step_backward_button_aluno.disabled = False
            play_button_mestre.disabled = False
            pause_button_mestre.disabled = True
            step_forward_button_mestre.disabled = False
            step_backward_button_mestre.disabled = False

            status_text.value = "Análise de pose concluída! ✨"
            page.update()

        except Exception as e:
            logger.error(f"Erro durante o processamento do vídeo: {e}", exc_info=True)
            status_text.value = f"Ocorreu um erro durante a análise do vídeo: {e}"
            play_button_aluno.disabled = True
            pause_button_aluno.disabled = True
            step_forward_button_aluno.disabled = True
            step_backward_button_aluno.disabled = True
            play_button_mestre.disabled = True
            pause_button_mestre.disabled = True
            step_forward_button_mestre.disabled = True
            step_backward_button_mestre.disabled = True
            page.update()
        finally:
            if analyzer:
                del analyzer

    def on_aluno_file_picked(e: ft.FilePickerResultEvent):
        global VIDEO_ALUNO_PATH
        if e.files:
            VIDEO_ALUNO_PATH = e.files[0].path
            status_text.value = (
                f"Vídeo do Aluno carregado: {os.path.basename(VIDEO_ALUNO_PATH)}"
            )
            logger.info(f"Vídeo do Aluno carregado: {VIDEO_ALUNO_PATH}")
        else:
            VIDEO_ALUNO_PATH = None
            status_text.value = "Seleção do vídeo do Aluno cancelada."
            logger.info("Seleção do vídeo do Aluno cancelada.")
        page.update()

    def on_mestre_file_picked(e: ft.FilePickerResultEvent):
        global VIDEO_MESTRE_PATH
        if e.files:
            VIDEO_MESTRE_PATH = e.files[0].path
            status_text.value = (
                f"Vídeo do Mestre carregado: {os.path.basename(VIDEO_MESTRE_PATH)}"
            )
            logger.info(f"Vídeo do Mestre carregado: {VIDEO_MESTRE_PATH}")
        else:
            VIDEO_MESTRE_PATH = None
            status_text.value = "Seleção do vídeo do Mestre cancelada."
            logger.info("Seleção do vídeo do Mestre cancelada.")
        page.update()

    file_picker_aluno = ft.FilePicker(on_result=on_aluno_file_picked)
    file_picker_mestre = ft.FilePicker(on_result=on_mestre_file_picked)
    page.overlay.append(file_picker_aluno)
    page.overlay.append(file_picker_mestre)

    def slider_aluno_on_change(e):
        global CURRENT_FRAME_ALUNO_INDEX, IS_PLAYING_ALUNO
        if IS_PLAYING_ALUNO:
            stop_playback_aluno(None)
        CURRENT_FRAME_ALUNO_INDEX = int(e.control.value)
        update_video_display()
        logger.debug(f"Slider Aluno alterado para frame: {CURRENT_FRAME_ALUNO_INDEX}")

    def slider_mestre_on_change(e):
        global CURRENT_FRAME_MESTRE_INDEX, IS_PLAYING_MESTRE
        if IS_PLAYING_MESTRE:
            stop_playback_mestre(None)
        CURRENT_FRAME_MESTRE_INDEX = int(e.control.value)
        update_video_display()
        logger.debug(f"Slider Mestre alterado para frame: {CURRENT_FRAME_MESTRE_INDEX}")

    slider_aluno.on_change = slider_aluno_on_change
    slider_mestre.on_change = slider_mestre_on_change

    page.add(
        ft.Row(
            [
                # Coluna Esquerda: Controles de Upload e Frame
                ft.Column(
                    [
                        ft.Text(
                            "🥋 Analisador de Movimentos de Krav Maga",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Text("Selecione um vídeo (Aluno):", size=16),
                        ft.ElevatedButton(
                            "Upload Vídeo do Aluno",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: file_picker_aluno.pick_files(
                                allow_multiple=False,
                                allowed_extensions=["mp4", "mov", "avi"],
                            ),
                        ),
                        ft.Text("Selecione um vídeo de ref (Mestre):", size=16),
                        ft.ElevatedButton(
                            "Upload Vídeo do Mestre",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: file_picker_mestre.pick_files(
                                allow_multiple=False,
                                allowed_extensions=["mp4", "mov", "avi"],
                            ),
                        ),
                        ft.Divider(),
                        ft.ElevatedButton(
                            "Analisar Movimentos",
                            on_click=lambda _: process_and_display_videos(),
                        ),
                        status_text,
                        ft.Divider(),
                        ft.Text("Controles de Vídeo (Aluno):", size=16),
                        slider_aluno,
                        ft.Row(
                            [
                                step_backward_button_aluno,
                                play_button_aluno,
                                pause_button_aluno,
                                step_forward_button_aluno,
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_AROUND,
                        ),
                        ft.Divider(),
                        ft.Text("Controles de Vídeo (Mestre):", size=16),
                        slider_mestre,
                        ft.Row(
                            [
                                step_backward_button_mestre,
                                play_button_mestre,
                                pause_button_mestre,
                                step_forward_button_mestre,
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_AROUND,
                        ),
                    ],
                    width=450,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=15,
                ),
                # Coluna Direita: Visualização dos Vídeos
                ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Text(
                                    "Vídeo do Aluno",
                                    size=18,
                                    weight=ft.FontWeight.MEDIUM,
                                ),
                                ft.Text(
                                    "Vídeo do Mestre",
                                    size=18,
                                    weight=ft.FontWeight.MEDIUM,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_AROUND,
                            width=900,
                        ),
                        ft.Row(
                            [
                                ft.Container(
                                    content=img_aluno,
                                    alignment=ft.alignment.center,
                                    width=450,
                                    height=350,
                                    bgcolor=ft.colors.BLACK,
                                    border_radius=10,
                                ),
                                ft.Container(
                                    content=img_mestre,
                                    alignment=ft.alignment.center,
                                    width=450,
                                    height=350,
                                    bgcolor=ft.colors.BLACK,
                                    border_radius=10,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_AROUND,
                            width=900,
                        ),
                        ft.Text(
                            "Resultados da Análise e Feedback: (Em breve)",
                            size=18,
                            weight=ft.FontWeight.MEDIUM,
                        ),
                        ft.Container(
                            content=ft.Text(
                                "Feedback textual aqui...", selectable=True
                            ),
                            width=900,
                            height=200,
                            bgcolor=ft.colors.BLUE_GREY_50,
                            padding=10,
                            border_radius=10,
                            alignment=ft.alignment.top_left,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                    expand=True,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=20,
            expand=True,
        )
    )
    page.update()
    logger.info("Elementos da UI Flet adicionados e página atualizada.")


# Ponto de entrada da aplicação Flet.
if __name__ == "__main__":
    ft.app(target=main)
