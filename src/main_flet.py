# src/main_flet.py

import flet as ft
import logging
import os
import cv2
import base64
import threading
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import setup_logging, get_logger
from src.video_analyzer import VideoAnalyzer

setup_logging()
logger = get_logger(__name__)


class KravMagaApp:
    """
    Encapsula toda a lógica e a interface do usuário da aplicação.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.video_analyzer = None
        self.is_playing = False
        self.playback_thread = None

        self.setup_controls()
        self.build_layout()
        logger.info("Aplicação Flet e UI inicializadas.")

    def setup_controls(self):
        """Inicializa todos os widgets Flet."""
        self.status_text = ft.Text(
            "Por favor, carregue os vídeos para iniciar.",
            text_align=ft.TextAlign.CENTER,
            size=16,
        )
        self.analyze_button = ft.ElevatedButton(
            "Analisar Movimentos",
            icon=ft.icons.ANALYTICS,
            on_click=self.analyze_videos,
            disabled=True,
        )

        self.img_aluno_control = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            visible=False,
            border_radius=ft.border_radius.all(10),
        )
        self.img_mestre_control = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            visible=False,
            border_radius=ft.border_radius.all(10),
        )

        self.aluno_placeholder = ft.Container(
            content=ft.Text("Vídeo do Aluno"),
            width=500,
            height=400,
            bgcolor=ft.colors.BLACK26,
            border_radius=ft.border_radius.all(10),
            alignment=ft.alignment.center,
        )
        self.mestre_placeholder = ft.Container(
            content=ft.Text("Vídeo do Mestre"),
            width=500,
            height=400,
            bgcolor=ft.colors.BLACK26,
            border_radius=ft.border_radius.all(10),
            alignment=ft.alignment.center,
        )

        self.slider_control = ft.Slider(
            min=0,
            max=0,
            divisions=1,
            value=0,
            disabled=True,
            on_change=self.on_slider_change,
            expand=True,
        )

        self.play_button = ft.IconButton(
            icon=ft.icons.PLAY_ARROW,
            on_click=self.toggle_play_pause,
            tooltip="Reproduzir/Pausar",
        )
        self.prev_frame_button = ft.IconButton(
            icon=ft.icons.SKIP_PREVIOUS,
            on_click=self.prev_frame,
            tooltip="Frame Anterior",
        )
        self.next_frame_button = ft.IconButton(
            icon=ft.icons.SKIP_NEXT, on_click=self.next_frame, tooltip="Próximo Frame"
        )
        self.playback_controls = ft.Row(
            [
                self.prev_frame_button,
                self.play_button,
                self.next_frame_button,
                self.slider_control,
            ],
            visible=False,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        self.file_picker_aluno = ft.FilePicker(on_result=self.on_pick_file_result_aluno)
        self.file_picker_mestre = ft.FilePicker(
            on_result=self.on_pick_file_result_mestre
        )
        self.page.overlay.extend([self.file_picker_aluno, self.file_picker_mestre])

    def build_layout(self):
        """Constrói o layout visual da aplicação."""
        self.page.title = "Analisador de Movimentos de Krav Maga"
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.scroll = ft.ScrollMode.ADAPTIVE
        self.page.theme_mode = ft.ThemeMode.DARK

        self.page.add(
            ft.Column(
                [
                    ft.Text(
                        "Analisador de Movimentos de Krav Maga 🥋",
                        size=28,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Row(
                        [
                            ft.ElevatedButton(
                                "Upload Vídeo do Aluno",
                                icon=ft.icons.UPLOAD_FILE,
                                on_click=lambda _: self.file_picker_aluno.pick_files(
                                    allow_multiple=False,
                                    allowed_extensions=["mp4", "mov", "avi"],
                                ),
                            ),
                            ft.ElevatedButton(
                                "Upload Vídeo do Mestre",
                                icon=ft.icons.UPLOAD_FILE,
                                on_click=lambda _: self.file_picker_mestre.pick_files(
                                    allow_multiple=False,
                                    allowed_extensions=["mp4", "mov", "avi"],
                                ),
                            ),
                            self.analyze_button,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    ft.Container(content=self.status_text, padding=10),
                    ft.Row(
                        [
                            ft.Column(
                                [
                                    ft.Text(
                                        "Vídeo do Aluno", weight=ft.FontWeight.BOLD
                                    ),
                                    ft.Stack(
                                        [self.aluno_placeholder, self.img_aluno_control]
                                    ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        "Vídeo do Mestre", weight=ft.FontWeight.BOLD
                                    ),
                                    ft.Stack(
                                        [
                                            self.mestre_placeholder,
                                            self.img_mestre_control,
                                        ]
                                    ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        spacing=30,
                    ),
                    self.playback_controls,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            )
        )
        self.page.update()

    def on_pick_file_result_aluno(self, e: ft.FilePickerResultEvent):
        self.pick_file_result(e, is_aluno=True)

    def on_pick_file_result_mestre(self, e: ft.FilePickerResultEvent):
        self.pick_file_result(e, is_aluno=False)

    def pick_file_result(self, e: ft.FilePickerResultEvent, is_aluno: bool):
        if not e.files:
            return
        video_path = e.files[0].path
        storage_key = "video_aluno_path" if is_aluno else "video_mestre_path"
        self.page.client_storage.set(storage_key, video_path)
        logger.info(
            f"Caminho do vídeo {'aluno' if is_aluno else 'mestre'} salvo: {video_path}"
        )
        self.update_status_and_button_state()

    def update_status_and_button_state(self):
        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")

        if aluno_path and mestre_path:
            self.analyze_button.disabled = False
            self.status_text.value = "Vídeos carregados. Pronto para analisar."
        elif aluno_path:
            self.status_text.value = (
                "Vídeo do aluno carregado. Aguardando vídeo do mestre."
            )
        elif mestre_path:
            self.status_text.value = (
                "Vídeo do mestre carregado. Aguardando vídeo do aluno."
            )

        self.page.update()

    def analyze_videos(self, e):
        """Inicia a análise dos vídeos em uma thread."""
        self.status_text.value = "Análise em andamento..."
        self.analyze_button.disabled = True
        self.aluno_placeholder.content = ft.ProgressRing()
        self.mestre_placeholder.content = ft.ProgressRing()
        self.aluno_placeholder.visible = True
        self.mestre_placeholder.visible = True
        self.page.update()

        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")

        self.video_analyzer = VideoAnalyzer()
        try:
            with open(aluno_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=True)
            with open(mestre_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=False)

            self.video_analyzer.analyze_and_compare(
                post_analysis_callback=self.setup_ui_post_analysis
            )
        except Exception as ex:
            logger.error(f"Falha ao carregar vídeos: {ex}", exc_info=True)
            self.status_text.value = f"Erro ao ler os arquivos: {ex}"
            self.page.update()

    def setup_ui_post_analysis(self):
        """Configura a UI após a conclusão da análise."""
        logger.info("Configurando a UI para exibir os resultados.")
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        if num_frames > 0:
            self.slider_control.max = num_frames - 1
            self.slider_control.divisions = num_frames - 1 if num_frames > 1 else 1
            self.slider_control.disabled = False
            self.playback_controls.visible = True

            self.status_text.value = "Análise completa! Use os controles abaixo."
            self.update_frame_display(0)
        else:
            self.status_text.value = "Erro: Não foi possível processar os vídeos."

        self.page.update()

    def on_slider_change(self, e):
        self.update_frame_display(int(e.control.value))

    def update_frame_display(self, frame_index):
        if not self.video_analyzer or frame_index >= len(
            self.video_analyzer.processed_frames_aluno
        ):
            return

        self.slider_control.value = frame_index

        self.img_aluno_control.src_base64 = self.frame_to_base64(
            self.video_analyzer.processed_frames_aluno[frame_index]
        )
        self.img_mestre_control.src_base64 = self.frame_to_base64(
            self.video_analyzer.processed_frames_mestre[frame_index]
        )

        self.aluno_placeholder.visible = False
        self.mestre_placeholder.visible = False
        self.img_aluno_control.visible = True
        self.img_mestre_control.visible = True

        self.page.update()

    def frame_to_base64(self, frame):
        """Converte um frame do OpenCV para uma string base64."""
        _, buffer = cv2.imencode(".png", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def toggle_play_pause(self, e):
        """Inicia ou pausa a reprodução automática dos frames."""
        self.is_playing = not self.is_playing
        self.play_button.icon = (
            ft.icons.PAUSE if self.is_playing else ft.icons.PLAY_ARROW
        )

        if self.is_playing:
            logger.info("Iniciando reprodução automática.")
            self.playback_thread = threading.Thread(
                target=self.play_video_loop, daemon=True
            )
            self.playback_thread.start()
        else:
            logger.info("Reprodução pausada.")

        self.page.update()

    def play_video_loop(self):
        """Loop que executa em uma thread para reproduzir os frames."""
        start_index = int(self.slider_control.value)
        num_frames = len(self.video_analyzer.processed_frames_aluno)

        for i in range(start_index, num_frames):
            if not self.is_playing:
                break

            self.update_frame_display(i)
            time.sleep(1 / 30)

        self.is_playing = False
        self.play_button.icon = ft.icons.PLAY_ARROW
        self.page.update()
        logger.info("Reprodução automática finalizada.")

    def prev_frame(self, e):
        """Vai para o frame anterior."""
        new_index = max(0, int(self.slider_control.value) - 1)
        self.update_frame_display(new_index)

    def next_frame(self, e):
        """Vai para o próximo frame."""
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        new_index = min(num_frames - 1, int(self.slider_control.value) + 1)
        self.update_frame_display(new_index)


def main(page: ft.Page):
    """Função de entrada que o Flet chama para iniciar a aplicação."""
    logger.info("Iniciando a aplicação Flet.")
    KravMagaApp(page)


if __name__ == "__main__":
    ft.app(target=main)
