# src/main_flet.py

import flet as ft
import logging
import os
import cv2
import base64
import asyncio
import sys

# Adiciona o diretório raiz do projeto ao path
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

        self.setup_controls()
        self.build_layout()
        logger.info("Aplicação Flet e UI inicializadas com sucesso.")

    def setup_controls(self):
        """Inicializa todos os widgets Flet que serão usados na UI."""
        logger.info("Inicializando todos os controles da UI.")

        self.status_text = ft.Text(
            "Por favor, carregue os vídeos do aluno e do mestre.",
            text_align=ft.TextAlign.CENTER,
            size=16,
        )
        self.analyze_button = ft.ElevatedButton(
            "Analisar Movimentos",
            icon=ft.icons.ANALYTICS,
            on_click=self.on_analyze_click,
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
                controls=[
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

    async def on_pick_file_result_aluno(self, e: ft.FilePickerResultEvent):
        await self.pick_file_result(e, is_aluno=True)

    async def on_pick_file_result_mestre(self, e: ft.FilePickerResultEvent):
        await self.pick_file_result(e, is_aluno=False)

    async def pick_file_result(self, e: ft.FilePickerResultEvent, is_aluno: bool):
        if not e.files:
            return
        video_path = e.files[0].path
        storage_key = "video_aluno_path" if is_aluno else "video_mestre_path"
        await self.page.client_storage.set_async(storage_key, video_path)
        logger.info(
            f"Caminho do vídeo {'aluno' if is_aluno else 'mestre'} salvo: {video_path}"
        )
        await self.update_status_and_button_state()

    async def update_status_and_button_state(self):
        aluno_path = await self.page.client_storage.get_async("video_aluno_path")
        mestre_path = await self.page.client_storage.get_async("video_mestre_path")

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

        await self.page.update_async()

    async def on_analyze_click(self, e):
        await self.page.run_task(self.analyze_videos_async)

    async def analyze_videos_async(self):
        """Função assíncrona que executa a análise de vídeo."""
        logger.info("Iniciando tarefa de análise assíncrona.")
        self.status_text.value = "Análise em andamento, por favor aguarde..."
        self.analyze_button.disabled = True
        self.aluno_placeholder.content = ft.ProgressRing()
        self.mestre_placeholder.content = ft.ProgressRing()
        self.aluno_placeholder.visible = True
        self.mestre_placeholder.visible = True
        await self.page.update_async()

        aluno_path = await self.page.client_storage.get_async("video_aluno_path")
        mestre_path = await self.page.client_storage.get_async("video_mestre_path")

        self.video_analyzer = VideoAnalyzer()
        try:
            with open(aluno_path, "rb") as f:
                aluno_bytes = f.read()
            with open(mestre_path, "rb") as f:
                mestre_bytes = f.read()

            self.video_analyzer.load_video_from_bytes(aluno_bytes, is_aluno=True)
            self.video_analyzer.load_video_from_bytes(mestre_bytes, is_aluno=False)

            await asyncio.to_thread(self.video_analyzer.analyze_and_compare_sync)

            logger.info("Análise na thread concluída. Atualizando UI.")
            await self.setup_ui_post_analysis()
        except Exception as ex:
            logger.error(f"Falha na tarefa de análise: {ex}", exc_info=True)
            self.status_text.value = f"Erro durante a análise: {ex}"
            await self.page.update_async()

    async def setup_ui_post_analysis(self):
        logger.info("Configurando a UI para exibir os resultados da análise.")
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        if num_frames > 0:
            self.slider_control.max = num_frames - 1
            self.slider_control.divisions = num_frames - 1 if num_frames > 1 else 1
            self.slider_control.disabled = False
            self.playback_controls.visible = True

            self.status_text.value = "Análise completa! Use os controles para navegar."
            await self.update_frame_display(0)
        else:
            self.status_text.value = "Erro: Não foi possível processar os vídeos."

        await self.page.update_async()

    async def on_slider_change(self, e):
        frame_index = int(e.control.value)
        await self.update_frame_display(frame_index)

    async def update_frame_display(self, frame_index):
        if not self.video_analyzer or frame_index >= len(
            self.video_analyzer.processed_frames_aluno
        ):
            return

        self.slider_control.value = frame_index

        aluno_frame = self.video_analyzer.processed_frames_aluno[frame_index]
        mestre_frame = self.video_analyzer.processed_frames_mestre[frame_index]

        self.img_aluno_control.src_base64 = self.frame_to_base64(aluno_frame)
        self.img_mestre_control.src_base64 = self.frame_to_base64(mestre_frame)

        self.aluno_placeholder.visible = False
        self.mestre_placeholder.visible = False
        self.img_aluno_control.visible = True
        self.img_mestre_control.visible = True

        await self.page.update_async()

    def frame_to_base64(self, frame):
        """Converte um frame do OpenCV para uma string base64."""
        _, buffer = cv2.imencode(".png", frame)
        return base64.b64encode(buffer).decode("utf-8")

    async def toggle_play_pause(self, e):
        self.is_playing = not self.is_playing
        self.play_button.icon = (
            ft.icons.PAUSE if self.is_playing else ft.icons.PLAY_ARROW
        )
        await self.page.update_async()

        if self.is_playing:
            logger.info("Iniciando reprodução automática.")
            await self.page.run_task(self.play_video_loop)

    async def play_video_loop(self):
        """Loop assíncrono para reproduzir os frames do vídeo."""
        start_index = int(self.slider_control.value)
        num_frames = len(self.video_analyzer.processed_frames_aluno)

        for i in range(start_index, num_frames):
            if not self.is_playing:
                logger.info("Reprodução interrompida pelo usuário.")
                break

            await self.update_frame_display(i)
            await asyncio.sleep(1 / 30)  # Simula 30 FPS.

        self.is_playing = False
        self.play_button.icon = ft.icons.PLAY_ARROW
        await self.page.update_async()
        logger.info("Reprodução automática finalizada.")

    async def prev_frame(self, e):
        """Vai para o frame anterior."""
        new_index = max(0, int(self.slider_control.value) - 1)
        await self.update_frame_display(new_index)

    async def next_frame(self, e):
        """Vai para o próximo frame."""
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        new_index = min(num_frames - 1, int(self.slider_control.value) + 1)
        await self.update_frame_display(new_index)


async def main(page: ft.Page):
    logger.info("Iniciando a aplicação Flet.")
    app = KravMagaApp(page)


if __name__ == "__main__":
    ft.app(target=main)
