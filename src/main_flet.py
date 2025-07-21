# src/main_flet.py

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas Essenciais
# --------------------------------------------------------------------------------------------------
import flet as ft
import logging
import os
import cv2
import base64
import threading
import sys
import asyncio

# --- Adiciona o diretório raiz do projeto ao path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Importação dos Módulos do Projeto ---
from src.utils import setup_logging, get_logger
from src.video_analyzer import VideoAnalyzer

# --------------------------------------------------------------------------------------------------
# Configuração Inicial
# --------------------------------------------------------------------------------------------------
setup_logging()
logger = get_logger(__name__)


# --------------------------------------------------------------------------------------------------
# Classe Principal da Aplicação Flet
# --------------------------------------------------------------------------------------------------
class KravMagaApp:
    """
    Classe que encapsula toda a lógica e a interface do usuário da aplicação.
    """

    def __init__(self, page: ft.Page):
        """
        Inicializador da classe da aplicação.
        """
        self.page = page
        self.video_analyzer = None
        self.is_playing = False  # Estado para controlar a reprodução automática.

        self.setup_controls()
        self.build_layout()
        logger.info("Aplicação Flet e UI inicializadas com sucesso.")

    def setup_controls(self):
        """Inicializa todos os widgets Flet que serão usados na UI."""
        logger.info("Inicializando todos os controles da UI.")

        self.status_text = ft.Text(
            "Por favor, carregue ambos os vídeos para iniciar a análise.",
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

        # Placeholders que serão mostrados antes e durante a análise.
        self.aluno_placeholder = ft.Container(
            content=ft.Text("Carregue o vídeo do Aluno"),
            width=500,
            height=400,
            bgcolor=ft.colors.BLACK26,
            border_radius=ft.border_radius.all(10),
            alignment=ft.alignment.center,
        )
        self.mestre_placeholder = ft.Container(
            content=ft.Text("Carregue o vídeo do Mestre"),
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
            visible=False,
            on_change=self.on_slider_change,
            expand=True,
        )

        # --- NOVOS CONTROLES DE REPRODUÇÃO ---
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

        self.file_picker_aluno = ft.FilePicker(
            on_result=lambda e: self.pick_file_result(e, is_aluno=True)
        )
        self.file_picker_mestre = ft.FilePicker(
            on_result=lambda e: self.pick_file_result(e, is_aluno=False)
        )
        self.page.overlay.extend([self.file_picker_aluno, self.file_picker_mestre])

    def build_layout(self):
        """Constrói o layout visual da aplicação."""
        logger.info("Construindo o layout da UI.")
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

    def pick_file_result(self, e: ft.FilePickerResultEvent, is_aluno: bool):
        """Manipulador de evento para quando um arquivo é selecionado."""
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
        """Verifica se ambos os vídeos foram carregados e atualiza o estado do botão."""
        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")

        if aluno_path and mestre_path:
            self.analyze_button.disabled = False
            self.status_text.value = "Vídeos carregados. Pronto para analisar."
            logger.info(
                "Ambos os vídeos foram selecionados. Botão de análise habilitado."
            )
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
        logger.info("Botão de análise clicado. Iniciando processo.")
        self.status_text.value = "Análise em andamento, por favor aguarde..."
        self.analyze_button.disabled = True
        self.aluno_placeholder.content = ft.ProgressRing()
        self.mestre_placeholder.content = ft.ProgressRing()
        self.page.update()

        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")

        self.video_analyzer = VideoAnalyzer()
        try:
            with open(aluno_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=True)
            with open(mestre_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=False)
        except Exception as ex:
            logger.error(f"Falha ao carregar vídeos para análise: {ex}")
            self.status_text.value = f"Erro ao ler os arquivos de vídeo: {ex}"
            self.page.update()
            return

        threading.Thread(target=self.run_analysis_and_update_ui, daemon=True).start()

    def run_analysis_and_update_ui(self):
        """Executa a análise e agenda a atualização da UI."""
        self.video_analyzer.analyze_and_compare()
        if self.video_analyzer.processing_thread:
            self.video_analyzer.processing_thread.join()

        logger.info("Análise na thread concluída. Agendando atualização da UI.")
        # CORREÇÃO: Usa 'self.page.invoke_rpc' que é a maneira correta de chamar uma função na thread da UI
        # a partir de uma thread de background no Flet. É uma chamada síncrona.
        self.setup_ui_post_analysis()

    def setup_ui_post_analysis(self):
        """Configura a UI após a conclusão da análise."""
        logger.info("Configurando a UI para exibir os resultados da análise.")
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        if num_frames > 0:
            self.slider_control.max = num_frames - 1
            self.slider_control.divisions = num_frames - 1 if num_frames > 1 else 1
            self.slider_control.disabled = False
            self.playback_controls.visible = True

            self.status_text.value = "Análise completa! Use os controles para navegar."
            self.update_frame_display(0)
        else:
            self.status_text.value = "Erro: Não foi possível processar os vídeos."

        self.page.update()

    async def on_slider_change(self, e):
        """Atualiza a exibição do frame quando o slider é movido."""
        frame_index = int(e.control.value)
        self.update_frame_display(frame_index)
        await self.page.update_async()

    def update_frame_display(self, frame_index):
        """Busca os frames processados e os exibe na UI."""
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

    def frame_to_base64(self, frame):
        """Converte um frame do OpenCV para uma string base64."""
        _, buffer = cv2.imencode(".png", frame)
        return base64.b64encode(buffer).decode("utf-8")

    # --- LÓGICA DOS NOVOS CONTROLES DE REPRODUÇÃO ---
    async def toggle_play_pause(self, e):
        """Inicia ou pausa a reprodução automática dos frames."""
        self.is_playing = not self.is_playing
        self.play_button.icon = (
            ft.icons.PAUSE if self.is_playing else ft.icons.PLAY_ARROW
        )
        await self.page.update_async()

        if self.is_playing:
            logger.info("Iniciando reprodução automática.")
            await self.page.run_task(self.play_video_async)

    async def play_video_async(self):
        """Loop assíncrono para reproduzir os frames do vídeo."""
        start_index = int(self.slider_control.value)
        num_frames = len(self.video_analyzer.processed_frames_aluno)

        for i in range(start_index, num_frames):
            if not self.is_playing:
                logger.info("Reprodução interrompida pelo usuário.")
                break

            self.update_frame_display(i)
            await self.page.update_async()
            await asyncio.sleep(1 / 30)  # Simula 30 FPS.

        self.is_playing = False
        self.play_button.icon = ft.icons.PLAY_ARROW
        await self.page.update_async()
        logger.info("Reprodução automática finalizada.")

    async def prev_frame(self, e):
        """Vai para o frame anterior."""
        new_index = max(0, int(self.slider_control.value) - 1)
        self.update_frame_display(new_index)
        await self.page.update_async()

    async def next_frame(self, e):
        """Vai para o próximo frame."""
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        new_index = min(num_frames - 1, int(self.slider_control.value) + 1)
        self.update_frame_display(new_index)
        await self.page.update_async()


def main(page: ft.Page):
    """Função de entrada que o Flet chama para iniciar a aplicação."""
    logger.info("Iniciando a aplicação Flet.")
    KravMagaApp(page)


if __name__ == "__main__":
    ft.app(target=main)
