# src/main_flet.py

import flet as ft
import logging
import os
import cv2
import base64
import threading
import time
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import setup_logging, get_logger
from src.video_analyzer import VideoAnalyzer
from src.report_generator import ReportGenerator

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
            icon=ft.Icons.ANALYTICS,
            on_click=self.analyze_videos,
            disabled=True,
        )

        # --- NOVO CONTROLE DE BARRA DE PROGRESSO ---
        self.progress_bar = ft.ProgressBar(width=400, visible=False)

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
            bgcolor=ft.Colors.BLACK26,
            border_radius=ft.border_radius.all(10),
            alignment=ft.alignment.center,
        )
        self.mestre_placeholder = ft.Container(
            content=ft.Text("Vídeo do Mestre"),
            width=500,
            height=400,
            bgcolor=ft.Colors.BLACK26,
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
            icon=ft.Icons.PLAY_ARROW,
            on_click=self.toggle_play_pause,
            tooltip="Reproduzir/Pausar",
        )
        self.prev_frame_button = ft.IconButton(
            icon=ft.Icons.SKIP_PREVIOUS,
            on_click=self.prev_frame,
            tooltip="Frame Anterior",
        )
        self.next_frame_button = ft.IconButton(
            icon=ft.Icons.SKIP_NEXT, on_click=self.next_frame, tooltip="Próximo Frame"
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

        self.report_button = ft.ElevatedButton(
            "Gerar Relatório PDF",
            icon=ft.Icons.PICTURE_AS_PDF,
            on_click=self.on_generate_report_click,
            visible=False,
        )

        self.file_picker_aluno = ft.FilePicker(on_result=self.on_pick_file_result_aluno)
        self.file_picker_mestre = ft.FilePicker(
            on_result=self.on_pick_file_result_mestre
        )
        self.save_file_picker = ft.FilePicker(on_result=self.on_report_saved)
        self.page.overlay.extend(
            [self.file_picker_aluno, self.file_picker_mestre, self.save_file_picker]
        )

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
                                icon=ft.Icons.UPLOAD_FILE,
                                on_click=lambda _: self.file_picker_aluno.pick_files(
                                    allow_multiple=False,
                                    allowed_extensions=["mp4", "mov", "avi"],
                                ),
                            ),
                            ft.ElevatedButton(
                                "Upload Vídeo do Mestre",
                                icon=ft.Icons.UPLOAD_FILE,
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
                    # Adiciona a barra de progresso ao layout
                    self.progress_bar,
                    ft.ResponsiveRow(
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
                                col={"xs": 12, "md": 6},
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
                                col={"xs": 12, "md": 6},
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                    self.playback_controls,
                    self.report_button,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            )
        )
        self.page.update()

    def update_progress(self, percent_complete):
        """Callback para atualizar a barra de progresso na UI."""
        self.progress_bar.value = percent_complete
        self.status_text.value = f"Analisando... {int(percent_complete * 100)}%"
        self.page.update()

    def analyze_videos(self, e):
        """Inicia a análise dos vídeos em uma thread."""
        self.status_text.value = "Análise em andamento..."
        self.analyze_button.disabled = True
        self.progress_bar.value = 0
        self.progress_bar.visible = True
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
                post_analysis_callback=self.setup_ui_post_analysis,
                progress_callback=self.update_progress,  # Passa a função de callback
            )
        except Exception as ex:
            logger.error(f"Falha ao carregar vídeos: {ex}", exc_info=True)
            self.status_text.value = f"Erro ao ler os arquivos: {ex}"
            self.progress_bar.visible = False
            self.page.update()

    def setup_ui_post_analysis(self):
        """Configura a UI após a conclusão da análise."""
        logger.info("Configurando a UI para exibir os resultados.")
        num_frames = len(self.video_analyzer.processed_frames_aluno)

        self.progress_bar.visible = False  # Esconde a barra de progresso

        if num_frames > 0:
            self.slider_control.max = num_frames - 1
            self.slider_control.divisions = num_frames - 1 if num_frames > 1 else 1
            self.slider_control.disabled = False
            self.playback_controls.visible = True
            self.report_button.visible = True

            self.status_text.value = "Análise completa! Use os controles abaixo."
            self.update_frame_display(0)
        else:
            self.status_text.value = "Erro: Não foi possível processar os vídeos."

        self.page.update()

    # ... (demais métodos como on_pick_file_result, on_report_saved, etc., permanecem os mesmos) ...
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
            ft.Icons.PAUSE if self.is_playing else ft.Icons.PLAY_ARROW
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
        self.play_button.icon = ft.Icons.PLAY_ARROW
        self.page.update()
        logger.info("Reprodução automática finalizada.")

    def prev_frame(self, e):
        new_index = max(0, int(self.slider_control.value) - 1)
        self.update_frame_display(new_index)

    def next_frame(self, e):
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        new_index = min(num_frames - 1, int(self.slider_control.value) + 1)
        self.update_frame_display(new_index)

    def on_generate_report_click(self, e):
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        file_name = f"relatorio_krav_maga_{timestamp}.pdf"
        self.save_file_picker.save_file(
            dialog_title="Salvar Relatório de Análise",
            file_name=file_name,
            allowed_extensions=["pdf"],
        )

    def on_report_saved(self, e: ft.FilePickerResultEvent):
        """Gera e salva o relatório PDF com feedback visual colorido."""
        if e.path:
            save_path = e.path
            scores = [res["score"] for res in self.video_analyzer.comparison_results]

            frame_aluno_melhor_raw, frame_mestre_melhor_raw = (
                self.video_analyzer.get_best_frames()
            )
            frame_aluno_pior_raw, frame_mestre_pior_raw = (
                self.video_analyzer.get_worst_frames()
            )

            if frame_aluno_melhor_raw is not None and frame_aluno_pior_raw is not None:
                _, frame_aluno_melhor_color = (
                    self.video_analyzer.pose_estimator.estimate_pose(
                        frame_aluno_melhor_raw,
                        style=self.video_analyzer.pose_estimator.correct_style,
                    )
                )
                _, frame_mestre_melhor_color = (
                    self.video_analyzer.pose_estimator.estimate_pose(
                        frame_mestre_melhor_raw,
                        style=self.video_analyzer.pose_estimator.correct_style,
                    )
                )
                _, frame_aluno_pior_color = (
                    self.video_analyzer.pose_estimator.estimate_pose(
                        frame_aluno_pior_raw,
                        style=self.video_analyzer.pose_estimator.incorrect_style,
                    )
                )
                _, frame_mestre_pior_color = (
                    self.video_analyzer.pose_estimator.estimate_pose(
                        frame_mestre_pior_raw,
                        style=self.video_analyzer.pose_estimator.incorrect_style,
                    )
                )

                generator = ReportGenerator(
                    scores,
                    self.video_analyzer.comparison_results,
                    frame_aluno_melhor_color,
                    frame_mestre_melhor_color,
                    frame_aluno_pior_color,
                    frame_mestre_pior_color,
                )
                success, error_message = generator.generate(save_path)

                if success:
                    snack_bar = ft.SnackBar(
                        ft.Text(f"Relatório salvo com sucesso!"),
                        bgcolor=ft.Colors.GREEN,
                    )
                    try:
                        os.startfile(save_path)
                    except AttributeError:
                        logger.warning(
                            "os.startfile() não está disponível neste sistema operacional. O arquivo não será aberto automaticamente."
                        )
                else:
                    snack_bar = ft.SnackBar(
                        ft.Text(f"Erro ao salvar relatório: {error_message}"),
                        bgcolor=ft.Colors.RED,
                    )

                self.page.snack_bar = snack_bar
                self.page.snack_bar.open = True
                self.page.update()


def main(page: ft.Page):
    logger.info("Iniciando a aplicação Flet.")
    KravMagaApp(page)


if __name__ == "__main__":
    ft.app(target=main)
