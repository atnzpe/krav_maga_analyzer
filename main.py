# main.py

import flet as ft
import logging
import os
import cv2
import base64
import threading
import time
import sys
from datetime import datetime

# Garante que os módulos do projeto possam ser importados corretamente.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import setup_logging, get_logger
from src.video_analyzer import VideoAnalyzer
from src.report_generator import ReportGenerator

# Configura o sistema de logging para a aplicação.
setup_logging()
# Obtém uma instância do logger para este módulo.
logger = get_logger(__name__)


class KravMagaApp:
    """
    Encapsula toda a lógica e a interface do usuário da aplicação.
    """

    def __init__(self, page: ft.Page):
        """
        Construtor da classe da aplicação.

        Args:
            page (ft.Page): A página principal do Flet onde a UI será construída.
        """
        # --- Atributos de Estado ---
        self.page = page  # A página Flet principal.
        self.video_analyzer = None  # Instância do analisador de vídeo.
        self.is_playing = False  # Flag para controlar a reprodução do vídeo.
        self.playback_thread = None  # Thread para a reprodução automática.

        # ALTERAÇÃO: Variáveis de estado para os caminhos dos vídeos na sessão atual.
        # Estas variáveis são zeradas a cada nova instância da classe, resolvendo o bug
        # de "vídeo pré-carregado".
        self.video_aluno_path = None
        self.video_mestre_path = None
        logger.info(
            "Variáveis de estado da sessão (video_aluno_path, video_mestre_path) inicializadas como None."
        )

        # --- Construção da UI ---
        self.setup_controls()  # Inicializa todos os widgets Flet.
        self.build_layout()  # Monta o layout da página.
        logger.info("Aplicação Flet e UI inicializadas.")

    def setup_controls(self):
        """Inicializa todos os widgets Flet que compõem a interface."""
        # Controle para exibir mensagens de status e feedback ao usuário.
        self.status_text = ft.Text(
            "Por favor, carregue os vídeos para iniciar.",
            text_align=ft.TextAlign.CENTER,
            size=16,
        )
        # Botão para iniciar a análise, inicialmente desabilitado.
        self.analyze_button = ft.ElevatedButton(
            "Analisar Movimentos",
            icon=ft.Icons.ANALYTICS,
            on_click=self.analyze_videos,
            disabled=True,
        )

        # Barra de progresso para a análise.
        self.progress_bar = ft.ProgressBar(width=400, visible=False)

        # Controles de imagem para exibir os frames processados.
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

        # Placeholders exibidos antes do carregamento dos vídeos.
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

        # Slider para navegar entre os frames do vídeo.
        self.slider_control = ft.Slider(
            min=0,
            max=0,
            divisions=1,
            value=0,
            disabled=True,
            on_change=self.on_slider_change,
            expand=True,
        )

        # Botões de controle de reprodução.
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

        # Botão para gerar o relatório em PDF.
        self.report_button = ft.ElevatedButton(
            "Gerar Relatório PDF",
            icon=ft.Icons.PICTURE_AS_PDF,
            on_click=self.on_generate_report_click,
            visible=False,
        )

        # Seletores de arquivo (FilePicker) para upload.
        self.file_picker_aluno = ft.FilePicker(on_result=self.on_pick_file_result_aluno)
        self.file_picker_mestre = ft.FilePicker(
            on_result=self.on_pick_file_result_mestre
        )
        self.save_file_picker = ft.FilePicker(on_result=self.on_report_saved)
        # Adiciona os FilePickers à camada de sobreposição da página.
        self.page.overlay.extend(
            [self.file_picker_aluno, self.file_picker_mestre, self.save_file_picker]
        )
        logger.info("Controles da UI Flet foram inicializados.")

    def build_layout(self):
        """Constrói o layout visual da aplicação, organizando os controles na página."""
        self.page.title = "Analisador de Movimentos de Krav Maga"
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.scroll = ft.ScrollMode.ADAPTIVE
        self.page.theme_mode = ft.ThemeMode.DARK

        # Adiciona a estrutura principal de colunas e linhas à página.
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
        logger.info("Layout da UI construído e renderizado.")

    # --- Lógica de Upload ---

    def on_pick_file_result_aluno(self, e: ft.FilePickerResultEvent):
        """Callback para o seletor de arquivo do aluno."""
        logger.debug("Callback on_pick_file_result_aluno acionado.")
        self.pick_file_result(e, is_aluno=True)

    def on_pick_file_result_mestre(self, e: ft.FilePickerResultEvent):
        """Callback para o seletor de arquivo do mestre."""
        logger.debug("Callback on_pick_file_result_mestre acionado.")
        self.pick_file_result(e, is_aluno=False)

    def pick_file_result(self, e: ft.FilePickerResultEvent, is_aluno: bool):
        """
        Lógica central para lidar com o resultado da seleção de um arquivo.

        Args:
            e (ft.FilePickerResultEvent): O evento retornado pelo FilePicker.
            is_aluno (bool): True se o arquivo for do aluno, False se for do mestre.
        """
        # Define quem é o "dono" do vídeo para as mensagens e variáveis.
        video_owner = "aluno" if is_aluno else "mestre"

        # Se nenhum arquivo for selecionado (o usuário cancelou), exibe uma mensagem.
        if not e.files:
            logger.warning(f"Nenhum arquivo selecionado para o {video_owner}.")
            self.status_text.value = f"Nenhum vídeo do {video_owner} selecionado."
            self.page.update()
            return

        # Pega o caminho do arquivo selecionado.
        video_path = e.files[0].path

        # ALTERAÇÃO: Atualiza a variável de estado da SESSÃO ATUAL.
        if is_aluno:
            self.video_aluno_path = video_path
        else:
            self.video_mestre_path = video_path

        logger.info(
            f"Caminho do vídeo do {video_owner} definido na sessão para: {video_path}"
        )

        # ALTERAÇÃO: Mensagem de sucesso específica para cada upload.
        self.status_text.value = f"Vídeo do {video_owner} carregado com sucesso."
        logger.info(self.status_text.value)

        # Chama a função que verifica se ambos os vídeos foram carregados.
        self.update_analyze_button_state()

    def update_analyze_button_state(self):
        """
        Verifica se ambos os vídeos foram carregados NA SESSÃO ATUAL e habilita/desabilita
        o botão "Analisar Movimentos" de acordo.
        """
        # ALTERAÇÃO: A lógica agora depende das variáveis de estado da instância,
        # não mais do client_storage.
        if self.video_aluno_path and self.video_mestre_path:
            self.analyze_button.disabled = False
            self.status_text.value = "Vídeos carregados. Pronto para analisar."
            logger.info(
                "Ambos os vídeos foram carregados. Botão de análise HABILITADO."
            )
        else:
            self.analyze_button.disabled = True
            logger.info(
                "Ainda falta um ou mais vídeos. Botão de análise permanece DESABILITADO."
            )
            # A mensagem de status já foi atualizada pelo pick_file_result, então não a alteramos aqui
            # a menos que queiramos um feedback diferente.

        self.page.update()

    # --- Lógica de Análise (Inalterada, mas com logging revisado) ---

    def analyze_videos(self, e):
        """Inicia a análise dos vídeos em uma thread separada para não travar a UI."""
        logger.info(
            "Botão 'Analisar Movimentos' clicado. Iniciando processo de análise."
        )
        self.status_text.value = "Análise em andamento..."
        self.analyze_button.disabled = True
        self.progress_bar.value = 0
        self.progress_bar.visible = True
        self.page.update()

        # Usa os caminhos das variáveis de estado da sessão.
        aluno_path = self.video_aluno_path
        mestre_path = self.video_mestre_path

        self.video_analyzer = VideoAnalyzer()
        try:
            logger.info(f"Lendo bytes do arquivo do aluno: {aluno_path}")
            with open(aluno_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=True)

            logger.info(f"Lendo bytes do arquivo do mestre: {mestre_path}")
            with open(mestre_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=False)

            # Inicia a análise em uma thread.
            self.video_analyzer.analyze_and_compare(
                post_analysis_callback=self.setup_ui_post_analysis,
                progress_callback=self.update_progress,
            )
        except Exception as ex:
            logger.error(f"Falha ao carregar ou analisar vídeos: {ex}", exc_info=True)
            self.status_text.value = f"Erro ao processar os arquivos: {ex}"
            self.progress_bar.visible = False
            self.page.update()

    # ... (O restante dos métodos como update_progress, setup_ui_post_analysis, on_slider_change, etc., permanecem os mesmos e foram omitidos por brevidade) ...

    def update_progress(self, percent_complete):
        """Callback para atualizar a barra de progresso na UI."""
        self.progress_bar.value = percent_complete
        self.status_text.value = f"Analisando... {int(percent_complete * 100)}%"
        # Log de progresso pode ser muito verboso, então é opcional.
        # logger.debug(f"Progresso da análise: {int(percent_complete * 100)}%")
        self.page.update()

    def setup_ui_post_analysis(self):
        """Configura a UI após a conclusão da análise."""
        logger.info("Configurando a UI para exibir os resultados da análise.")
        num_frames = len(self.video_analyzer.processed_frames_aluno)

        self.progress_bar.visible = False

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
            logger.error("Análise concluída, mas nenhum frame foi processado.")

        self.page.update()

    def on_slider_change(self, e):
        """Callback acionado quando o valor do slider é alterado."""
        self.update_frame_display(int(e.control.value))

    def update_frame_display(self, frame_index):
        """Atualiza as imagens dos vídeos para um frame específico."""
        if not self.video_analyzer or frame_index >= len(
            self.video_analyzer.processed_frames_aluno
        ):
            return

        self.slider_control.value = frame_index

        # Converte os frames de numpy array para base64 e atualiza os controles de imagem.
        self.img_aluno_control.src_base64 = self.frame_to_base64(
            self.video_analyzer.processed_frames_aluno[frame_index]
        )
        self.img_mestre_control.src_base64 = self.frame_to_base64(
            self.video_analyzer.processed_frames_mestre[frame_index]
        )

        # Esconde os placeholders e mostra as imagens.
        self.aluno_placeholder.visible = False
        self.mestre_placeholder.visible = False
        self.img_aluno_control.visible = True
        self.img_mestre_control.visible = True

        self.page.update()

    def frame_to_base64(self, frame):
        """Converte um frame do OpenCV (numpy array) para uma string base64."""
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
        """Loop que executa em uma thread para reproduzir os frames sequencialmente."""
        start_index = int(self.slider_control.value)
        num_frames = len(self.video_analyzer.processed_frames_aluno)

        for i in range(start_index, num_frames):
            if not self.is_playing:
                break
            self.update_frame_display(i)
            time.sleep(1 / 30)  # Simula uma reprodução a 30 FPS.

        self.is_playing = False
        self.play_button.icon = ft.Icons.PLAY_ARROW
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

    def on_generate_report_click(self, e):
        """Abre o diálogo para salvar o relatório em PDF."""
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        file_name = f"relatorio_krav_maga_{timestamp}.pdf"
        self.save_file_picker.save_file(
            dialog_title="Salvar Relatório de Análise",
            file_name=file_name,
            allowed_extensions=["pdf"],
        )

    def on_report_saved(self, e: ft.FilePickerResultEvent):
        """Gera e salva o relatório PDF após o usuário escolher o local."""
        if not e.path:
            logger.warning("Operação de salvar relatório foi cancelada pelo usuário.")
            return

        save_path = e.path
        logger.info(f"Tentando salvar relatório em: {save_path}")

        scores = [res["score"] for res in self.video_analyzer.comparison_results]
        frame_aluno_melhor, frame_mestre_melhor = self.video_analyzer.get_best_frames()
        frame_aluno_pior, frame_mestre_pior = self.video_analyzer.get_worst_frames()

        if frame_aluno_melhor is not None and frame_aluno_pior is not None:
            generator = ReportGenerator(
                scores,
                self.video_analyzer.comparison_results,
                frame_aluno_melhor,
                frame_mestre_melhor,
                frame_aluno_pior,
                frame_mestre_pior,
            )
            success, error_message = generator.generate(save_path)

            if success:
                snack_bar = ft.SnackBar(
                    ft.Text("Relatório salvo com sucesso!"), bgcolor=ft.Colors.GREEN
                )
                try:
                    os.startfile(save_path)
                except Exception as ex:
                    logger.warning(
                        f"Não foi possível abrir o arquivo automaticamente: {ex}"
                    )
            else:
                snack_bar = ft.SnackBar(
                    ft.Text(f"Erro ao salvar: {error_message}"), bgcolor=ft.Colors.RED
                )

            self.page.snack_bar = snack_bar
            self.page.snack_bar.open = True
            self.page.update()


def main(page: ft.Page):
    """Função principal que inicia a aplicação Flet."""
    logger.info("Iniciando a aplicação Flet Krav Maga Analyzer.")
    KravMagaApp(page)


if __name__ == "__main__":
    # Ponto de entrada para executar a aplicação.
    ft.app(target=main)
