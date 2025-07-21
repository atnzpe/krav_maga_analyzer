# src/main_flet.py

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas Essenciais
# --------------------------------------------------------------------------------------------------
import flet as ft  # Framework principal para a construção da interface gráfica.
import logging  # Para registrar eventos e facilitar a depuração.
import os  # Para interagir com o sistema operacional, como obter caminhos de arquivos.
import cv2  # OpenCV para manipulação de vídeo e imagem.
import base64  # Para codificar imagens para exibição no Flet.
import threading  # Para executar a análise de vídeo em uma thread separada e não travar a UI.
import sys  # Para manipular o path do sistema e permitir importações de outros diretórios.

# --- Adiciona o diretório raiz do projeto ao path para garantir que as importações funcionem ---
# Zen of Python: "Flat is better than nested." - Simplifica a estrutura de importação.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Importação dos Módulos do Projeto ---
from src.utils import (
    setup_logging,
    get_logger,
)  # Funções utilitárias, incluindo nosso logger.
from src.video_analyzer import (
    VideoAnalyzer,
)  # Classe principal para a lógica de análise.

# Plotly é usado para os gráficos, embora a lógica principal do gráfico não esteja neste arquivo.
from flet.plotly_chart import PlotlyChart
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------------------
# Configuração Inicial
# --------------------------------------------------------------------------------------------------
# Configura e obtém uma instância do logger para este arquivo.
setup_logging()
logger = get_logger(__name__)


# --------------------------------------------------------------------------------------------------
# Classe Principal da Aplicação Flet
# --------------------------------------------------------------------------------------------------
class KravMagaApp:
    """
    Classe que encapsula toda a lógica e a interface do usuário da aplicação.
    Zen of Python: "Encapsulation is a good thing." (Implicitamente)
    """

    def __init__(self, page: ft.Page):
        """
        Inicializador da classe da aplicação.

        Args:
            page (ft.Page): O objeto da página principal fornecido pelo Flet.
        """
        self.page = page
        self.video_analyzer = None  # Será instanciado após o upload dos vídeos.

        # --- CORREÇÃO PRINCIPAL: Inicializa todos os controles da UI como atributos da classe ---
        # Isso garante que eles possam ser acessados de qualquer método da classe usando 'self'.
        self.status_text = ft.Text(
            "Por favor, carregue ambos os vídeos para iniciar a análise.",
            text_align=ft.TextAlign.CENTER,
        )
        self.analyze_button = ft.ElevatedButton(
            "Analisar Movimentos",
            icon=ft.icons.ANALYTICS,
            on_click=self.analyze_videos,
            disabled=True,
        )
        self.img_aluno_control = ft.Image(
            fit=ft.ImageFit.CONTAIN, expand=True, src_base64=""
        )
        self.img_mestre_control = ft.Image(
            fit=ft.ImageFit.CONTAIN, expand=True, src_base64=""
        )
        self.slider_control = ft.Slider(
            min=0,
            max=0,
            divisions=1,
            value=0,
            expand=True,
            disabled=True,
            on_change=self.on_slider_change,
        )

        # Configura os seletores de arquivo (FilePickers).
        self.file_picker_aluno = ft.FilePicker(
            on_result=lambda e: self.pick_file_result(e, is_aluno=True)
        )
        self.file_picker_mestre = ft.FilePicker(
            on_result=lambda e: self.pick_file_result(e, is_aluno=False)
        )
        self.page.overlay.extend([self.file_picker_aluno, self.file_picker_mestre])

        # Constrói o layout inicial.
        self.build_layout()
        logger.info("Aplicação Flet e UI inicializadas com sucesso.")

    def build_layout(self):
        """
        Constrói o layout visual da aplicação, adicionando os controles à página.
        """
        logger.info("Construindo o layout da UI...")
        self.page.title = "Analisador de Movimentos de Krav Maga"
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.scroll = ft.ScrollMode.ADAPTIVE

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
                    ft.ResponsiveRow(
                        [
                            ft.Column(
                                [
                                    ft.Text(
                                        "Vídeo do Aluno", weight=ft.FontWeight.BOLD
                                    ),
                                    self.img_aluno_control,
                                ],
                                col={"md": 6},
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        "Vídeo do Mestre", weight=ft.FontWeight.BOLD
                                    ),
                                    self.img_mestre_control,
                                ],
                                col={"md": 6},
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    self.slider_control,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            )
        )
        logger.info("Layout da UI construído e adicionado à página.")
        self.page.update()

    def pick_file_result(self, e: ft.FilePickerResultEvent, is_aluno: bool):
        """
        Manipulador de evento para quando um arquivo é selecionado pelo usuário.

        Args:
            e (ft.FilePickerResultEvent): O evento disparado pelo FilePicker.
            is_aluno (bool): Flag para identificar se o vídeo é do aluno ou do mestre.
        """
        if not e.files:
            logger.warning(
                f"Seleção de arquivo cancelada para {'aluno' if is_aluno else 'mestre'}."
            )
            return

        video_path = e.files[0].path

        # Armazena o caminho do arquivo no armazenamento do cliente do Flet (uma forma de guardar estado).
        storage_key = "video_aluno_path" if is_aluno else "video_mestre_path"
        self.page.client_storage.set(storage_key, video_path)
        logger.info(
            f"Caminho do vídeo {'aluno' if is_aluno else 'mestre'} salvo: {video_path}"
        )

        self.update_status_and_button_state()
        self.page.update()

    def update_status_and_button_state(self):
        """
        Verifica se ambos os vídeos foram carregados e atualiza o estado do botão de análise.
        """
        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")

        if aluno_path and mestre_path:
            # CORREÇÃO: Acessa o botão diretamente através de 'self.analyze_button'.
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
        """Inicia a análise dos vídeos em uma thread para não bloquear a UI."""
        logger.info("Botão de análise clicado. Iniciando processo.")
        self.status_text.value = "Análise em andamento, por favor aguarde..."
        # Desabilita o botão para evitar cliques múltiplos durante a análise.
        self.analyze_button.disabled = True
        self.page.update()

        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")

        # Cria uma nova instância do analisador para esta sessão de análise.
        self.video_analyzer = VideoAnalyzer()
        # Carrega os vídeos a partir dos caminhos (a lógica interna cria arquivos temporários).
        self.video_analyzer.load_video_from_bytes(
            open(aluno_path, "rb").read(), is_aluno=True
        )
        self.video_analyzer.load_video_from_bytes(
            open(mestre_path, "rb").read(), is_aluno=False
        )

        # Inicia a análise em uma thread separada.
        threading.Thread(target=self.run_analysis_and_update_ui, daemon=True).start()

    def run_analysis_and_update_ui(self):
        """
        Função que executa a análise e, ao final, chama a atualização da UI na thread principal.
        """
        # Este método bloqueia até que a análise na thread do VideoAnalyzer termine.
        self.video_analyzer.analyze_and_compare()
        if self.video_analyzer.processing_thread:
            self.video_analyzer.processing_thread.join()

        logger.info("Análise na thread concluída. Agendando atualização da UI.")
        # O Flet requer que as atualizações da UI sejam feitas a partir da thread principal.
        # self.page.run() agenda a execução da função na thread correta.
        self.page.run(self.setup_ui_post_analysis)

    def setup_ui_post_analysis(self):
        """Configura a UI após a conclusão da análise."""
        logger.info("Configurando a UI para exibir os resultados da análise.")
        num_frames = len(self.video_analyzer.aluno_landmarks)
        if num_frames > 0:
            self.slider_control.max = num_frames - 1
            self.slider_control.divisions = num_frames - 1 if num_frames > 1 else 1
            self.slider_control.disabled = False
            self.status_text.value = "Análise completa! Use o slider para navegar."
            self.update_frame_display(0)  # Exibe o primeiro frame.
        else:
            self.status_text.value = (
                "Não foi possível extrair dados dos vídeos para análise."
            )

        self.page.update()

    def on_slider_change(self, e):
        """Atualiza a exibição do frame quando o slider é movido."""
        frame_index = int(e.control.value)
        logger.debug(f"Slider movido para o frame {frame_index}.")
        self.update_frame_display(frame_index)

    def update_frame_display(self, frame_index):
        """
        Busca os frames processados e os exibe na UI.

        Args:
            frame_index (int): O índice do frame a ser exibido.
        """
        if not self.video_analyzer:
            return

        # Para exibir os frames, precisamos lê-los dos arquivos de vídeo novamente,
        # pois não os armazenamos em memória para economizar recursos.

        # Lógica para exibir frame do aluno
        if self.video_analyzer.cap_aluno.isOpened():
            self.video_analyzer.cap_aluno.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_analyzer.cap_aluno.read()
            if ret:
                results, annotated_image = (
                    self.video_analyzer.pose_estimator.estimate_pose(frame)
                )
                self.img_aluno_control.src_base64 = self.frame_to_base64(
                    annotated_image
                )

        # Lógica para exibir frame do mestre
        if self.video_analyzer.cap_mestre.isOpened():
            self.video_analyzer.cap_mestre.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.video_analyzer.cap_mestre.read()
            if ret:
                results, annotated_image = (
                    self.video_analyzer.pose_estimator.estimate_pose(frame)
                )
                self.img_mestre_control.src_base64 = self.frame_to_base64(
                    annotated_image
                )

        self.page.update()

    def frame_to_base64(self, frame):
        """Converte um frame do OpenCV para uma string base64 para exibição no Flet."""
        _, buffer = cv2.imencode(".png", frame)
        return base64.b64encode(buffer).decode("utf-8")


# --- Função Principal ---
def main(page: ft.Page):
    """Função de entrada que o Flet chama para iniciar a aplicação."""
    logger.info("Iniciando a aplicação Flet.")
    KravMagaApp(page)


# Ponto de entrada padrão para scripts Python.
if __name__ == "__main__":
    ft.app(target=main)
