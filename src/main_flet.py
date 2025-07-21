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
from flet.plotly_chart import PlotlyChart
import plotly.graph_objects as go

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

        Args:
            page (ft.Page): O objeto da página principal fornecido pelo Flet.
        """
        self.page = page
        self.video_analyzer = None
        
        # Inicializa todos os controles da UI como atributos da classe
        self.setup_controls()
        # Constrói o layout inicial.
        self.build_layout()
        logger.info("Aplicação Flet e UI inicializadas com sucesso.")

    def setup_controls(self):
        """Inicializa todos os widgets Flet que serão usados na UI."""
        logger.info("Inicializando todos os controles da UI.")
        
        self.status_text = ft.Text(
            "Por favor, carregue ambos os vídeos para iniciar a análise.",
            text_align=ft.TextAlign.CENTER,
            size=16
        )
        self.analyze_button = ft.ElevatedButton(
            "Analisar Movimentos",
            icon=ft.icons.ANALYTICS,
            on_click=self.analyze_videos,
            disabled=True
        )
        
        # CORREÇÃO: Os controles de imagem agora começam invisíveis.
        # Eles só aparecerão quando tiverem conteúdo, evitando o erro "src or src_base64 must be specified".
        self.img_aluno_control = ft.Image(fit=ft.ImageFit.CONTAIN, visible=False, border_radius=ft.border_radius.all(10))
        self.img_mestre_control = ft.Image(fit=ft.ImageFit.CONTAIN, visible=False, border_radius=ft.border_radius.all(10))
        
        # Placeholder que será mostrado antes da análise.
        self.aluno_placeholder = ft.Container(
            content=ft.ProgressRing(),
            width=500,
            height=400,
            bgcolor=ft.colors.BLACK26,
            border_radius=ft.border_radius.all(10),
            alignment=ft.alignment.center
        )
        self.mestre_placeholder = ft.Container(
            content=ft.ProgressRing(),
            width=500,
            height=400,
            bgcolor=ft.colors.BLACK26,
            border_radius=ft.border_radius.all(10),
            alignment=ft.alignment.center
        )
        
        self.slider_control = ft.Slider(
            min=0, max=0, divisions=1, value=0,
            disabled=True,
            visible=False, # Começa invisível
            on_change=self.on_slider_change
        )
        
        self.file_picker_aluno = ft.FilePicker(on_result=lambda e: self.pick_file_result(e, is_aluno=True))
        self.file_picker_mestre = ft.FilePicker(on_result=lambda e: self.pick_file_result(e, is_aluno=False))
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
                    ft.Text("Analisador de Movimentos de Krav Maga 🥋", size=28, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        [
                            ft.ElevatedButton("Upload Vídeo do Aluno", icon=ft.icons.UPLOAD_FILE, on_click=lambda _: self.file_picker_aluno.pick_files(allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"])),
                            ft.ElevatedButton("Upload Vídeo do Mestre", icon=ft.icons.UPLOAD_FILE, on_click=lambda _: self.file_picker_mestre.pick_files(allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"])),
                            self.analyze_button,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER, spacing=20
                    ),
                    ft.Container(content=self.status_text, padding=10),
                    
                    # CORREÇÃO DE LAYOUT: Usando ft.Row para garantir que os vídeos fiquem lado a lado.
                    # ft.Stack permite sobrepor o placeholder e a imagem, controlando a visibilidade.
                    ft.Row(
                        [
                            ft.Column(
                                [
                                    ft.Text("Vídeo do Aluno", weight=ft.FontWeight.BOLD),
                                    ft.Stack([self.aluno_placeholder, self.img_aluno_control])
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER
                            ),
                            ft.Column(
                                [
                                    ft.Text("Vídeo do Mestre", weight=ft.FontWeight.BOLD),
                                    ft.Stack([self.mestre_placeholder, self.img_mestre_control])
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        spacing=30
                    ),
                    self.slider_control,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15
            )
        )
        self.page.update()

    def pick_file_result(self, e: ft.FilePickerResultEvent, is_aluno: bool):
        """Manipulador de evento para quando um arquivo é selecionado."""
        if not e.files:
            logger.warning(f"Seleção de arquivo cancelada para {'aluno' if is_aluno else 'mestre'}.")
            return

        video_path = e.files[0].path
        storage_key = "video_aluno_path" if is_aluno else "video_mestre_path"
        self.page.client_storage.set(storage_key, video_path)
        logger.info(f"Caminho do vídeo {'aluno' if is_aluno else 'mestre'} salvo: {video_path}")
        
        self.update_status_and_button_state()
        self.page.update()
        
    def update_status_and_button_state(self):
        """Verifica se ambos os vídeos foram carregados e atualiza o estado do botão."""
        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")
        
        if aluno_path and mestre_path:
            self.analyze_button.disabled = False
            self.status_text.value = "Vídeos carregados. Pronto para analisar."
            logger.info("Ambos os vídeos foram selecionados. Botão de análise habilitado.")
        elif aluno_path:
            self.status_text.value = "Vídeo do aluno carregado. Aguardando vídeo do mestre."
        elif mestre_path:
            self.status_text.value = "Vídeo do mestre carregado. Aguardando vídeo do aluno."
        
        self.page.update()

    def analyze_videos(self, e):
        """Inicia a análise dos vídeos em uma thread."""
        logger.info("Botão de análise clicado. Iniciando processo.")
        self.status_text.value = "Análise em andamento, por favor aguarde..."
        self.analyze_button.disabled = True
        
        # Mostra os anéis de progresso
        self.aluno_placeholder.visible = True
        self.mestre_placeholder.visible = True
        self.img_aluno_control.visible = False
        self.img_mestre_control.visible = False
        
        self.page.update()
        
        aluno_path = self.page.client_storage.get("video_aluno_path")
        mestre_path = self.page.client_storage.get("video_mestre_path")
        
        self.video_analyzer = VideoAnalyzer()
        
        # Carrega os vídeos usando a lógica existente.
        try:
            with open(aluno_path, 'rb') as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=True)
            with open(mestre_path, 'rb') as f:
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
        
        # CORREÇÃO: Usa 'page.run_thread_safe' para atualizar a UI a partir de uma thread.
        # Este é o método correto e seguro para comunicação entre threads no Flet.
        if self.page:
            self.page.run_thread_safe(self.setup_ui_post_analysis)

    def setup_ui_post_analysis(self):
        """Configura a UI após a conclusão da análise."""
        logger.info("Configurando a UI para exibir os resultados da análise.")
        num_frames = len(self.video_analyzer.aluno_landmarks)
        if num_frames > 0:
            self.slider_control.max = num_frames - 1
            self.slider_control.divisions = num_frames - 1 if num_frames > 1 else 1
            self.slider_control.disabled = False
            self.slider_control.visible = True
            
            self.status_text.value = "Análise completa! Use o slider para navegar pelos frames."
            self.update_frame_display(0)
        else:
            self.status_text.value = "Erro: Não foi possível extrair dados dos vídeos para análise."
        
        self.page.update()

    def on_slider_change(self, e):
        """Atualiza a exibição do frame quando o slider é movido."""
        frame_index = int(e.control.value)
        logger.debug(f"Slider movido para o frame {frame_index}.")
        self.update_frame_display(frame_index)

    def update_frame_display(self, frame_index):
        """Busca os frames processados e os exibe na UI."""
        if not self.video_analyzer: return

        def get_annotated_frame(is_aluno):
            """Função auxiliar para ler e anotar um frame específico."""
            cap = self.video_analyzer.cap_aluno if is_aluno else self.video_analyzer.cap_mestre
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    _, annotated_image = self.video_analyzer.pose_estimator.estimate_pose(frame)
                    return self.frame_to_base64(annotated_image)
            return None

        # Obtém os frames e os exibe
        base64_aluno = get_annotated_frame(is_aluno=True)
        base64_mestre = get_annotated_frame(is_aluno=False)
        
        if base64_aluno and base64_mestre:
            self.img_aluno_control.src_base64 = base64_aluno
            self.img_mestre_control.src_base64 = base64_mestre
            
            # Garante que os placeholders desapareçam e as imagens apareçam
            self.aluno_placeholder.visible = False
            self.mestre_placeholder.visible = False
            self.img_aluno_control.visible = True
            self.img_mestre_control.visible = True
        else:
            logger.error(f"Não foi possível obter os frames para o índice {frame_index}.")

        self.page.update()

    def frame_to_base64(self, frame):
        """Converte um frame do OpenCV para uma string base64."""
        _, buffer = cv2.imencode('.png', frame)
        return base64.b64encode(buffer).decode('utf-8')

def main(page: ft.Page):
    """Função de entrada que o Flet chama para iniciar a aplicação."""
    logger.info("Iniciando a aplicação Flet.")
    KravMagaApp(page)

if __name__ == "__main__":
    ft.app(target=main)