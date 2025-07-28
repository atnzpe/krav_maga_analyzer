# main.py

import flet as ft
import logging
import os
import cv2
import base64
import threading
import time
import sys
import numpy as np
from datetime import datetime

# Garante que os módulos do projeto na pasta 'src' possam ser importados corretamente.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Importa os módulos customizados da aplicação.
from src.utils import setup_logging, get_logger
from src.video_analyzer import VideoAnalyzer
from src.report_generator import ReportGenerator
from src.renderer_3d import render_3d_skeleton # Importa a nova função de renderização 3D.

# Configura o sistema de logging para a aplicação. Deve ser chamado uma vez no início.
setup_logging()
# Obtém uma instância do logger específica para este módulo (main.py).
logger = get_logger(__name__)


class KravMagaApp:
    """
    Classe principal que encapsula toda a lógica e a interface do usuário (UI) da aplicação.
    """

    def __init__(self, page: ft.Page):
        """
        Construtor da classe da aplicação.

        Args:
            page (ft.Page): A página principal do Flet onde a UI será construída.
        """
        # --- Atributos de Estado da Aplicação ---
        self.page = page  # A página Flet principal, usada para desenhar e atualizar a UI.
        self.video_analyzer = None  # Instância do analisador de vídeo, criada ao clicar em "Analisar".
        self.is_playing = False  # Flag para controlar a reprodução automática do vídeo (Play/Pause).
        self.playback_thread = None  # Thread dedicada para a reprodução, para não travar a UI.

        # Variáveis de estado para os caminhos dos vídeos na sessão atual.
        # São zeradas a cada execução para evitar o bug de "vídeo pré-carregado".
        self.video_aluno_path = None
        self.video_mestre_path = None
        logger.info("Variáveis de estado da sessão (video_aluno_path, video_mestre_path) inicializadas como None.")

        # --- Construção da UI ---
        self.setup_controls()  # Inicializa todos os widgets Flet.
        self.build_layout()  # Monta o layout da página com os widgets.
        logger.info("Aplicação Flet e UI inicializadas.")

    def setup_controls(self):
        """
        Inicializa todos os widgets (controles) Flet que compõem a interface.
        Esta função apenas cria os objetos, eles são organizados na UI pelo `build_layout`.
        """
        # Controle para exibir mensagens de status e feedback ao usuário.
        self.status_text = ft.Text("Por favor, carregue os vídeos para iniciar.", text_align=ft.TextAlign.CENTER, size=16)
        
        # Botão para iniciar a análise, que começa desabilitado.
        self.analyze_button = ft.ElevatedButton("Analisar Movimentos", icon=ft.Icons.ANALYTICS, on_click=self.analyze_videos, disabled=True)
        
        # Barra de progresso para a análise.
        self.progress_bar = ft.ProgressBar(width=400, visible=False)

        # Controles de imagem para exibir os frames processados em 2D.
        self.img_aluno_control = ft.Image(fit=ft.ImageFit.CONTAIN, visible=False, border_radius=ft.border_radius.all(10))
        self.img_mestre_control = ft.Image(fit=ft.ImageFit.CONTAIN, visible=False, border_radius=ft.border_radius.all(10))

        # NOVO: Controles de imagem para a renderização 3D. Ficam invisíveis até serem ativados.
        self.img_aluno_3d_control = ft.Image(fit=ft.ImageFit.CONTAIN, visible=False, border_radius=ft.border_radius.all(10))
        self.img_mestre_3d_control = ft.Image(fit=ft.ImageFit.CONTAIN, visible=False, border_radius=ft.border_radius.all(10))

        # Placeholders exibidos antes do carregamento dos vídeos.
        self.aluno_placeholder = ft.Container(content=ft.Text("Vídeo do Aluno"), width=500, height=400, bgcolor=ft.Colors.BLACK26, border_radius=ft.border_radius.all(10), alignment=ft.alignment.center)
        self.mestre_placeholder = ft.Container(content=ft.Text("Vídeo do Mestre"), width=500, height=400, bgcolor=ft.Colors.BLACK26, border_radius=ft.border_radius.all(10), alignment=ft.alignment.center)

        # Slider para navegar entre os frames do vídeo.
        self.slider_control = ft.Slider(min=0, max=0, divisions=1, value=0, disabled=True, on_change=self.on_slider_change, expand=True)

        # Botões de controle de reprodução (Play, Pause, Anterior, Próximo).
        self.play_button = ft.IconButton(icon=ft.Icons.PLAY_ARROW, on_click=self.toggle_play_pause, tooltip="Reproduzir/Pausar")
        self.prev_frame_button = ft.IconButton(icon=ft.Icons.SKIP_PREVIOUS, on_click=self.prev_frame, tooltip="Frame Anterior")
        self.next_frame_button = ft.IconButton(icon=ft.Icons.SKIP_NEXT, on_click=self.next_frame, tooltip="Próximo Frame")

        # NOVO: Interruptor (Switch) para alternar entre as visualizações 2D e 3D.
        self.view_3d_switch = ft.Switch(label="Visualização 3D", value=False, on_change=self.toggle_3d_view, visible=False)
        
        # Agrupa os controles de playback em uma linha (Row).
        self.playback_controls = ft.Row([self.prev_frame_button, self.play_button, self.next_frame_button, self.slider_control], visible=False, alignment=ft.MainAxisAlignment.CENTER)
        
        # Botão para gerar o relatório em PDF.
        self.report_button = ft.ElevatedButton("Gerar Relatório PDF", icon=ft.Icons.PICTURE_AS_PDF, on_click=self.on_generate_report_click, visible=False)

        # Seletores de arquivo (FilePicker) para upload de vídeos e salvamento do relatório.
        self.file_picker_aluno = ft.FilePicker(on_result=self.on_pick_file_result_aluno)
        self.file_picker_mestre = ft.FilePicker(on_result=self.on_pick_file_result_mestre)
        self.save_file_picker = ft.FilePicker(on_result=self.on_report_saved)
        self.page.overlay.extend([self.file_picker_aluno, self.file_picker_mestre, self.save_file_picker])
        
        logger.info("Controles da UI Flet, incluindo a opção 3D, foram inicializados.")

    def build_layout(self):
        """Constrói o layout visual da aplicação, organizando os controles na página."""
        self.page.title = "Analisador de Movimentos de Krav Maga"
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.scroll = ft.ScrollMode.ADAPTIVE
        self.page.theme_mode = ft.ThemeMode.DARK

        # Adiciona a estrutura principal de colunas e linhas à página.
        self.page.add(
            ft.Column([
                ft.Text("Analisador de Movimentos de Krav Maga 🥋", size=28, weight=ft.FontWeight.BOLD),
                ft.Row([
                    ft.ElevatedButton("Upload Vídeo do Aluno", icon=ft.Icons.UPLOAD_FILE, on_click=lambda _: self.file_picker_aluno.pick_files(allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"])),
                    ft.ElevatedButton("Upload Vídeo do Mestre", icon=ft.Icons.UPLOAD_FILE, on_click=lambda _: self.file_picker_mestre.pick_files(allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"])),
                    self.analyze_button,
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
                ft.Container(content=self.status_text, padding=10),
                self.progress_bar,
                ft.ResponsiveRow([
                    ft.Column([
                        ft.Text("Visualização do Aluno", weight=ft.FontWeight.BOLD),
                        # O Stack permite sobrepor widgets. Colocamos o placeholder e as imagens 2D/3D no mesmo lugar.
                        ft.Stack([self.aluno_placeholder, self.img_aluno_control, self.img_aluno_3d_control])
                    ], col={"xs": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Column([
                        ft.Text("Visualização do Mestre", weight=ft.FontWeight.BOLD),
                        ft.Stack([self.mestre_placeholder, self.img_mestre_control, self.img_mestre_3d_control])
                    ], col={"xs": 12, "md": 6}, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                ], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.START),
                self.playback_controls,
                # A linha de controles agora inclui o interruptor 3D e o botão de relatório.
                ft.Row([self.view_3d_switch, self.report_button], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15)
        )
        self.page.update()
        logger.info("Layout da UI construído com placeholders para a visão 3D.")

    def on_pick_file_result_aluno(self, e: ft.FilePickerResultEvent):
        """Callback para o seletor de arquivo do aluno."""
        self.pick_file_result(e, is_aluno=True)

    def on_pick_file_result_mestre(self, e: ft.FilePickerResultEvent):
        """Callback para o seletor de arquivo do mestre."""
        self.pick_file_result(e, is_aluno=False)

    def pick_file_result(self, e: ft.FilePickerResultEvent, is_aluno: bool):
        """Lógica central para lidar com o resultado da seleção de um arquivo."""
        video_owner = "aluno" if is_aluno else "mestre"
        if not e.files:
            self.status_text.value = f"Nenhum vídeo do {video_owner} selecionado."
            logger.warning(f"Nenhum arquivo selecionado para o {video_owner}.")
            self.page.update()
            return

        video_path = e.files[0].path
        if is_aluno:
            self.video_aluno_path = video_path
        else:
            self.video_mestre_path = video_path
        
        self.status_text.value = f"Vídeo do {video_owner} carregado com sucesso."
        logger.info(f"Caminho do vídeo do {video_owner} definido na sessão para: {video_path}")
        self.update_analyze_button_state()

    def update_analyze_button_state(self):
        """Habilita o botão 'Analisar' somente quando ambos os vídeos são carregados."""
        if self.video_aluno_path and self.video_mestre_path:
            self.analyze_button.disabled = False
            self.status_text.value = "Vídeos carregados. Pronto para analisar."
            logger.info("Ambos os vídeos foram carregados. Botão de análise HABILITADO.")
        else:
            self.analyze_button.disabled = True
            logger.info("Ainda falta um ou mais vídeos. Botão de análise permanece DESABILITADO.")
        self.page.update()

    def analyze_videos(self, e):
        """Inicia a análise dos vídeos em uma thread separada para não travar a UI."""
        logger.info("Botão 'Analisar Movimentos' clicado. Iniciando processo de análise.")
        self.status_text.value = "Análise em andamento..."
        self.analyze_button.disabled = True
        self.progress_bar.value = 0
        self.progress_bar.visible = True
        self.page.update()

        self.video_analyzer = VideoAnalyzer()
        try:
            with open(self.video_aluno_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=True)
            with open(self.video_mestre_path, "rb") as f:
                self.video_analyzer.load_video_from_bytes(f.read(), is_aluno=False)
            
            self.video_analyzer.analyze_and_compare(
                post_analysis_callback=self.setup_ui_post_analysis,
                progress_callback=self.update_progress,
            )
        except Exception as ex:
            logger.error(f"Falha ao carregar ou analisar vídeos: {ex}", exc_info=True)
            self.status_text.value = f"Erro ao processar os arquivos: {ex}"
            self.progress_bar.visible = False
            self.page.update()

    def update_progress(self, percent_complete):
        """Callback para atualizar a barra de progresso na UI."""
        self.progress_bar.value = percent_complete
        self.status_text.value = f"Analisando... {int(percent_complete * 100)}%"
        self.page.update()

    def setup_ui_post_analysis(self):
        """Configura a UI para exibir os resultados após a conclusão da análise."""
        logger.info("Configurando a UI para exibir os resultados da análise.")
        num_frames = len(self.video_analyzer.processed_frames_aluno)
        self.progress_bar.visible = False

        if num_frames > 0:
            self.slider_control.max = num_frames - 1
            self.slider_control.divisions = num_frames - 1 if num_frames > 1 else 1
            self.slider_control.disabled = False
            self.playback_controls.visible = True
            self.report_button.visible = True
            self.view_3d_switch.visible = True # Torna o interruptor 3D visível.
            self.status_text.value = "Análise completa! Use os controles e o seletor 3D abaixo."
            self.update_frame_display(0)
        else:
            self.status_text.value = "Erro: Não foi possível processar os vídeos."
            logger.error("Análise concluída, mas nenhum frame foi processado.")
        self.page.update()

    def on_slider_change(self, e):
        """Callback acionado quando o valor do slider é alterado pelo usuário."""
        self.update_frame_display(int(e.control.value))

    def update_frame_display(self, frame_index):
        """
        Função central que atualiza as imagens exibidas para um frame específico, 
        alternando entre a visualização 2D (vídeo) e 3D (render) com base no interruptor.
        """
        frame_index = int(frame_index) # Garante que o índice é um inteiro.
        if not self.video_analyzer or frame_index >= len(self.video_analyzer.processed_frames_aluno):
            return
        
        self.slider_control.value = frame_index

        # Esconde os placeholders e mostra as áreas de visualização.
        self.aluno_placeholder.visible = False
        self.mestre_placeholder.visible = False

        # Verifica se o modo 3D está ativado.
        if self.view_3d_switch.value:
            # --- MODO 3D ---
            self.img_aluno_control.visible = False; self.img_mestre_control.visible = False
            self.img_aluno_3d_control.visible = True; self.img_mestre_3d_control.visible = True
            
            aluno_landmarks = self.video_analyzer.aluno_landmarks_list[frame_index]
            mestre_landmarks = self.video_analyzer.mestre_landmarks_list[frame_index]
            
            # Chama a função de renderização e atualiza o SRC da imagem com o resultado.
            self.img_aluno_3d_control.src_base64 = self.frame_to_base64(render_3d_skeleton(aluno_landmarks))
            self.img_mestre_3d_control.src_base64 = self.frame_to_base64(render_3d_skeleton(mestre_landmarks))
        else:
            # --- MODO 2D (Comportamento padrão) ---
            self.img_aluno_3d_control.visible = False; self.img_mestre_3d_control.visible = False
            self.img_aluno_control.visible = True; self.img_mestre_control.visible = True

            # Atualiza o SRC das imagens 2D com os frames pré-processados do vídeo.
            self.img_aluno_control.src_base64 = self.frame_to_base64(self.video_analyzer.processed_frames_aluno[frame_index])
            self.img_mestre_control.src_base64 = self.frame_to_base64(self.video_analyzer.processed_frames_mestre[frame_index])
        
        self.page.update()
    
    def toggle_3d_view(self, e):
        """Callback que redesenha a visão atual ao acionar o interruptor 3D."""
        logger.info(f"Visualização 3D alterada para: {self.view_3d_switch.value}")
        self.update_frame_display(self.slider_control.value)

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Converte um frame do OpenCV (numpy array) para uma string base64."""
        _, buffer = cv2.imencode(".png", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def toggle_play_pause(self, e):
        """Inicia ou pausa a reprodução automática dos frames."""
        self.is_playing = not self.is_playing
        self.play_button.icon = ft.Icons.PAUSE if self.is_playing else ft.Icons.PLAY_ARROW
        if self.is_playing:
            logger.info("Iniciando reprodução automática.")
            self.playback_thread = threading.Thread(target=self.play_video_loop, daemon=True)
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
        if self.is_playing:
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
        self.save_file_picker.save_file(dialog_title="Salvar Relatório de Análise", file_name=file_name, allowed_extensions=["pdf"])

    def on_report_saved(self, e: ft.FilePickerResultEvent):
        """Gera e salva o relatório PDF com feedback visual colorido e detalhado."""
        if not e.path:
            logger.warning("Operação de salvar relatório foi cancelada pelo usuário.")
            return
        save_path = e.path
        logger.info(f"Iniciando geração do relatório PDF para: {save_path}")

        try:
            scores = [res["score"] for res in self.video_analyzer.comparison_results]
            best_frame_index = np.argmax(scores) if scores else 0
            worst_frame_index = np.argmin(scores) if scores else 0
            logger.info(f"Melhor frame (índice {best_frame_index}) e pior frame (índice {worst_frame_index}) identificados.")

            # --- GERAÇÃO DAS IMAGENS DE FEEDBACK PARA O PDF ---
            # ... (código para obter dados dos frames)
            best_frame_aluno_raw = self.video_analyzer.raw_frames_aluno[best_frame_index]
            best_frame_mestre_raw = self.video_analyzer.raw_frames_mestre[best_frame_index]
            best_landmarks_aluno_list = self.video_analyzer.aluno_landmarks_list[best_frame_index]
            best_landmarks_mestre_raw = self.video_analyzer.mestre_landmarks_raw[best_frame_index]
            best_diffs = self.video_analyzer.comparison_results[best_frame_index]["diffs"]
            worst_frame_aluno_raw = self.video_analyzer.raw_frames_aluno[worst_frame_index]
            worst_frame_mestre_raw = self.video_analyzer.raw_frames_mestre[worst_frame_index]
            worst_landmarks_aluno_list = self.video_analyzer.aluno_landmarks_list[worst_frame_index]
            worst_landmarks_mestre_raw = self.video_analyzer.mestre_landmarks_raw[worst_frame_index]
            worst_diffs = self.video_analyzer.comparison_results[worst_frame_index]["diffs"]

            # Desenha os esqueletos de feedback (vermelho/verde) para o aluno.
            frame_aluno_melhor_feedback = self.video_analyzer.pose_estimator.draw_feedback_skeleton(best_frame_aluno_raw, best_landmarks_aluno_list, best_diffs, self.video_analyzer.motion_comparator.KEY_ANGLES)
            frame_aluno_pior_feedback = self.video_analyzer.pose_estimator.draw_feedback_skeleton(worst_frame_aluno_raw, worst_landmarks_aluno_list, worst_diffs, self.video_analyzer.motion_comparator.KEY_ANGLES)
            
            # O esqueleto do mestre é desenhado por lado (azul/laranja) como referência.
            frame_mestre_melhor_feedback = self.video_analyzer.pose_estimator.draw_skeleton_by_side(best_frame_mestre_raw, best_landmarks_mestre_raw)
            frame_mestre_pior_feedback = self.video_analyzer.pose_estimator.draw_skeleton_by_side(worst_frame_mestre_raw, worst_landmarks_mestre_raw)
            logger.info("Imagens de feedback com esqueletos coloridos foram geradas para o PDF.")

            generator = ReportGenerator(
                scores=scores,
                feedbacks=self.video_analyzer.comparison_results,
                frame_aluno_melhor=frame_aluno_melhor_feedback,
                frame_mestre_melhor=frame_mestre_melhor_feedback,
                frame_aluno_pior=frame_aluno_pior_feedback,
                frame_mestre_pior=frame_mestre_pior_feedback,
                best_angle_diffs=best_diffs,
                worst_angle_diffs=worst_diffs,
                key_angles_map=self.video_analyzer.motion_comparator.readable_angle_names,
            )
            success, error_message = generator.generate(save_path)

            if success:
                snack_bar = ft.SnackBar(ft.Text("Relatório salvo com sucesso!"), bgcolor=ft.Colors.GREEN)
                try: os.startfile(save_path)
                except Exception: logger.warning("os.startfile() não disponível. O arquivo não será aberto.")
            else:
                snack_bar = ft.SnackBar(ft.Text(f"Erro ao salvar: {error_message}"), bgcolor=ft.Colors.RED)
            
            self.page.snack_bar = snack_bar
            self.page.snack_bar.open = True
            self.page.update()

        except Exception as ex:
            logger.error(f"Ocorreu um erro inesperado ao gerar o relatório: {ex}", exc_info=True)
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Erro inesperado: {ex}"), bgcolor=ft.Colors.RED)
            self.page.snack_bar.open = True
            self.page.update()


def main(page: ft.Page):
    """Função principal que inicia a aplicação Flet."""
    logger.info("Iniciando a aplicação Flet Krav Maga Analyzer.")
    KravMagaApp(page)


if __name__ == "__main__":
    # Ponto de entrada para executar a aplicação.
    ft.app(target=main)