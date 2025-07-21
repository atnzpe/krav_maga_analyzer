# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------------
#  Krav Maga Motion Analyzer
#  version 1.1.0
#  Copyright (C) 2024,
#  開発者名 [Sujeito Programador]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------
import flet as ft  # Biblioteca Flet para criar a interface gráfica.
import cv2  # Biblioteca OpenCV para converter imagens para formatos que o Flet entende.
import base64  # Para codificar as imagens em base64 e exibi-las no Flet.
import logging  # Para registrar logs.
from src.video_analyzer import VideoAnalyzer  # Importa nossa classe de análise.

# --------------------------------------------------------------------------------------------------
# Configuração do Logging
# --------------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------------------------------------------------------
# Classe Principal da Aplicação Flet
# --------------------------------------------------------------------------------------------------


class KravMagaApp:
    """
    Classe principal que constrói e gerencia a interface do usuário com Flet.
    """

    def __init__(self, page: ft.Page):
        """
        Inicializador da aplicação.

        Args:
            page (ft.Page): O objeto da página principal do Flet.
        """
        self.page = page
        self.setup_page()  # Configurações iniciais da janela/página.

        self.video_analyzer = None  # O analisador será criado após o upload dos vídeos.
        self.video_path_aluno = None  # Caminho do vídeo do aluno.
        self.video_path_mestre = None  # Caminho do vídeo do mestre.

        # --- Controles da UI (Widgets do Flet) ---
        # Pickers para seleção de arquivos.
        self.file_picker_aluno = ft.FilePicker(on_result=self.on_file_aluno_selected)
        self.file_picker_mestre = ft.FilePicker(on_result=self.on_file_mestre_selected)
        self.page.overlay.extend([self.file_picker_aluno, self.file_picker_mestre])

        # Exibidores de imagem para os vídeos.
        self.image_aluno = ft.Image(width=400, height=400)
        self.image_mestre = ft.Image(width=400, height=400)

        # Slider para navegar pelos frames do vídeo.
        self.slider = ft.Slider(
            min=0,
            max=100,
            divisions=100,
            label="{value}%",
            on_change=self.on_slider_change,
            visible=False,
        )

        # Botões e textos informativos.
        self.analyze_button = ft.Button(
            text="Analisar Movimentos", on_click=self.analyze, disabled=True
        )
        self.status_text = ft.Text("Por favor, carregue o vídeo do aluno e do mestre.")

        # --- NOVOS CONTROLES PARA FEEDBACK ---
        self.score_text = ft.Text(
            "Pontuação: -",
            size=24,
            weight=ft.FontWeight.BOLD,
            color=ft.colors.WHITE,
            visible=False,
        )
        self.feedback_text = ft.Text(
            "Dica: -", size=18, color=ft.colors.AMBER, visible=False
        )
        # ------------------------------------

        self.build_layout()  # Monta a UI.
        logging.info("Aplicação Flet inicializada e layout construído.")

    def setup_page(self):
        """Configura as propriedades da página principal."""
        self.page.title = "Analisador de Movimentos de Krav Maga"
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.theme_mode = (
            ft.ThemeMode.DARK
        )  # Zen: "A escuridão na borda da cidade."

    def build_layout(self):
        """Constrói o layout da interface com os controles Flet."""
        logging.info("Construindo o layout da UI.")
        # O layout é organizado em Colunas e Linhas para posicionar os elementos.
        self.page.add(
            ft.Column(
                [
                    ft.Text(
                        "Analisador de Movimentos de Krav Maga",
                        size=32,
                        weight=ft.FontWeight.BOLD,
                    ),
                    self.status_text,
                    ft.Row(
                        [
                            ft.ElevatedButton(
                                "Carregar Vídeo do Aluno",
                                on_click=lambda _: self.file_picker_aluno.pick_files(
                                    allow_multiple=False, allowed_extensions=["mp4"]
                                ),
                            ),
                            ft.ElevatedButton(
                                "Carregar Vídeo do Mestre",
                                on_click=lambda _: self.file_picker_mestre.pick_files(
                                    allow_multiple=False, allowed_extensions=["mp4"]
                                ),
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    self.analyze_button,
                    ft.Row(
                        [
                            ft.Column([ft.Text("Aluno"), self.image_aluno]),
                            ft.Column([ft.Text("Mestre"), self.image_mestre]),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                    # --- Adicionando novos widgets de feedback ao layout ---
                    self.score_text,
                    self.feedback_text,
                    # ----------------------------------------------------
                    self.slider,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20,
            )
        )
        self.page.update()

    def on_file_aluno_selected(self, e: ft.FilePickerResultEvent):
        """Callback executado quando o arquivo do aluno é selecionado."""
        if e.files:
            self.video_path_aluno = e.files[0].path
            self.status_text.value = f"Aluno: {e.files[0].name}"
            logging.info(f"Vídeo do aluno selecionado: {self.video_path_aluno}")
            self.check_files_loaded()
        self.page.update()

    def on_file_mestre_selected(self, e: ft.FilePickerResultEvent):
        """Callback executado quando o arquivo do mestre é selecionado."""
        if e.files:
            self.video_path_mestre = e.files[0].path
            self.status_text.value += f" | Mestre: {e.files[0].name}"
            logging.info(f"Vídeo do mestre selecionado: {self.video_path_mestre}")
            self.check_files_loaded()
        self.page.update()

    def check_files_loaded(self):
        """Verifica se ambos os vídeos foram carregados e habilita o botão de análise."""
        if self.video_path_aluno and self.video_path_mestre:
            self.analyze_button.disabled = False
            self.status_text.value = "Vídeos prontos! Clique em 'Analisar Movimentos'."
            logging.info(
                "Ambos os vídeos foram carregados. Botão de análise habilitado."
            )
        self.page.update()

    def analyze(self, e):
        """Inicia o processo de análise quando o botão é clicado."""
        logging.info("Botão de análise clicado.")
        self.status_text.value = "Analisando... por favor, aguarde."
        self.analyze_button.disabled = True
        self.page.update()

        # Instancia e inicia o analisador de vídeo.
        self.video_analyzer = VideoAnalyzer(
            self.video_path_aluno, self.video_path_mestre
        )
        self.video_analyzer.start_analysis()

        # Verifica periodicamente se a análise terminou.
        self.page.run_thread(self.wait_for_analysis_completion)

    def wait_for_analysis_completion(self):
        """Espera a thread de análise terminar e então configura a UI pós-análise."""
        self.video_analyzer.processing_thread.join()  # Espera a thread finalizar.
        logging.info("Análise concluída. Configurando o slider e a visualização.")

        # Configura o slider com o número correto de frames.
        frame_count = self.video_analyzer.get_frame_count()
        if frame_count > 0:
            self.slider.max = frame_count - 1
            self.slider.divisions = frame_count - 1
            self.slider.visible = True
            # --- Torna os widgets de feedback visíveis ---
            self.score_text.visible = True
            self.feedback_text.visible = True
            # ------------------------------------------
            self.status_text.value = "Análise completa! Use o slider para navegar."
            self.update_frame_display(0)  # Exibe o primeiro frame.
        else:
            self.status_text.value = "Erro: Não foi possível processar os vídeos."

        self.page.update()

    def on_slider_change(self, e):
        """Callback executado quando o valor do slider muda."""
        frame_index = int(e.control.value)
        logging.debug(f"Slider movido para o frame {frame_index}.")
        self.update_frame_display(frame_index)

    def update_frame_display(self, frame_index):
        """Atualiza as imagens e os textos de feedback na tela."""
        if self.video_analyzer:
            # Pega todos os dados para o frame atual.
            frame_aluno, frame_mestre, score, feedback = (
                self.video_analyzer.get_data_at_frame(frame_index)
            )

            if frame_aluno is not None and frame_mestre is not None:
                # Converte os frames (arrays numpy) para strings base64 que o Flet pode exibir.
                self.image_aluno.src_base64 = self.frame_to_base64(frame_aluno)
                self.image_mestre.src_base64 = self.frame_to_base64(frame_mestre)

                # --- ATUALIZA A UI COM OS NOVOS DADOS ---
                self.score_text.value = f"Pontuação: {score:.2f}%"
                self.feedback_text.value = f"Dica: {feedback}"
                # Muda a cor da pontuação com base no valor para um feedback visual rápido.
                if score >= 85:
                    self.score_text.color = ft.colors.GREEN
                elif score >= 60:
                    self.score_text.color = ft.colors.ORANGE
                else:
                    self.score_text.color = ft.colors.RED
                # -----------------------------------------

                self.page.update()

    def frame_to_base64(self, frame):
        """Converte um frame OpenCV para uma string base64."""
        # Codifica o frame para o formato JPEG em memória.
        _, buffer = cv2.imencode(".jpg", frame)
        # Converte os bytes do buffer para uma string base64.
        return base64.b64encode(buffer).decode("utf-8")


def main(page: ft.Page):
    """Função principal que inicia a aplicação Flet."""
    KravMagaApp(page)


# Ponto de entrada da aplicação.
if __name__ == "__main__":
    ft.app(target=main)
