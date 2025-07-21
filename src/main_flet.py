# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------------
#  Krav Maga Motion Analyzer
#  version 1.2.0
#  Copyright (C) 2024,
#  開発者名 [Sujeito Programador]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------
import flet as ft
import cv2
import base64
import logging
from src.video_analyzer import VideoAnalyzer

# --- NOVA IMPORTAÇÃO PARA O RELATÓRIO ---
from src.report_generator import ReportGenerator

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
        """
        self.page = page
        self.setup_page()

        self.video_analyzer = None
        self.video_path_aluno = None
        self.video_path_mestre = None

        # --- Controles da UI (Widgets do Flet) ---
        self.file_picker_aluno = ft.FilePicker(on_result=self.on_file_aluno_selected)
        self.file_picker_mestre = ft.FilePicker(on_result=self.on_file_mestre_selected)

        # --- NOVO FILE PICKER PARA SALVAR O RELATÓRIO ---
        self.save_file_picker = ft.FilePicker(on_result=self.on_report_saved)
        # Adiciona todos os pickers à sobreposição da página (necessário no Flet).
        self.page.overlay.extend(
            [self.file_picker_aluno, self.file_picker_mestre, self.save_file_picker]
        )

        self.image_aluno = ft.Image(width=400, height=400)
        self.image_mestre = ft.Image(width=400, height=400)
        self.slider = ft.Slider(
            min=0,
            max=100,
            divisions=100,
            label="{value}%",
            on_change=self.on_slider_change,
            visible=False,
        )
        self.analyze_button = ft.Button(
            text="Analisar Movimentos", on_click=self.analyze, disabled=True
        )
        self.status_text = ft.Text("Por favor, carregue o vídeo do aluno e do mestre.")
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

        # --- NOVO BOTÃO PARA GERAR O RELATÓRIO ---
        self.report_button = ft.ElevatedButton(
            text="Gerar Relatório PDF",
            icon=ft.icons.PICTURE_AS_PDF,
            on_click=self.on_generate_report_click,
            visible=False,  # Só fica visível após a análise.
        )

        self.build_layout()
        logging.info("Aplicação Flet inicializada e layout construído.")

    def setup_page(self):
        """Configura as propriedades da página principal."""
        self.page.title = "Analisador de Movimentos de Krav Maga"
        self.page.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.page.theme_mode = ft.ThemeMode.DARK

    def build_layout(self):
        """Constrói o layout da interface com os controles Flet."""
        logging.info("Construindo o layout da UI.")
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
                    self.score_text,
                    self.feedback_text,
                    self.slider,
                    # --- Adiciona o novo botão de relatório ao final do layout ---
                    self.report_button,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,  # Ajustado para melhor espaçamento
            )
        )
        self.page.update()

    # on_file_aluno_selected, on_file_mestre_selected, check_files_loaded, analyze: NENHUMA MUDANÇA
    def on_file_aluno_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            self.video_path_aluno = e.files[0].path
            self.status_text.value = f"Aluno: {e.files[0].name}"
            logging.info(f"Vídeo do aluno selecionado: {self.video_path_aluno}")
            self.check_files_loaded()
        self.page.update()

    def on_file_mestre_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            self.video_path_mestre = e.files[0].path
            self.status_text.value += f" | Mestre: {e.files[0].name}"
            logging.info(f"Vídeo do mestre selecionado: {self.video_path_mestre}")
            self.check_files_loaded()
        self.page.update()

    def check_files_loaded(self):
        if self.video_path_aluno and self.video_path_mestre:
            self.analyze_button.disabled = False
            self.status_text.value = "Vídeos prontos! Clique em 'Analisar Movimentos'."
            logging.info(
                "Ambos os vídeos foram carregados. Botão de análise habilitado."
            )
        self.page.update()

    def analyze(self, e):
        logging.info("Botão de análise clicado.")
        self.status_text.value = "Analisando... por favor, aguarde."
        self.analyze_button.disabled = True
        self.page.update()
        self.video_analyzer = VideoAnalyzer(
            self.video_path_aluno, self.video_path_mestre
        )
        self.video_analyzer.start_analysis()
        self.page.run_thread(self.wait_for_analysis_completion)

    def wait_for_analysis_completion(self):
        """Espera a análise terminar e atualiza a UI."""
        self.video_analyzer.processing_thread.join()
        logging.info("Análise concluída. Configurando a UI pós-análise.")

        frame_count = self.video_analyzer.get_frame_count()
        if frame_count > 0:
            self.slider.max = frame_count - 1
            self.slider.divisions = frame_count - 1
            self.slider.visible = True
            self.score_text.visible = True
            self.feedback_text.visible = True
            # --- TORNA O BOTÃO DE RELATÓRIO VISÍVEL ---
            self.report_button.visible = True
            self.status_text.value = (
                "Análise completa! Use o slider ou gere um relatório."
            )
            self.update_frame_display(0)
        else:
            self.status_text.value = "Erro: Não foi possível processar os vídeos."

        self.page.update()

    # on_slider_change, update_frame_display, frame_to_base64: NENHUMA MUDANÇA
    def on_slider_change(self, e):
        frame_index = int(e.control.value)
        logging.debug(f"Slider movido para o frame {frame_index}.")
        self.update_frame_display(frame_index)

    def update_frame_display(self, frame_index):
        if self.video_analyzer:
            frame_aluno, frame_mestre, score, feedback = (
                self.video_analyzer.get_data_at_frame(frame_index)
            )
            if frame_aluno is not None and frame_mestre is not None:
                self.image_aluno.src_base64 = self.frame_to_base64(frame_aluno)
                self.image_mestre.src_base64 = self.frame_to_base64(frame_mestre)
                self.score_text.value = f"Pontuação: {score:.2f}%"
                self.feedback_text.value = f"Dica: {feedback}"
                if score >= 85:
                    self.score_text.color = ft.colors.GREEN
                elif score >= 60:
                    self.score_text.color = ft.colors.ORANGE
                else:
                    self.score_text.color = ft.colors.RED
                self.page.update()

    def frame_to_base64(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    # --- NOVOS MÉTODOS PARA GERAR O RELATÓRIO ---
    def on_generate_report_click(self, e):
        """
        Callback para o clique no botão de gerar relatório.
        Abre o diálogo para salvar o arquivo.
        """
        logging.info("Botão 'Gerar Relatório' clicado. Abrindo diálogo para salvar.")
        self.save_file_picker.save_file(
            dialog_title="Salvar Relatório de Análise",
            file_name="relatorio_krav_maga.pdf",
            allowed_extensions=["pdf"],
        )

    def on_report_saved(self, e: ft.FilePickerResultEvent):
        """
        Callback executado após o usuário escolher o local para salvar o PDF.
        """
        if e.path:
            save_path = e.path
            logging.info(f"Usuário escolheu salvar o relatório em: {save_path}")

            # Pega os dados necessários para o relatório.
            scores = self.video_analyzer.scores
            feedbacks = self.video_analyzer.feedbacks
            frame_aluno, frame_mestre = self.video_analyzer.get_best_frames()

            if frame_aluno is not None:
                # Gera o relatório.
                generator = ReportGenerator(
                    scores, feedbacks, frame_aluno, frame_mestre
                )
                success, error_message = generator.generate(save_path)

                # Mostra uma mensagem de confirmação (ou erro) para o usuário.
                if success:
                    snack_bar = ft.SnackBar(
                        ft.Text(f"Relatório salvo com sucesso em {save_path}"),
                        bgcolor=ft.colors.GREEN,
                    )
                else:
                    snack_bar = ft.SnackBar(
                        ft.Text(f"Erro ao salvar relatório: {error_message}"),
                        bgcolor=ft.colors.RED,
                    )

                self.page.snack_bar = snack_bar
                self.page.snack_bar.open = True
                self.page.update()
            else:
                logging.error(
                    "Não foi possível gerar o relatório pois os melhores frames não foram encontrados."
                )


def main(page: ft.Page):
    """Função principal que inicia a aplicação Flet."""
    KravMagaApp(page)


if __name__ == "__main__":
    ft.app(target=main)
