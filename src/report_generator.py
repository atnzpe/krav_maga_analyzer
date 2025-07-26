# src/report_generator.py

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------
import logging
import os
from datetime import datetime
import numpy as np
from fpdf import FPDF
import cv2

# --- Importação dos Módulos do Projeto ---
from src.utils import get_logger

# --------------------------------------------------------------------------------------------------
# Configuração do Logging
# --------------------------------------------------------------------------------------------------
logger = get_logger(__name__)


# --------------------------------------------------------------------------------------------------
# Classe Geradora de Relatório
# --------------------------------------------------------------------------------------------------
class ReportGenerator:
    """
    Classe responsável por gerar um relatório de análise em formato PDF.
    """

    def __init__(self, scores, feedbacks, frame_aluno, frame_mestre):
        """
        Inicializador da classe ReportGenerator.

        Args:
            scores (list): Lista com as pontuações de similaridade de cada frame.
            feedbacks (list): Lista com os feedbacks textuais de cada frame.
            frame_aluno (numpy.ndarray): O frame do aluno com a melhor pontuação.
            frame_mestre (numpy.ndarray): O frame do mestre correspondente.
        """
        self.scores = scores
        self.feedbacks = feedbacks
        self.frame_aluno = frame_aluno
        self.frame_mestre = frame_mestre
        self.pdf = FPDF()
        logger.info("ReportGenerator inicializado com os dados da análise.")

    def _add_header(self):
        """Adiciona o cabeçalho ao documento PDF."""
        logger.info("Adicionando cabeçalho ao PDF.")
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, "Relatório de Análise de Movimento - Krav Maga", 0, 1, "C")
        self.pdf.set_font("Arial", "", 10)
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.pdf.cell(0, 10, f"Gerado em: {now}", 0, 1, "C")
        self.pdf.ln(10)

    def _add_summary(self):
        """Adiciona a seção de resumo estatístico ao relatório."""
        logger.info("Adicionando resumo estatístico ao PDF.")
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Resumo da Sessão", 0, 1, "L")

        if self.scores:
            avg_score = np.mean(self.scores)
            max_score = np.max(self.scores)
            min_score = np.min(self.scores)
        else:
            avg_score, max_score, min_score = 0, 0, 0

        self.pdf.set_font("Arial", "", 11)
        self.pdf.cell(
            0, 8, f"Pontuação Média de Similaridade: {avg_score:.2f}%", 0, 1, "L"
        )
        self.pdf.cell(0, 8, f"Melhor Pontuação Obtida: {max_score:.2f}%", 0, 1, "L")
        self.pdf.cell(0, 8, f"Pior Pontuação Obtida: {min_score:.2f}%", 0, 1, "L")
        self.pdf.ln(10)

    def _add_key_moment_analysis(self):
        """Adiciona a análise do momento chave (melhor pontuação) ao relatório."""
        logger.info("Adicionando análise do momento chave ao PDF.")
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Análise do Momento de Melhor Execução", 0, 1, "L")
        self.pdf.set_font("Arial", "", 11)

        if self.scores:
            best_score_index = np.argmax(self.scores)
            best_feedback = self.feedbacks[best_score_index]["feedback"]
            best_score = self.scores[best_score_index]
        else:
            best_feedback, best_score = "N/A", 0

        self.pdf.multi_cell(
            0,
            8,
            f'No momento de maior similaridade ({best_score:.2f}%), o feedback foi: "{best_feedback}".',
            0,
            "L",
        )
        self.pdf.ln(5)

        temp_aluno_path = "temp_aluno.png"
        temp_mestre_path = "temp_mestre.png"
        try:
            cv2.imwrite(temp_aluno_path, self.frame_aluno)
            cv2.imwrite(temp_mestre_path, self.frame_mestre)

            self.pdf.image(temp_aluno_path, x=20, w=80)
            self.pdf.image(temp_mestre_path, x=110, w=80)
            self.pdf.ln(85)

            self.pdf.set_x(20)
            self.pdf.cell(80, 10, "Sua Execução (Aluno)", 0, 0, "C")
            self.pdf.cell(90, 10, "Execução de Referência (Mestre)", 0, 1, "C")

        except Exception as e:
            logger.error(f"Erro ao adicionar imagens ao PDF: {e}")
            self.pdf.cell(
                0, 10, "Erro ao processar as imagens do momento chave.", 0, 1, "L"
            )
        finally:
            if os.path.exists(temp_aluno_path):
                os.remove(temp_aluno_path)
            if os.path.exists(temp_mestre_path):
                os.remove(temp_mestre_path)

    def generate(self, output_path):
        """
        Gera e salva o arquivo PDF completo.
        """
        try:
            logger.info(f"Iniciando a geração do PDF para: {output_path}")
            self.pdf.add_page()
            self._add_header()
            self._add_summary()
            self._add_key_moment_analysis()

            self.pdf.output(output_path)
            logger.info(f"Relatório PDF gerado com sucesso em: {output_path}")
            return True, None
        except Exception as e:
            logger.error(f"Falha ao gerar o relatório PDF: {e}")
            return False, str(e)
