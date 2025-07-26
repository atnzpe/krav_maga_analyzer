# src/report_generator.py

import logging
import os
from datetime import datetime
import numpy as np
from fpdf import FPDF
import cv2

from src.utils import get_logger

logger = get_logger(__name__)

class PDF(FPDF):
    """
    Classe customizada que herda de FPDF para permitir cabeçalhos e rodapés padronizados.
    """
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Relatório de Análise de Movimento - Krav Maga', 0, 1, 'C')
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

class ReportGenerator:
    """
    Gera um relatório de análise em PDF com layout aprimorado e paginação correta.
    """
    
    def __init__(self, scores, feedbacks, frame_aluno_melhor, frame_mestre_melhor, frame_aluno_pior, frame_mestre_pior):
        self.scores = [s for s in scores if s is not None]
        self.feedbacks = feedbacks
        self.frame_aluno_melhor = frame_aluno_melhor
        self.frame_mestre_melhor = frame_mestre_melhor
        self.frame_aluno_pior = frame_aluno_pior
        self.frame_mestre_pior = frame_mestre_pior
        self.pdf = PDF()
        logger.info("ReportGenerator inicializado com dados de melhor e pior momentos.")

    def _add_section_title(self, title):
        """Adiciona um título de seção padronizado."""
        self.pdf.set_font('Arial', 'B', 14)
        self.pdf.cell(0, 10, title, 0, 1, 'L')
        self.pdf.line(self.pdf.get_x(), self.pdf.get_y(), self.pdf.get_x() + 190, self.pdf.get_y())
        self.pdf.ln(5)

    def _add_summary(self):
        """Adiciona a seção de resumo estatístico ao relatório."""
        self._add_section_title('Resumo de Performance')
        if self.scores:
            avg_score, max_score, min_score = np.mean(self.scores), np.max(self.scores), np.min(self.scores)
        else:
            avg_score, max_score, min_score = 0, 0, 0
        self.pdf.set_font('Arial', '', 11)
        self.pdf.cell(0, 8, f'Pontuação Média de Similaridade: {avg_score:.2f}%', 0, 1, 'L')
        self.pdf.cell(0, 8, f'Melhor Pontuação Obtida (Destaque): {max_score:.2f}%', 0, 1, 'L')
        self.pdf.cell(0, 8, f'Pior Pontuação Obtida (Ponto de Melhoria): {min_score:.2f}%', 0, 1, 'L')
        self.pdf.ln(10)

    def _add_moment_analysis(self, title, score, feedback, frame_aluno, frame_mestre, temp_suffix):
        """Função genérica para adicionar uma seção de análise de momento (melhor ou pior)."""
        self._add_section_title(title)
        self.pdf.set_font('Arial', '', 11)
        self.pdf.multi_cell(0, 8, f'Com uma pontuação de {score:.2f}%, o feedback foi: "{feedback}".', 0, 'L')
        self.pdf.ln(5)

        temp_aluno_path = f'temp_aluno_{temp_suffix}.png'
        temp_mestre_path = f'temp_mestre_{temp_suffix}.png'
        try:
            cv2.imwrite(temp_aluno_path, frame_aluno)
            cv2.imwrite(temp_mestre_path, frame_mestre)
            
            image_y_pos = self.pdf.get_y()
            # Ajuste das coordenadas X para aproximar as imagens
            self.pdf.image(temp_aluno_path, x=25, y=image_y_pos, w=75)
            self.pdf.image(temp_mestre_path, x=110, y=image_y_pos, w=75)
            
            # Pula o espaço vertical ocupado pelas imagens
            img_height = 75 * frame_aluno.shape[0] / frame_aluno.shape[1] # Calcula a altura proporcional
            self.pdf.ln(img_height + 5)
            
            self.pdf.set_font('Arial', 'I', 9)
            self.pdf.set_x(25)
            self.pdf.cell(75, 10, 'Sua Execução (Aluno)', 0, 0, 'C')
            self.pdf.set_x(110)
            self.pdf.cell(75, 10, 'Execução de Referência (Mestre)', 0, 1, 'C')
            
        finally:
            if os.path.exists(temp_aluno_path): os.remove(temp_aluno_path)
            if os.path.exists(temp_mestre_path): os.remove(temp_mestre_path)

    def generate(self, output_path):
        """
        Gera e salva o arquivo PDF completo, com paginação corrigida.
        """
        try:
            logger.info(f"Iniciando a geração do PDF para: {output_path}")
            self.pdf.add_page()
            self._add_summary()

            best_score_index = np.argmax(self.scores)
            self._add_moment_analysis(
                'Destaque da Sessão (Melhor Execução)',
                self.scores[best_score_index],
                self.feedbacks[best_score_index]['feedback'],
                self.frame_aluno_melhor,
                self.frame_mestre_melhor,
                'melhor'
            )
            
            # --- CORREÇÃO DE LAYOUT: Adiciona uma nova página ---
            # Isso força a próxima seção a começar em uma nova página, organizando o layout.
            self.pdf.add_page()

            worst_score_index = np.argmin(self.scores)
            self._add_moment_analysis(
                'Ponto de Melhoria (Pior Execução)',
                self.scores[worst_score_index],
                self.feedbacks[worst_score_index]['feedback'],
                self.frame_aluno_pior,
                self.frame_mestre_pior,
                'pior'
            )
            
            self.pdf.output(output_path)
            logger.info(f"Relatório PDF gerado com sucesso em: {output_path}")
            return True, None
        except Exception as e:
            logger.error(f"Falha ao gerar o relatório PDF: {e}", exc_info=True)
            return False, str(e)