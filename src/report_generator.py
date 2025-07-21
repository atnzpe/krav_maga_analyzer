# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------------
#  Krav Maga Motion Analyzer
#  version 1.2.0
#  Copyright (C) 2024,
# Gleyson Atanazio [Sujeito Programador]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------

# Importa a biblioteca fpdf para a criação de arquivos PDF.
# Foi escolhida por ser leve e poderosa para manipulação de PDFs em Python.
from fpdf import FPDF
# Importa a biblioteca logging para registrar o processo de geração do relatório.
import logging
# Importa a biblioteca os para manipulação de caminhos de arquivo.
import os
# Importa a biblioteca numpy para cálculos estatísticos como média.
import numpy as np
# Importa a biblioteca datetime para registrar a data e hora no relatório.
from datetime import datetime

# --------------------------------------------------------------------------------------------------
# Configuração do Logging
# --------------------------------------------------------------------------------------------------

# Configura o logging para registrar mensagens informativas sobre o processo.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------------------------------------------------------------------------------------
# Classe Geradora de Relatório
# --------------------------------------------------------------------------------------------------

class ReportGenerator:
    """
    Classe responsável por gerar um relatório de análise em formato PDF.
    Encapsula toda a lógica de criação e formatação do documento.
    (Zen of Python: "Complexidade encapsulada é melhor que complexidade espalhada").
    """
    
    def __init__(self, scores, feedbacks, frame_aluno, frame_mestre):
        """
        Inicializador da classe ReportGenerator.

        Args:
            scores (list): Lista com as pontuações de similaridade de cada frame.
            feedbacks (list): Lista com os feedbacks textuais de cada frame.
            frame_aluno (numpy.ndarray): O frame do aluno com a melhor pontuação.
            frame_mestre (numpy.ndarray): O frame do mestre correspondente à melhor pontuação do aluno.
        """
        self.scores = scores
        self.feedbacks = feedbacks
        self.frame_aluno = frame_aluno
        self.frame_mestre = frame_mestre
        self.pdf = FPDF() # Instancia o objeto PDF.
        logging.info("ReportGenerator inicializado com os dados da análise.")

    def _add_header(self):
        """
        Adiciona o cabeçalho ao documento PDF.
        Contém o título e a data da geração do relatório.
        """
        logging.info("Adicionando cabeçalho ao PDF.")
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Relatório de Análise de Movimento - Krav Maga', 0, 1, 'C')
        self.pdf.set_font('Arial', '', 10)
        # Registra a data e hora exatas da geração do relatório.
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.pdf.cell(0, 10, f'Gerado em: {now}', 0, 1, 'C')
        self.pdf.ln(10) # Adiciona um espaço vertical.

    def _add_summary(self):
        """
        Adiciona a seção de resumo estatístico ao relatório.
        Inclui pontuação média, máxima e mínima.
        """
        logging.info("Adicionando resumo estatístico ao PDF.")
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Resumo da Sessão', 0, 1, 'L')

        # Realiza os cálculos estatísticos usando a biblioteca NumPy.
        # Adicionamos uma verificação para evitar erros com listas vazias.
        if self.scores:
            avg_score = np.mean(self.scores)
            max_score = np.max(self.scores)
            min_score = np.min(self.scores)
        else:
            avg_score, max_score, min_score = 0, 0, 0

        self.pdf.set_font('Arial', '', 11)
        self.pdf.cell(0, 8, f'Pontuação Média de Similaridade: {avg_score:.2f}%', 0, 1, 'L')
        self.pdf.cell(0, 8, f'Melhor Pontuação Obtida: {max_score:.2f}%', 0, 1, 'L')
        self.pdf.cell(0, 8, f'Pior Pontuação Obtida: {min_score:.2f}%', 0, 1, 'L')
        self.pdf.ln(10)

    def _add_key_moment_analysis(self):
        """
        Adiciona a análise do momento chave (melhor pontuação) ao relatório.
        Inclui as imagens lado a lado do aluno e do mestre.
        """
        logging.info("Adicionando análise do momento chave ao PDF.")
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Análise do Momento de Melhor Execução', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 11)
        
        # Encontra o índice da melhor pontuação para obter o feedback correspondente.
        if self.scores:
            best_score_index = np.argmax(self.scores)
            best_feedback = self.feedbacks[best_score_index]
            best_score = self.scores[best_score_index]
        else:
            best_feedback, best_score = "N/A", 0

        self.pdf.multi_cell(0, 8, f'No momento de maior similaridade ({best_score:.2f}%), o feedback foi: "{best_feedback}". Isso indica uma sincronia excelente com o movimento do mestre.', 0, 'L')
        self.pdf.ln(5)

        # Adiciona as imagens ao PDF. As imagens precisam ser salvas temporariamente.
        # Usamos try/finally para garantir que os arquivos temporários sejam sempre deletados.
        temp_aluno_path = 'temp_aluno.png'
        temp_mestre_path = 'temp_mestre.png'
        try:
            import cv2
            cv2.imwrite(temp_aluno_path, self.frame_aluno)
            cv2.imwrite(temp_mestre_path, self.frame_mestre)
            
            logging.info(f"Salvando imagens temporárias em {os.path.abspath(temp_aluno_path)} e {os.path.abspath(temp_mestre_path)}")
            
            # Adiciona as imagens lado a lado.
            self.pdf.image(temp_aluno_path, x=20, w=80)
            self.pdf.image(temp_mestre_path, x=110, w=80)
            self.pdf.ln(85) # Espaço para passar as imagens.
            
            self.pdf.set_x(20)
            self.pdf.cell(80, 10, 'Sua Execução (Aluno)', 0, 0, 'C')
            self.pdf.cell(90, 10, 'Execução de Referência (Mestre)', 0, 1, 'C')
            
        except Exception as e:
            logging.error(f"Erro ao adicionar imagens ao PDF: {e}")
            self.pdf.cell(0, 10, "Erro ao processar as imagens do momento chave.", 0, 1, 'L')
        finally:
            # Garante a limpeza dos arquivos temporários.
            if os.path.exists(temp_aluno_path):
                os.remove(temp_aluno_path)
                logging.info(f"Removido arquivo temporário: {temp_aluno_path}")
            if os.path.exists(temp_mestre_path):
                os.remove(temp_mestre_path)
                logging.info(f"Removido arquivo temporário: {temp_mestre_path}")

    def generate(self, output_path):
        """
        Gera e salva o arquivo PDF completo.

        Args:
            output_path (str): O caminho completo onde o arquivo PDF será salvo.
        """
        try:
            logging.info(f"Iniciando a geração do PDF para ser salvo em: {output_path}")
            self.pdf.add_page()
            self._add_header()
            self._add_summary()
            self._add_key_moment_analysis()
            
            self.pdf.output(output_path)
            logging.info(f"Relatório PDF gerado com sucesso em: {output_path}")
            return True, None
        except Exception as e:
            # Captura qualquer exceção durante a geração para evitar que a aplicação quebre.
            logging.error(f"Falha ao gerar o relatório PDF: {e}")
            return False, str(e)