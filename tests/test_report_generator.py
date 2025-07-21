# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------------
#  Krav Maga Motion Analyzer - Testes
#  version 1.2.0
#  Copyright (C) 2024,
#  Gleyson Atanazio [Sujeito Programador]
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# Importação de Bibliotecas
# --------------------------------------------------------------------------------------------------

import pytest  # Framework de testes.
import os  # Para interagir com o sistema de arquivos.
import numpy as np  # Para criar arrays de imagem falsos.
from unittest.mock import patch, MagicMock  # Para "mockar" (simular) objetos e métodos.

# Importa a classe que será testada.
from src.report_generator import ReportGenerator

# --------------------------------------------------------------------------------------------------
# Fixtures e Dados de Teste
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def mock_analysis_data():
    """
    Cria um conjunto de dados de análise falsos para serem usados nos testes.
    Isso nos permite testar o ReportGenerator de forma isolada.
    """
    scores = [85.5, 92.1, 78.3, 95.8]
    feedbacks = ["Bom", "Ótimo", "Ajuste o quadril", "Excelente movimento!"]
    # Cria uma imagem falsa (um array 10x10 preenchido com zeros) para simular um frame de vídeo.
    mock_image = np.zeros((10, 10, 3), dtype=np.uint8)
    return {
        "scores": scores,
        "feedbacks": feedbacks,
        "frame_aluno": mock_image,
        "frame_mestre": mock_image,
    }


# --------------------------------------------------------------------------------------------------
# Casos de Teste para ReportGenerator
# --------------------------------------------------------------------------------------------------


def test_report_generator_initialization(mock_analysis_data):
    """
    Testa se a classe ReportGenerator é inicializada corretamente com os dados.

    Cenário: Criar uma instância da classe.
    Resultado Esperado: Os atributos da instância devem conter os dados fornecidos.
    """
    generator = ReportGenerator(**mock_analysis_data)

    assert generator.scores == mock_analysis_data["scores"]
    assert generator.feedbacks == mock_analysis_data["feedbacks"]
    assert np.array_equal(generator.frame_aluno, mock_analysis_data["frame_aluno"])


# O decorator @patch substitui temporariamente um objeto ou método por um Mock.
# Aqui, estamos substituindo toda a biblioteca 'fpdf' e o método 'cv2.imwrite'
# para que não sejam realmente chamados durante o teste.
@patch("src.report_generator.FPDF")
@patch("src.report_generator.cv2.imwrite")
@patch("src.report_generator.os.remove")
def test_generate_report_success(
    mock_os_remove, mock_imwrite, mock_fpdf_class, mock_analysis_data
):
    """
    Testa o fluxo de sucesso da geração de um relatório PDF.

    Cenário: Chamar o método generate() com dados válidos.
    Resultado Esperado: O método deve retornar sucesso, e os métodos da biblioteca FPDF
                       e de manipulação de arquivos devem ser chamados corretamente.
    """
    # Configuração dos Mocks
    mock_pdf_instance = MagicMock()
    mock_fpdf_class.return_value = mock_pdf_instance

    # Cria a instância do gerador com dados falsos.
    generator = ReportGenerator(**mock_analysis_data)

    output_path = "test_report.pdf"

    # Chama o método que queremos testar.
    success, error = generator.generate(output_path)

    # Asserções: verificamos se o comportamento foi o esperado.
    assert success is True
    assert error is None

    # Verifica se o método `add_page` do PDF foi chamado uma vez.
    mock_pdf_instance.add_page.assert_called_once()
    # Verifica se o método `cell` (usado para escrever texto) foi chamado.
    assert mock_pdf_instance.cell.call_count > 0
    # Verifica se o método `output` foi chamado com o caminho de arquivo correto.
    mock_pdf_instance.output.assert_called_once_with(output_path)

    # Verifica se `imwrite` foi chamado para salvar as imagens temporárias.
    assert mock_imwrite.call_count == 2
    # Verifica se `os.remove` foi chamado para limpar os arquivos temporários.
    assert mock_os_remove.call_count == 2


def test_generate_report_failure_on_exception(mock_analysis_data):
    """
    Testa o tratamento de erro quando uma exceção ocorre durante a geração do PDF.

    Cenário: O método `output` da biblioteca FPDF lança uma exceção.
    Resultado Esperado: O método generate() deve capturar a exceção, retornar falha
                       e a mensagem de erro.
    """
    # Usamos 'with patch' para aplicar o mock apenas neste teste.
    with patch("src.report_generator.FPDF") as mock_fpdf_class:
        mock_pdf_instance = MagicMock()
        # Configuramos o mock para levantar uma exceção quando `output` for chamado.
        mock_pdf_instance.output.side_effect = IOError("Disk full")
        mock_fpdf_class.return_value = mock_pdf_instance

        generator = ReportGenerator(**mock_analysis_data)

        # Chama o método e verifica o resultado.
        success, error_message = generator.generate("dummy_path.pdf")

        assert success is False
        assert "Disk full" in error_message
