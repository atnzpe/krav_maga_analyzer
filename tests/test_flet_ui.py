import pytest
from flet.testing import AppTest
import os
import sys
import asyncio
import flet as ft

# Adiciona o diretório raiz do projeto ao sys.path para que as importações funcionem corretamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importa a função main da sua aplicação Flet
from src.main_flet import main

@pytest.fixture(scope="module", autouse=True)
def run_flet_app_for_tests():
    """
    Fixture para rodar a aplicação Flet em um ambiente de teste simulado.
    Define um escopo de módulo para que a aplicação seja iniciada apenas uma vez.
    """
    # Usa AppTest para simular a execução da aplicação Flet.
    # O timeout padrão pode ser ajustado conforme a complexidade da inicialização.
    app_test = AppTest(target=main)
    # Roda a aplicação assincronamente para que a UI seja construída
    asyncio.run(app_test.run_async())
    yield app_test
    # A limpeza aqui seria mínima, pois AppTest simula um ciclo de vida.
    # Para testes mais complexos, pode ser necessário um `app_test.shutdown()`.


def test_app_loads_successfully(run_flet_app_for_tests):
    """
    Testa se a aplicação Flet carrega sem exceções e se o título está correto.
    Verifica a presença de elementos chave na UI.
    """
    at = run_flet_app_for_tests

    # Asserts para verificar se não há exceções durante o carregamento da aplicação
    assert not at.error, f"A aplicação Flet falhou ao carregar com erro: {at.error}"
    assert not at.exception, f"A aplicação Flet falhou ao carregar com exceções: {at.exception}"
    
    # Verifica o título da página
    assert at.page.title == "Analisador de Movimentos de Krav Maga (Flet)"
    
    # Verifica a presença da mensagem de status inicial
    assert "Por favor, carregue ambos os vídeos para iniciar a análise." in at.text[0].value

    # Verifica se os botões de upload existem e estão habilitados inicialmente
    assert at.get(lambda c: c.text == "Upload Vídeo do Aluno" and isinstance(c, ft.ElevatedButton)), "Botão 'Upload Vídeo do Aluno' não encontrado."
    assert at.get(lambda c: c.text == "Upload Vídeo do Mestre" and isinstance(c, ft.ElevatedButton)), "Botão 'Upload Vídeo do Mestre' não encontrado."
    
    # Verifica se o botão "Analisar Movimentos" está inicialmente desabilitado
    analyze_button = at.get(lambda c: c.text == "Analisar Movimentos" and isinstance(c, ft.ElevatedButton))
    assert analyze_button is not None, "Botão 'Analisar Movimentos' não encontrado."
    assert analyze_button.disabled is True, "Botão 'Analisar Movimentos' deveria estar desabilitado inicialmente."

    # Verifica se os IconButtons são criados corretamente sem o erro de 'text'
    # Procuramos pelo ícone esperado, pois o 'text' foi removido
    play_button_aluno = at.get(lambda c: c.tooltip == "Reproduzir/Pausar vídeo do aluno" and isinstance(c, ft.IconButton))
    assert play_button_aluno is not None, "IconButton do aluno não encontrado."
    assert play_button_aluno.icon == ft.Icons.PLAY_ARROW, "Ícone do botão de reprodução do aluno incorreto."

    play_button_mestre = at.get(lambda c: c.tooltip == "Reproduzir/Pausar vídeo do mestre" and isinstance(c, ft.IconButton))
    assert play_button_mestre is not None, "IconButton do mestre não encontrado."
    assert play_button_mestre.icon == ft.Icons.PLAY_ARROW, "Ícone do botão de reprodução do mestre incorreto."

    print("\nDebug: AppTest page controls after initial load:")
    for control in at.page.controls:
        print(f"  - {control.type}: {getattr(control, 'value', '') if isinstance(control, ft.Text) else getattr(control, 'text', '')}")