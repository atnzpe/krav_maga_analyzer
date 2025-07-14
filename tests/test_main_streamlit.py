import pytest # Importa o framework de testes pytest
from streamlit.testing.v1 import AppTest # Importa a classe AppTest para testar aplicações Streamlit
import os # Módulo para interagir com o sistema operacional
import sys # Módulo para interagir com o interpretador Python (usado para sys.path)
import numpy as np # Importa NumPy para manipulação de arrays (necessário para o mock de frames)
import io # Módulo para trabalhar com streams de I/O (necessário para simular upload de arquivos)

# IMPORTANTE: Adiciona o diretório raiz do projeto ao sys.path para que as importações
# de módulos como 'src.video_analyzer' funcionem corretamente durante os testes,
# independentemente de onde o pytest é executado.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Nota: AppTest.from_file já cuida da importação do script principal da aplicação.
# Se você tiver mocks complexos que precisam ser configurados antes de at.run(),
# pode ser necessário importar os módulos aqui para mocar (como VideoAnalyzer).

@pytest.fixture(autouse=True)
def clean_streamlit_cache():
    """
    Fixture (função executada antes de cada teste) para garantir um estado limpo
    para os testes do Streamlit. Embora o AppTest.from_file já forneça um bom
    isolamento, esta fixture serve como um lembrete e um ponto de extensão
    para futuras necessidades de limpeza de cache ou estado.
    """
    # No Streamlit testing v1, o isolamento entre testes é aprimorado,
    # então uma limpeza explícita de cache de `@st.cache_data` ou `@st.cache_resource`
    # não é estritamente necessária aqui, mas pode ser adicionada se houver problemas
    # de estado persistente entre os testes.
    pass


def test_streamlit_app_loads():
    """
    Testa se a aplicação Streamlit carrega e exibe o título principal sem erros.
    Este é um teste de fumaça (smoke test) para garantir que a aplicação
    pode ser inicializada e renderizada corretamente.
    """
    # INFO: Criando uma instância do AppTest a partir do arquivo principal da aplicação Streamlit.
    # default_timeout define o tempo máximo de espera para a aplicação carregar.
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    
    # INFO: Executa a aplicação Streamlit no ambiente de teste.
    at.run()

    # Verifica se a execução foi bem-sucedida, ou seja, se não houve exceções
    # que interromperam o carregamento da aplicação.
    # `at.exception` retorna uma lista de exceções que ocorreram. Esperamos que esteja vazia.
    assert len(at.exception) == 0, f"A aplicação Streamlit falhou ao carregar com exceções: {at.exception}"
    
    # Verifica se o título principal da aplicação está presente na interface renderizada.
    # `at.title` acessa elementos de título. `at.title[0].value` pega o texto do primeiro título.
    # Ajuste o texto se o título exato em 'src/main_streamlit.py' for diferente.
    assert "🥋 Analisador de Movimentos de Krav Maga" in at.title[0].value
    # LOG: Teste de carregamento da aplicação Streamlit concluído com sucesso.


def test_streamlit_initial_status_message():
    """
    Testa se a mensagem de status inicial correta é exibida
    quando a aplicação é carregada e nenhum vídeo foi feito upload ainda.
    """
    # INFO: Cria e executa a aplicação Streamlit.
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    at.run()
    
    # Verifica se a mensagem de aviso esperada é exibida.
    # `at.warning[0].body` acessa o corpo (texto) da primeira mensagem de aviso (`st.warning`).
    # Ajuste para `at.info[0].body` ou `at.text[0].value` se você usar `st.info` ou `st.text`.
    assert "Por favor, carregue ambos os vídeos para iniciar a análise." in at.warning[0].body
    # LOG: Teste de mensagem de status inicial concluído com sucesso.


def test_streamlit_analyze_button_initial_disabled_state():
    """
    Testa se o botão 'Analisar Movimentos' está desabilitado por padrão
    quando a aplicação é carregada, antes de qualquer upload de vídeo.
    """
    # INFO: Cria e executa a aplicação Streamlit.
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=5)
    at.run()
    
    # Acessa o botão pelo seu 'key' (definido em src/main_streamlit.py) e verifica
    # a propriedade 'disabled' do seu objeto 'proto' (o protobuffer subjacente do widget).
    assert at.button("analyze_button").proto.disabled is True
    # LOG: Teste de estado inicial desabilitado do botão 'Analisar Movimentos' concluído com sucesso.


def test_streamlit_analyze_button_enabled_after_simulated_upload():
    """
    Testa se o botão 'Analisar Movimentos' se torna habilitado
    após a simulação de upload de ambos os vídeos (Aluno e Mestre).
    """
    # INFO: Cria e executa a aplicação Streamlit.
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=10) # Aumenta o timeout para uploads
    at.run() # Roda o app inicialmente para que os widgets sejam renderizados

    # Simula o upload de arquivos para os `st.file_uploader`s.
    # Acessamos os file_uploaders diretamente pela lista `at.main.file_uploaders`
    # na ordem em que aparecem no script src/main_streamlit.py.
    # ATENÇÃO: Esta abordagem depende da ordem de criação dos file_uploaders no seu código Streamlit.
    # O primeiro file_uploader criado será `at.main.file_uploaders[0]`, o segundo `[1]`, e assim por diante.
    
    # Certifique-se de que "aluno_video_uploader" é o primeiro st.file_uploader no seu main_streamlit.py
    # e "mestre_video_uploader" é o segundo.
    
    # Simula o upload para o uploader do Aluno (geralmente o primeiro file_uploader na UI)
    aluno_uploader = at.main.file_uploaders[0] # Acessa o primeiro file_uploader
    aluno_uploader.set_value(
        io.BytesIO(b"dummy_video_data_aluno_mp4"), "aluno.mp4" 
    )

    # Simula o upload para o uploader do Mestre (geralmente o segundo file_uploader na UI)
    mestre_uploader = at.main.file_uploaders[1] # Acessa o segundo file_uploader
    mestre_uploader.set_value(
        io.BytesIO(b"dummy_video_data_mestre_mp4"), "mestre.mp4"
    )
    
    # INFO: Roda a aplicação novamente para que as mudanças de estado (uploads) sejam processadas
    # e a UI seja atualizada.
    at.run() 

    # Verifica se o botão "Analisar Movimentos" agora está habilitado.
    assert at.button("analyze_button").proto.disabled is False
    # LOG: Teste de habilitação do botão 'Analisar Movimentos' após upload concluído com sucesso.


def test_streamlit_analysis_flow_and_success_message(mocker):
    """
    Testa o fluxo completo de análise (simulada) de vídeos e a exibição
    das mensagens de progresso e sucesso.
    Utiliza `mocker` para simular o comportamento do `VideoAnalyzer`
    sem precisar processar vídeos reais, tornando o teste rápido.
    """
    # INFO: Cria e executa a aplicação Streamlit.
    at = AppTest.from_file("src/main_streamlit.py", default_timeout=20) # Timeout maior para simulação de processo

    # Mock do VideoAnalyzer: Substitui o método `analyze_video` do VideoAnalyzer
    # por uma função que retorna dados simulados. Isso evita que o teste
    # tente carregar e processar arquivos de vídeo reais, que seria lento.
    # `mock_frame` é um array NumPy que representa um frame de imagem simples.
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8) + 128 # Um frame cinza simulado
    # `mock_landmarks` simula os dados de landmarks que seriam retornados.
    mock_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}] 

    # Configura o mock: Quando `VideoAnalyzer.analyze_video` for chamado,
    # ele vai retornar um gerador que produz 5 pares de (mock_frame, mock_landmarks).
    mocker.patch('src.video_analyzer.VideoAnalyzer.analyze_video', 
                 return_value=[(mock_frame, mock_landmarks)] * 5)
    
    at.run() # Roda o app inicialmente

    # Simula o upload de arquivos para os `st.file_uploader`s.
    # Acessamos os file_uploaders diretamente pela lista `at.main.file_uploaders`
    # na ordem em que aparecem no script src/main_streamlit.py.
    # ATENÇÃO: Esta abordagem depende da ordem de criação dos file_uploaders no seu código Streamlit.
    
    # Simula o upload para o uploader do Aluno (geralmente o primeiro file_uploader na UI)
    aluno_uploader = at.main.file_uploaders[0] # Acessa o primeiro file_uploader
    aluno_uploader.set_value(
        io.BytesIO(b"dummy_video_data_aluno_mp4"), "aluno.mp4"
    )

    # Simula o upload para o uploader do Mestre (geralmente o segundo file_uploader na UI)
    mestre_uploader = at.main.file_uploaders[1] # Acessa o segundo file_uploader
    mestre_uploader.set_value(
        io.BytesIO(b"dummy_video_data_mestre_mp4"), "mestre.mp4"
    )
    
    at.run() # Atualiza UI após uploads para habilitar o botão de análise
    
    # Simula clique no botão "Analisar Movimentos".
    # O método `.click()` já re-executa a aplicação Streamlit para processar o clique.
    at.button("analyze_button").click()

    # Verifica se a mensagem de "em progresso" aparece.
    # `at.info[0].body` acessa o corpo da primeira mensagem `st.info`.
    assert "Iniciando a análise dos vídeos. Isso pode levar alguns minutos, por favor aguarde..." in at.info[0].body
    
    # Verifica a mensagem de sucesso final após a simulação de processamento.
    # `at.success[0].body` acessa o corpo da primeira mensagem `st.success`.
    assert "Ambos os vídeos processados! Exibindo resultados..." in at.success[0].body
    
    # Verifica se a mensagem final de conclusão. Pode ser um `st.text` ou outro `st.success`.
    # `at.text[-1].value` pega o texto do último elemento de texto.
    assert "Análise de pose concluída! ✨" in at.text[-1].value 

    # Verifica se as imagens processadas foram exibidas na UI.
    # Espera-se que pelo menos duas imagens (Aluno e Mestre) sejam renderizadas.
    assert len(at.image) >= 2 

    # Verifica se os sliders de frame foram criados e se o valor inicial é 0.
    assert at.slider("aluno_frame_slider").value == 0
    assert at.slider("mestre_frame_slider").value == 0
    # LOG: Teste de fluxo de análise e mensagens de sucesso concluído com sucesso.