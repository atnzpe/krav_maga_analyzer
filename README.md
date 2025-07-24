# Krav Maga Movement Analyzer

## Visão Geral do Projeto

O Krav Maga Movement Analyzer é uma aplicação inovadora projetada para auxiliar praticantes e instrutores de Krav Maga na análise e aprimoramento de seus movimentos. Utilizando técnicas avançadas de Visão Computacional e Machine Learning (especificamente detecção de pose), a ferramenta permite comparar vídeos de alunos com vídeos de referência (mestre), fornecendo feedback visual e potencialmente análises quantitativas sobre a execução dos movimentos.

A interface do usuário é construída com [Flet](https://flet.dev/), uma estrutura que permite a criação de aplicativos web, desktop e móveis com Python.

# Analisador de Movimentos de Krav Maga

**Versão:** 1.9.0

Uma aplicação de desktop para análise e aprimoramento de técnicas de Krav Maga utilizando Visão Computacional, MediaPipe e Flet.

## Funcionalidades

- **Upload de Vídeos:** Carregue facilmente o vídeo do aluno e do mestre.
- **Visualização Lado a Lado:** Assista aos dois vídeos sincronizados para uma comparação visual direta.
- **Detecção de Pose:** Utiliza o Google MediaPipe para rastrear os principais pontos do corpo.
- **✨ Controles de Reprodução Completos:**
    - **Play/Pause:** Assista à execução do movimento em velocidade normal.
    - **Navegação Frame a Frame:** Use os botões de "Próximo" e "Anterior" para uma análise minuciosa.
    - **Slider Interativo:** Arraste o slider para pular para qualquer ponto do vídeo instantaneamente.
- **Feedback em Tempo Real:** Receba uma pontuação de similaridade e dicas de correção para cada frame.
- **Geração de Relatório em PDF:** Exporte um resumo profissional da sua análise.

## Tecnologias Utilizadas

- **Python 3.11**
- **Flet**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **scikit-learn**
- **fpdf2**

## Como Usar

1.  Execute a aplicação.
2.  Carregue os vídeos do aluno e do mestre.
3.  Clique em "Analisar Movimentos".
4.  Aguarde o processamento.
5.  **Use os botões de play, pause, avançar e voltar, ou o slider, para analisar os movimentos em detalhe.**
6.  Observe a pontuação e as dicas que mudam a cada frame.

## Como Instalar e Rodar

Siga os passos abaixo para configurar e executar a aplicação no seu ambiente.

### Pré-requisitos

* **Python 3.11:** É crucial ter o Python 3.11 instalado. Versões muito recentes (como 3.13) podem não ter suporte completo de bibliotecas como MediaPipe.
    * **No Windows:** Baixe o instalador em [python.org](https://www.python.org/downloads/windows/). Durante a instalação, **marque as opções "Add python.exe to PATH" e "Install launcher for all users (recommended)"**. É recomendado desinstalar outras versões problemáticas do Python e reiniciar o computador antes de instalar o Python 3.11 para evitar conflitos.
    * **No macOS (com Homebrew):** `brew install python@3.11`
    * **No Linux (com apt):** `sudo apt update && sudo apt install python3.11 python3.11-venv`

### Passos de Instalação

1.  **Clone o Repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd krav_maga_analyzer
    ```

2.  **Crie e Ative um Ambiente Virtual:**
    É altamente recomendado usar um ambiente virtual para gerenciar as dependências do projeto.

    * **No Windows:**
        ```powershell
        py -3.11 -m venv venv
        .\venv\Scripts\activate
        ```
    * **No macOS/Linux:**
        ```bash
        python3.11 -m venv venv
        source venv/bin/activate
        ```

3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    O arquivo `requirements.txt` deve conter as seguintes dependências:
    ```
    flet==0.23.0 # ou a versão mais recente compatível
    mediapipe==0.10.14 # ou a versão mais recente compatível com Python 3.11
    opencv-python==4.9.0.80 # ou a versão mais recente compatível
    numpy==1.26.4 # ou a versão mais recente compatível
    Pillow==10.3.0 # ou a versão mais recente compatível
    ```

## Como Usar

1.  **Execute a Aplicação Flet:**
    Certifique-se de que seu ambiente virtual está ativado.
    ```bash
    python src/main_flet.py
    ```
    Uma janela da aplicação Flet será aberta.

2.  **Carregue os Vídeos:**
    * Clique em "Upload Vídeo do Aluno" para selecionar o vídeo do aluno.
    * Clique em "Upload Vídeo do Mestre" para selecionar o vídeo de referência.

3.  **Analise os Movimentos:**
    * Após carregar ambos os vídeos, clique no botão "Analisar Movimentos".
    * O aplicativo processará os vídeos (isso pode levar um tempo, dependendo do tamanho e duração dos vídeos).

4.  **Explore os Resultados:**
    * Os vídeos processados aparecerão lado a lado com os esqueletos de pose.
    * Use os sliders para navegar frame a frame.
    * Use os botões "Play" e "Pause" para reproduzir os vídeos.

## Estrutura do Projeto


krav_maga_analyzer/

├── .gitignore                # Arquivos e pastas a serem ignorados pelo Git

├── README.md                 # Este arquivo de documentação

├── requirements.txt          # Lista de dependências do Python

├── videos/                   # Diretório para armazenar vídeos de exemplo ou teste
│   ├── master                #   Subdiretório para vídeos do mestre
│   └── user                  #   Subdiretório para vídeos do usuário/aluno

├── src/                      # Código fonte da aplicação
│   ├── main_flet.py          # Ponto de entrada da aplicação Flet (prioridade atual)
│   ├── main_streamlit.py     # Ponto de entrada da aplicação Streamlit (protótipo/web)
│   ├── video_analyzer.py     # Módulo para processamento e análise de vídeo
│   ├── pose_estimator.py     # Módulo para detecção de pose com MediaPipe
│   ├── movement_comparer.py  # Módulo para lógica de comparação de movimentos (futuro)
│   ├── feedback_generator.py # Módulo para gerar feedback ao usuário (futuro)
│   └── utils.py  #Funções utilitárias diversas (e.g., setup_logging)

├── tests/                    # Testes automatizados para os módulos
│   ├── test_video_analyzer.py
│   ├── test_pose_estimator.py
│   ├── test_main_streamlit.py # Testes para a aplicação Streamlit
│   └── test_main_flet.py      # Testes para a aplicação Flet (futuro)

└── logs/                     # Diretório para arquivos de log da aplicação
└── app.log               # Arquivo de log principal da aplicação


## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues, propor melhorias ou enviar pull requests.



## Licença

Este projeto está licenciado sob a [Apache](LICENSE.md).

