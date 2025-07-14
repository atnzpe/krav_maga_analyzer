import flet as ft

import logging
import os
import cv2
import numpy as np
import tempfile
import threading
import time
import asyncio  # Importa o módulo asyncio para permitir operações assíncronas, como `asyncio.sleep` para controlar o FPS da reprodução de vídeo.

# IMPORTANTE: Adicione estas linhas no topo para resolver ModuleNotFoundError
import sys

# Adiciona o diretório raiz do projeto ao sys.path. Isso é crucial para que as importações
# de módulos locais, como `src.utils` e `src.video_analyzer`, funcionem corretamente
# quando o script é executado de diferentes diretórios.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import setup_logging  # Importa a função `setup_logging` do módulo `utils` para configurar o sistema de log da aplicação.
from src.video_analyzer import VideoAnalyzer  # Importa a classe `VideoAnalyzer` do módulo `video_analyzer`, responsável por processar vídeos e detectar poses.

logger = setup_logging()  # Inicializa o logger da aplicação, configurado para registrar informações e erros.

# Variáveis globais (ou de sessão) para armazenar os caminhos dos arquivos e dados processados.
# Estas variáveis são acessíveis e modificáveis por qualquer função dentro do módulo.
VIDEO_ALUNO_PATH = None  # Armazena o caminho completo para o arquivo de vídeo do aluno. Inicialmente `None`.
VIDEO_MESTRE_PATH = None  # Armazena o caminho completo para o arquivo de vídeo do mestre. Inicialmente `None`.
PROCESSED_FRAMES_ALUNO = []  # Uma lista para armazenar os frames processados do vídeo do aluno (com landmarks de pose desenhados).
PROCESSED_FRAMES_MESTRE = []  # Uma lista para armazenar os frames processados do vídeo do mestre.
CURRENT_FRAME_ALUNO_INDEX = 0  # O índice do frame atual sendo exibido ou processado para o vídeo do aluno.
CURRENT_FRAME_MESTRE_INDEX = 0  # O índice do frame atual para o vídeo do mestre.
IS_PLAYING_ALUNO = False  # Um booleano que indica se o vídeo do aluno está em reprodução (`True`) ou pausado (`False`).
IS_PLAYING_MESTRE = False  # Um booleano que indica se o vídeo do mestre está em reprodução.
PLAYBACK_THREAD_ALUNO = None  # Referência à tarefa (Task) Flet que gerencia a reprodução do vídeo do aluno.
PLAYBACK_THREAD_MESTRE = None  # Referência à tarefa (Task) Flet que gerencia a reprodução do vídeo do mestre.
FPS = 30  # Frames por segundo. Define a taxa de atualização da reprodução dos vídeos na UI. Idealmente, este valor deveria ser extraído do vídeo original.


# =============================================================================
# NOVAS VARIÁVEIS GLOBAIS
# =============================================================================
video_analyzer_instance = None  # Uma instância da classe `VideoAnalyzer`, usada para realizar a detecção de pose e comparação.
feedback_text_container = None  # Referência ao `ft.Container` na UI onde o feedback textual é exibido.
aluno_video_player = (
    None  # Referência ao `ft.Image` control que exibirá os frames do vídeo do aluno.
)
mestre_video_player = (
    None  # Referência ao `ft.Image` control que exibirá os frames do vídeo do mestre.
)


def main(page: ft.Page):
    """
    Função principal da aplicação Flet.
    Configura a interface do usuário (UI), gerencia uploads de vídeo,
    análise de pose e a reprodução dos vídeos.

    Args:
        page (ft.Page): O objeto da página Flet principal da aplicação.
                        Representa a janela ou aba do navegador onde a aplicação Flet será renderizada.
    """
    logger.info("Iniciando a aplicação Flet...")

    page.title = "Analisador de Movimentos de Krav Maga (Flet)"  # Define o título da página ou da janela da aplicação.
    page.vertical_alignment = ft.MainAxisAlignment.START  # Alinha o conteúdo verticalmente no topo da página.

    # =========================================================================
    # NOVA INSTÂNCIA DO VIDEOANALYZER E FUNÇÕES AUXILIARES
    # =========================================================================
    global video_analyzer_instance, feedback_text_container, aluno_video_player, mestre_video_player
    # Inicializa o VideoAnalyzer para processamento de vídeo e detecção de pose.
    # Esta instância será reutilizada para todas as análises.
    video_analyzer_instance = VideoAnalyzer()

    def clear_analysis_data():
        """
        Limpa todos os dados de análise de vídeo e estados da aplicação.
        Isso inclui caminhos de arquivo, frames processados, índices de reprodução
        e threads ativas. Atualiza a UI para refletir o estado limpo, como
        redefinir o texto de feedback e as visualizações de vídeo.
        """
        global VIDEO_ALUNO_PATH, VIDEO_MESTRE_PATH, PROCESSED_FRAMES_ALUNO, PROCESSED_FRAMES_MESTRE, CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX, IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, PLAYBACK_THREAD_ALUNO, PLAYBACK_THREAD_MESTRE, video_analyzer_instance

        # Redefine os caminhos dos vídeos para `None`.
        VIDEO_ALUNO_PATH = None
        VIDEO_MESTRE_PATH = None
        # Limpa as listas de frames processados.
        PROCESSED_FRAMES_ALUNO = []
        PROCESSED_FRAMES_MESTRE = []
        # Redefine os índices dos frames atuais para o início.
        CURRENT_FRAME_ALUNO_INDEX = 0
        CURRENT_FRAME_MESTRE_INDEX = 0
        # Define os estados de reprodução como `False` (parado).
        IS_PLAYING_ALUNO = False
        IS_PLAYING_MESTRE = False

        # Interrompe as threads de reprodução se estiverem ativas.
        # Em Flet, `page.run_task` gerencia a Task internamente, então definir a referência como `None`
        # e usar a flag `IS_PLAYING` no loop de reprodução é a forma de sinalizar para a task terminar.
        if PLAYBACK_THREAD_ALUNO:
            PLAYBACK_THREAD_ALUNO = None
        if PLAYBACK_THREAD_MESTRE:
            PLAYBACK_THREAD_MESTRE = None

        # Resetar a instância do VideoAnalyzer para limpar os históricos de landmarks.
        # Isso é importante para garantir que uma nova análise comece sem dados residuais de análises anteriores.
        video_analyzer_instance = VideoAnalyzer()

        # Limpar a interface do usuário (UI).
        if feedback_text_container:
            # Redefine o conteúdo do container de feedback para o texto inicial.
            feedback_text_container.content.value = "Feedback textual aqui..."
        if aluno_video_player:  # Verifica se o controle de imagem do aluno existe.
            aluno_video_player.src = None  # Remove a fonte da imagem, efetivamente limpando o player.
        if mestre_video_player:  # Verifica se o controle de imagem do mestre existe.
            mestre_video_player.src = None  # Remove a fonte da imagem.

        # Atualiza a página para refletir as mudanças na UI.
        page.update()
        logger.info("Dados de análise e UI limpos.")

    def update_feedback_text(text: str):
        """
        Atualiza o texto de feedback exibido na interface do usuário.
        Adiciona o novo texto ao conteúdo existente, criando um histórico de mensagens.

        Args:
            text (str): O novo texto a ser anexado ao feedback existente.
        """
        if feedback_text_container:
            # Anexa o novo texto com uma nova linha, mantendo os anteriores para um histórico de feedback.
            feedback_text_container.content.value += f"\n{text}"
            page.update()  # Atualiza a UI para exibir o novo texto.
            logger.info(f"Feedback atualizado: {text}")

    # =========================================================================
    # FUNÇÕES DE UPLOAD E ANÁLISE (MODIFICADAS)
    # =========================================================================
    def upload_aluno_video(e: ft.FilePickerResultEvent):
        """
        Manipula o evento de seleção de arquivo para o vídeo do aluno.
        Define o caminho do vídeo do aluno e atualiza a mensagem de status na UI.

        Args:
            e (ft.FilePickerResultEvent): O objeto de evento retornado pelo `FilePicker` após a seleção do arquivo.
                                           Contém informações sobre os arquivos selecionados.
        """
        global VIDEO_ALUNO_PATH
        if e.files:  # Verifica se algum arquivo foi selecionado.
            VIDEO_ALUNO_PATH = e.files[0].path  # Armazena o caminho do primeiro arquivo selecionado.
            logger.info(f"Vídeo do Aluno selecionado: {VIDEO_ALUNO_PATH}")
            update_status_message()  # Chama a função para atualizar a mensagem de status da UI.
        else:
            VIDEO_ALUNO_PATH = None  # Reseta o caminho se a seleção for cancelada.
            logger.warning("Seleção de vídeo do aluno cancelada.")

    def upload_mestre_video(e: ft.FilePickerResultEvent):
        """
        Manipula o evento de seleção de arquivo para o vídeo do mestre.
        Define o caminho do vídeo do mestre e atualiza a mensagem de status na UI.

        Args:
            e (ft.FilePickerResultEvent): O objeto de evento retornado pelo `FilePicker` após a seleção do arquivo.
        """
        global VIDEO_MESTRE_PATH
        if e.files:  # Verifica se algum arquivo foi selecionado.
            VIDEO_MESTRE_PATH = e.files[0].path  # Armazena o caminho do primeiro arquivo selecionado.
            logger.info(f"Vídeo do Mestre selecionado: {VIDEO_MESTRE_PATH}")
            update_status_message()  # Chama a função para atualizar a mensagem de status da UI.
        else:
            VIDEO_MESTRE_PATH = None  # Reseta o caminho se a seleção for cancelada.
            logger.warning("Seleção de vídeo do mestre cancelada.")

    def update_status_message():
        """
        Atualiza a mensagem de status na UI e o estado dos botões 'Analisar' e 'Limpar'.
        O botão 'Analisar' só é habilitado quando ambos os vídeos são carregados.
        """
        # Usa a referência direta ao controle `ft.Text` para atualizar seu valor.
        if VIDEO_ALUNO_PATH and VIDEO_MESTRE_PATH:  # Verifica se ambos os caminhos de vídeo estão definidos.
            status_message_control.value = (
                "Ambos os vídeos carregados! Clique em 'Analisar' para iniciar."
            )
            analyze_button.disabled = False  # Habilita o botão de análise.
            clear_button.disabled = False  # Habilita o botão de limpar.
        else:
            status_message_control.value = (
                "Por favor, carregue ambos os vídeos para iniciar a análise."
            )
            analyze_button.disabled = True  # Desabilita o botão de análise.
            clear_button.disabled = True  # Desabilita o botão de limpar.
        page.update()  # Atualiza a UI para exibir a nova mensagem e estado dos botões.

    def numpy_frame_to_image_src(frame: np.ndarray) -> str:
        """
        Converte um frame de vídeo NumPy (formato BGR, como retornado pelo OpenCV)
        para uma string Base64 formatada para ser usada como 'src' em um controle `ft.Image` do Flet.
        Isso permite exibir frames de vídeo diretamente na UI do Flet.

        Args:
            frame (np.ndarray): O frame de imagem no formato NumPy array (BGR).

        Returns:
            str: Uma string formatada no padrão `data:image/png;base64,...` para uso
                 no atributo `src` de um `ft.Image`.
        """
        # Converte o frame de BGR (OpenCV padrão) para RGB, que é o formato esperado para a maioria das exibições de imagem.
        # Em seguida, codifica a imagem para o formato PNG em um buffer de memória.
        _, buffer = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Codifica o buffer (bytes da imagem PNG) para Base64 e o decodifica para uma string UTF-8.
        # Retorna a string formatada com o prefixo `data:image/png;base64,` para ser reconhecida como URL de dados pelo Flet.
        return f"data:image/png;base64,{np.base64encode(buffer).decode('utf-8')}"

    # Funções de reprodução
    async def play_video(player_type: str):
        """
        Gerencia a reprodução de vídeo assíncrona para o player especificado ('aluno' ou 'mestre').
        Atualiza os frames do vídeo na UI continuamente, controlando a velocidade com base no FPS.

        Args:
            player_type (str): O tipo de player para controlar a reprodução ('aluno' ou 'mestre').
        """
        global IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, CURRENT_FRAME_ALUNO_INDEX, CURRENT_FRAME_MESTRE_INDEX, aluno_video_player, mestre_video_player

        # Obtém as variáveis globais relevantes dinamicamente com base no `player_type`.
        # `is_playing_var` é a flag booleana que controla se a reprodução deve continuar.
        is_playing_var = globals()[f"IS_PLAYING_{player_type.upper()}"]
        # `current_frame_index_var` é o índice do frame atual.
        current_frame_index_var = globals()[
            f"CURRENT_FRAME_{player_type.upper()}_INDEX"
        ]
        # `processed_frames_var` é a lista de frames processados.
        processed_frames_var = globals()[f"PROCESSED_FRAMES_{player_type.upper()}"]
        # `video_player_ref` é a referência ao controle `ft.Image` do player correspondente.
        video_player_ref = (
            aluno_video_player if player_type == "aluno" else mestre_video_player
        )

        # Se a flag `is_playing_var` for `False`, significa que a reprodução foi pausada ou interrompida.
        if not is_playing_var:
            logger.debug(f"Reprodução de {player_type} pausada.")
            return  # Sai da função.

        logger.info(f"Iniciando reprodução de {player_type}...")
        # Loop de reprodução contínua: continua enquanto o vídeo estiver marcado para reprodução
        # e o índice do frame atual for menor que o número total de frames.
        while globals()[
            f"IS_PLAYING_{player_type.upper()}"
        ] and current_frame_index_var < len(processed_frames_var):
            # Atualiza a variável global `CURRENT_FRAME_INDEX` para o tipo de player correto.
            globals()[
                f"CURRENT_FRAME_{player_type.upper()}_INDEX"
            ] = current_frame_index_var  # Atualiza a global

            # Obtém o frame a ser exibido da lista de frames processados.
            frame_to_display = processed_frames_var[current_frame_index_var]
            # Converte o frame NumPy para Base64 e define como a fonte do controle `ft.Image`.
            video_player_ref.src = numpy_frame_to_image_src(frame_to_display)
            await page.update_async()  # Atualiza a página de forma assíncrona para exibir o novo frame, evitando bloquear a UI.

            # Avança para o próximo frame.
            current_frame_index_var += 1
            # Se o vídeo chegou ao fim, reseta o índice para o início para reprodução contínua.
            if current_frame_index_var >= len(processed_frames_var):
                current_frame_index_var = 0  # Loop contínuo

            # Atualiza a variável global `CURRENT_FRAME_INDEX` novamente após o ajuste (se houver loop).
            globals()[
                f"CURRENT_FRAME_{player_type.upper()}_INDEX"
            ] = current_frame_index_var

            # Controle de FPS para reprodução suave: pausa por uma fração de segundo.
            # O tempo de espera é 1 dividido pelo FPS (e.g., 1/30 = 0.033 segundos).
            await asyncio.sleep(1 / FPS)
        logger.info(f"Reprodução de {player_type} finalizada ou pausada.")

    def toggle_play(e: ft.ControlEvent, player_type: str):
        """
        Alterna o estado de reprodução (play/pause) para o vídeo do aluno ou mestre.
        Inicia ou para a tarefa de reprodução correspondente e atualiza o ícone do botão.

        Args:
            e (ft.ControlEvent): O objeto de evento de clique do controle (botão de play/pause).
            player_type (str): O tipo de player ('aluno' ou 'mestre') para alternar o estado de reprodução.
        """
        global IS_PLAYING_ALUNO, IS_PLAYING_MESTRE, PLAYBACK_THREAD_ALUNO, PLAYBACK_THREAD_MESTRE

        if player_type == "aluno":
            IS_PLAYING_ALUNO = not IS_PLAYING_ALUNO  # Inverte o estado de reprodução do aluno.
            # Se estiver iniciando a reprodução e não houver uma tarefa de reprodução ativa para o aluno:
            if IS_PLAYING_ALUNO and not (
                PLAYBACK_THREAD_ALUNO and PLAYBACK_THREAD_ALUNO.is_alive()
            ):
                # Inicia a reprodução do vídeo do aluno em uma nova tarefa assíncrona.
                # `page.run_task` é usado para executar funções assíncronas em segundo plano no Flet.
                PLAYBACK_THREAD_ALUNO = page.run_task(play_video(player_type))
                logger.info("Iniciando thread de reprodução do Aluno.")
            elif not IS_PLAYING_ALUNO:  # Se estiver pausando a reprodução do aluno:
                logger.info("Pausando reprodução do Aluno.")
        elif player_type == "mestre":
            IS_PLAYING_MESTRE = not IS_PLAYING_MESTRE  # Inverte o estado de reprodução do mestre.
            # Se estiver iniciando a reprodução e não houver uma tarefa de reprodução ativa para o mestre:
            if IS_PLAYING_MESTRE and not (
                PLAYBACK_THREAD_MESTRE and PLAYBACK_THREAD_MESTRE.is_alive()
            ):
                # Inicia a reprodução do vídeo do mestre em uma nova tarefa assíncrona.
                PLAYBACK_THREAD_MESTRE = page.run_task(play_video(player_type))
                logger.info("Iniciando thread de reprodução do Mestre.")
            elif not IS_PLAYING_MESTRE:  # Se estiver pausando a reprodução do mestre:
                logger.info("Pausando reprodução do Mestre.")

        # Atualiza o ícone do botão para refletir o estado de reprodução (Play ou Pause).
        e.control.icon = (
            ft.Icons.PAUSE
            if globals()[f"IS_PLAYING_{player_type.upper()}"]  # Se estiver tocando, mostra o ícone de pausa.
            else ft.Icons.PLAY_ARROW  # Se estiver pausado, mostra o ícone de play.
        )
        page.update()  # Atualiza a UI para mostrar o novo ícone do botão.

    def seek_aluno_video(e: ft.ControlEvent):
        """
        Atualiza o frame exibido do vídeo do aluno com base na posição do slider.
        Isso permite ao usuário navegar manualmente pelos frames do vídeo.

        Args:
            e (ft.ControlEvent): O objeto de evento de mudança do slider.
                                   `e.control.value` contém o valor atual do slider.
        """
        global CURRENT_FRAME_ALUNO_INDEX
        CURRENT_FRAME_ALUNO_INDEX = int(e.control.value)  # Converte o valor do slider para um inteiro e define como o índice do frame atual.
        if PROCESSED_FRAMES_ALUNO:  # Verifica se existem frames processados para o aluno.
            aluno_video_player.src = numpy_frame_to_image_src(
                PROCESSED_FRAMES_ALUNO[CURRENT_FRAME_ALUNO_INDEX]  # Define a fonte da imagem para o frame correspondente ao índice do slider.
            )
            page.update()  # Atualiza a UI para exibir o novo frame.
            logger.info(f"Slider do Aluno movido para o frame: {CURRENT_FRAME_ALUNO_INDEX}")

    def seek_mestre_video(e: ft.ControlEvent):
        """
        Atualiza o frame exibido do vídeo do mestre com base na posição do slider.
        Args:
            e (ft.ControlEvent): O objeto de evento de mudança do slider.
        """
        global CURRENT_FRAME_MESTRE_INDEX
        CURRENT_FRAME_MESTRE_INDEX = int(e.control.value)  # Define o índice do frame do mestre com base no slider.
        if PROCESSED_FRAMES_MESTRE:  # Verifica se existem frames processados para o mestre.
            mestre_video_player.src = numpy_frame_to_image_src(
                PROCESSED_FRAMES_MESTRE[CURRENT_FRAME_MESTRE_INDEX]  # Define a fonte da imagem para o frame correspondente.
            )
            page.update()  # Atualiza a UI.
            logger.info(
                f"Slider do Mestre movido para o frame: {CURRENT_FRAME_MESTRE_INDEX}"
            )

    def analyze_videos(e):
        """
        Inicia o processo de análise dos vídeos do aluno e do mestre.
        Isso inclui a detecção de pose em cada frame e a comparação de movimentos entre os dois vídeos.
        Atualiza a UI com feedback do progresso e os resultados finais.

        Args:
            e: O objeto de evento de clique do botão 'Analisar'.
        """
        global PROCESSED_FRAMES_ALUNO, PROCESSED_FRAMES_MESTRE, VIDEO_ALUNO_PATH, VIDEO_MESTRE_PATH

        # Desabilitar botões durante a análise para evitar interrupções e novas ações do usuário enquanto o processamento ocorre.
        analyze_button.disabled = True
        upload_aluno_button.disabled = True
        upload_mestre_button.disabled = True
        clear_button.disabled = True  # Desabilita o botão de limpar durante a análise.

        update_feedback_text(
            "Iniciando a análise dos vídeos. Isso pode levar alguns minutos, por favor aguarde..."
        )
        page.update()  # Atualiza a UI para mostrar a mensagem de status e o estado dos botões.

        try:
            # Limpar dados de análise anteriores antes de iniciar uma nova, garantindo um estado limpo.
            clear_analysis_data()
            update_feedback_text("Limpando dados anteriores...")

            # --- Processar Vídeo do Aluno ---
            update_feedback_text("Processando vídeo do Aluno...")
            temp_aluno_path = None  # Variável para armazenar o caminho de um arquivo temporário, se necessário.
            # Verifica se `VIDEO_ALUNO_PATH` é uma string (caminho de arquivo) ou `io.BytesIO` (dados em memória).
            if isinstance(VIDEO_ALUNO_PATH, str):
                video_source_aluno = VIDEO_ALUNO_PATH  # Se for string, usa o caminho diretamente.
            else:
                # Assume que é `io.BytesIO` se não for string.
                # Se for `BytesIO`, é necessário salvá-lo temporariamente em um arquivo para que o OpenCV possa lê-lo.
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"  # Cria um arquivo temporário com extensão .mp4.
                ) as temp_file:
                    temp_file.write(VIDEO_ALUNO_PATH.read())  # Escreve o conteúdo do BytesIO no arquivo temporário.
                    temp_aluno_path = temp_file.name  # Armazena o caminho do arquivo temporário.
                video_source_aluno = temp_aluno_path  # Define a fonte do vídeo como o arquivo temporário.

            # Resetar os históricos de landmarks na instância do `VideoAnalyzer`
            # para garantir que não haja acúmulo de processamentos anteriores.
            video_analyzer_instance.aluno_landmarks_history = []
            PROCESSED_FRAMES_ALUNO = []
            # Itera sobre os frames gerados pela análise do vídeo do aluno.
            for (
                annotated_frame,  # O frame processado com os landmarks desenhados.
                landmarks_data,  # Os dados dos landmarks (coordenadas e visibilidade).
            ) in video_analyzer_instance.analyze_video(video_source_aluno):
                PROCESSED_FRAMES_ALUNO.append(
                    annotated_frame
                )  # Adiciona o frame anotado à lista.
                video_analyzer_instance.aluno_landmarks_history.append(
                    landmarks_data
                )  # Armazena os dados dos landmarks.
            update_feedback_text(
                f"Vídeo do Aluno processado: {len(PROCESSED_FRAMES_ALUNO)} frames."
            )
            logger.info(
                f"Total de frames processados do aluno: {len(PROCESSED_FRAMES_ALUNO)}"
            )

            # Limpa o arquivo temporário do aluno, se foi criado.
            if temp_aluno_path and os.path.exists(temp_aluno_path):
                os.remove(temp_aluno_path)
                logger.info(f"Arquivo temporário do aluno removido: {temp_aluno_path}")

            # --- Processar Vídeo do Mestre ---
            update_feedback_text("Processando vídeo do Mestre...")
            temp_mestre_path = None  # Variável para armazenar o caminho de um arquivo temporário.
            # Verifica se `VIDEO_MESTRE_PATH` é uma string (caminho de arquivo) ou `io.BytesIO`.
            if isinstance(VIDEO_MESTRE_PATH, str):
                video_source_mestre = VIDEO_MESTRE_PATH  # Se for string, usa o caminho diretamente.
            else:
                # Se for `BytesIO`, salva temporariamente.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(VIDEO_MESTRE_PATH.read())
                    temp_mestre_path = temp_file.name
                video_source_mestre = temp_mestre_path

            # Resetar os históricos de landmarks na instância do `VideoAnalyzer`.
            video_analyzer_instance.mestre_landmarks_history = []
            PROCESSED_FRAMES_MESTRE = []
            # Itera sobre os frames gerados pela análise do vídeo do mestre.
            for annotated_frame, landmarks_data in video_analyzer_instance.analyze_video(
                video_source_mestre
            ):
                PROCESSED_FRAMES_MESTRE.append(annotated_frame)
                video_analyzer_instance.mestre_landmarks_history.append(landmarks_data)
            update_feedback_text(
                f"Vídeo do Mestre processado: {len(PROCESSED_FRAMES_MESTRE)} frames."
            )
            logger.info(
                f"Total de frames processados do mestre: {len(PROCESSED_FRAMES_MESTRE)}"
            )

            # Limpa o arquivo temporário do mestre, se foi criado.
            if temp_mestre_path and os.path.exists(temp_mestre_path):
                os.remove(temp_mestre_path)
                logger.info(
                    f"Arquivo temporário do mestre removido: {temp_mestre_path}"
                )

            # --- Análise e Comparação (Exemplo simplificado) ---
            update_feedback_text("Realizando análise de comparação de movimentos...")
            # Aqui você implementaria a lógica de comparação de movimentos.
            # Por exemplo, comparar `aluno_landmarks_history` com `mestre_landmarks_history`.
            # A complexidade da comparação dependerá dos requisitos específicos (e.g., DTW, similaridade de pose).

            # Exemplo de feedback baseado na duração dos vídeos
            if PROCESSED_FRAMES_ALUNO and PROCESSED_FRAMES_MESTRE:
                min_frames = min(
                    len(PROCESSED_FRAMES_ALUNO), len(PROCESSED_FRAMES_MESTRE)
                )
                update_feedback_text(
                    f"Ambos os vídeos processados! Exibindo resultados... Duração comparável de {min_frames / FPS:.2f} segundos."
                )
                # Atualizar sliders com o número correto de frames
                slider_aluno.max = len(PROCESSED_FRAMES_ALUNO) - 1
                slider_aluno.divisions = len(PROCESSED_FRAMES_ALUNO) - 1
                slider_aluno.value = 0

                slider_mestre.max = len(PROCESSED_FRAMES_MESTRE) - 1
                slider_mestre.divisions = len(PROCESSED_FRAMES_MESTRE) - 1
                slider_mestre.value = 0

                # Exibir o primeiro frame de cada vídeo
                aluno_video_player.src = numpy_frame_to_image_src(
                    PROCESSED_FRAMES_ALUNO[0]
                )
                mestre_video_player.src = numpy_frame_to_image_src(
                    PROCESSED_FRAMES_MESTRE[0]
                )
            else:
                update_feedback_text(
                    "Não foi possível processar um ou ambos os vídeos. Verifique os arquivos."
                )

        except Exception as ex:
            # Captura e registra qualquer exceção que ocorra durante o processo de análise.
            logger.error(f"Erro durante a análise de vídeo: {ex}", exc_info=True)
            update_feedback_text(f"Ocorreu um erro durante a análise: {ex}")
        finally:
            # Reabilitar botões após a análise, independentemente do sucesso ou falha.
            analyze_button.disabled = False
            upload_aluno_button.disabled = False
            upload_mestre_button.disabled = False
            clear_button.disabled = False  # Reabilita o botão de limpar.
            page.update()  # Garante que a UI seja atualizada com o estado final dos botões.

    # =========================================================================
    # CONFIGURAÇÃO DA UI
    # =========================================================================

    # Permite ao usuário selecionar arquivos de vídeo.
    file_picker_aluno = ft.FilePicker(on_result=upload_aluno_video)
    file_picker_mestre = ft.FilePicker(on_result=upload_mestre_video)

    # Adiciona o file picker à página (necessário para que ele funcione).
    page.overlay.append(file_picker_aluno)
    page.overlay.append(file_picker_mestre)

    # Botões de upload de vídeo.
    upload_aluno_button = ft.ElevatedButton(
        "Upload Vídeo do Aluno",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda e: file_picker_aluno.pick_files(
            allow_multiple=False, allowed_extensions=["mp4", "avi", "mov"]
        ),
    )
    upload_mestre_button = ft.ElevatedButton(
        "Upload Vídeo do Mestre",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda e: file_picker_mestre.pick_files(
            allow_multiple=False, allowed_extensions=["mp4", "avi", "mov"]
        ),
    )

    # Botão para iniciar a análise (inicialmente desabilitado).
    analyze_button = ft.ElevatedButton(
        "Analisar Movimentos",
        icon=ft.Icons.ANALYTICS,
        on_click=analyze_videos,
        disabled=True,  # Começa desabilitado até que os vídeos sejam carregados.
        key="analyze_button",  # Adiciona uma chave para identificação em testes.
    )

    # Botão para limpar a análise.
    clear_button = ft.ElevatedButton(
        "Limpar Análise",
        icon=ft.Icons.CLEAR,
        on_click=lambda e: clear_analysis_data(),
        disabled=True,  # Começa desabilitado.
    )

    # Mensagem de status para o usuário.
    status_message_control = ft.Text(
        "Por favor, carregue ambos os vídeos para iniciar a análise.",
        size=16,
        color=ft.Colors.AMBER_500,
    )

    # Controles de vídeo e sliders.
    aluno_video_player = ft.Image(
        src=None,
        fit=ft.ImageFit.CONTAIN,  # Ajusta a imagem dentro dos limites do controle.
        expand=True, # Definindo expand para que ocupe o espaço disponível
        height=300, # Mantendo a altura fixa para consistência
    )
    mestre_video_player = ft.Image(
        src=None,
        fit=ft.ImageFit.CONTAIN,
        expand=True, # Definindo expand para que ocupe o espaço disponível
        height=300, # Mantendo a altura fixa para consistência
    )

    play_button_aluno = ft.IconButton(
        icon=ft.Icons.PLAY_ARROW,
        on_click=lambda e: toggle_play(e, "aluno"),
        tooltip="Reproduzir/Pausar vídeo do aluno",
    )
    play_button_mestre = ft.IconButton(
        icon=ft.Icons.PLAY_ARROW,
        on_click=lambda e: toggle_play(e, "mestre"),
        tooltip="Reproduzir/Pausar vídeo do mestre",
    )

    slider_aluno = ft.Slider(
        min=0,
        max=0,  # Será atualizado dinamicamente após o carregamento do vídeo
        divisions=0,  # Será atualizado dinamicamente
        value=0,  # Será atualizado dinamicamente
        on_change=seek_aluno_video,
    )
    slider_mestre = ft.Slider(
        min=0,
        max=0,  # Será atualizado dinamicamente
        divisions=0,  # Será atualizado dinamicamente
        value=0,  # Será atualizado dinamicamente
        on_change=seek_mestre_video,
    )

    feedback_text_container = ft.Container(
        content=ft.Text("Feedback textual aqui...", selectable=True),
        width=None, # Removendo a largura fixa
        height=200,
        bgcolor=ft.Colors.BLUE_GREY_50,
        padding=10,
        border_radius=10,
        alignment=ft.alignment.top_left,
        expand=True, # Permitindo que ocupe o espaço disponível
    )


    # Adiciona os controles à página.
    page.add(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Image(src="assets/logo_FBKKLN.png", width=100, height=100),
                        ft.Text(
                            "Analisador de Movimentos de Krav Maga",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                status_message_control,  # Mensagem de status.
                ft.Row(
                    [
                        upload_aluno_button,  # Botão de upload do aluno.
                        upload_mestre_button,  # Botão de upload do mestre.
                        analyze_button,  # Botão de análise.
                        clear_button,  # Botão de limpar.
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,  # Centraliza os botões na linha.
                    spacing=10,  # Espaçamento entre os botões.
                ),
                ft.Column(
                    [
                        ft.Row(  # Linha para os vídeos
                            [
                                ft.Container(
                                    content=aluno_video_player,
                                    expand=True,  # Ocupa o espaço disponível horizontalmente
                                    height=300,
                                    bgcolor=ft.Colors.BLACK,
                                    alignment=ft.alignment.center,
                                ),
                                ft.Container(
                                    content=mestre_video_player,
                                    expand=True,  # Ocupa o espaço disponível horizontalmente
                                    height=300,
                                    bgcolor=ft.Colors.BLACK,
                                    alignment=ft.alignment.center,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_AROUND,  # Distribui o espaço igualmente ao redor dos itens na linha.
                            expand=True,  # A linha se expande para preencher a largura disponível
                        ),
                        ft.Row(  # Linha para os botões de play e sliders.
                            [
                                play_button_aluno,
                                ft.Container(content=slider_aluno, expand=True),  # Slider do aluno ocupa o espaço restante
                                play_button_mestre,
                                ft.Container(content=slider_mestre, expand=True),  # Slider do mestre ocupa o espaço restante
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            expand=True,  # A linha se expande para preencher a largura disponível
                        ),
                        ft.Text(
                            "Resultados da Análise e Feedback:",
                            size=18,
                            weight=ft.FontWeight.NORMAL,
                        ),
                        feedback_text_container,  # O container onde o feedback será exibido.
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Centraliza o conteúdo da coluna principal horizontalmente.
                    spacing=10,  # Espaçamento entre os elementos da coluna.
                    expand=True,  # Permite que a coluna se expanda para preencher o espaço disponível.
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Centraliza o conteúdo da página horizontalmente.
            spacing=20,  # Espaçamento entre os elementos da página.
            expand=True,  # A coluna principal se expande para preencher a altura disponível.
        )
    )
    page.update()  # Atualiza a página para renderizar todos os elementos da UI.
    logger.info("Elementos da UI Flet adicionados e página atualizada.")


# Ponto de entrada da aplicação Flet.
if __name__ == "__main__":
    # Inicia a aplicação Flet, apontando para a função 'main'.
    # O Flet cuida do ciclo de vida da aplicação, criando a janela e executando a função `main`.
    ft.app(target=main)