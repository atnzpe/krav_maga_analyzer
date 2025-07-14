import streamlit as st
import cv2
import numpy as np
import logging
import io
import tempfile
import os

# IMPORTANTE: Adicione estas linhas no topo para resolver ModuleNotFoundError
import sys

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import setup_logging
from src.video_analyzer import VideoAnalyzer

logger = setup_logging()

st.set_page_config(layout="wide", page_title="Analisador de Movimentos de Krav Maga")


def streamlit_main():
    """
    Função principal da aplicação Streamlit.
    """
    st.title("🥋 Analisador de Movimentos de Krav Maga")

    # Inicializa variáveis de estado da sessão se não existirem
    if "video_aluno_bytes" not in st.session_state:
        st.session_state["video_aluno_bytes"] = None
    if "video_mestre_bytes" not in st.session_state:
        st.session_state["video_mestre_bytes"] = None
    if "processed_frames_aluno" not in st.session_state:
        st.session_state["processed_frames_aluno"] = []
    if "processed_frames_mestre" not in st.session_state:
        st.session_state["processed_frames_mestre"] = []
    if "current_frame_aluno_index" not in st.session_state:
        st.session_state["current_frame_aluno_index"] = 0
    if "current_frame_mestre_index" not in st.session_state:
        st.session_state["current_frame_mestre_index"] = 0

    col1, col2 = st.columns(2)

    with col1:
        st.header("Vídeo do Aluno")
        uploaded_file_aluno = st.file_uploader(
            "Upload Vídeo do Aluno",
            type=["mp4", "mov", "avi"],
            key="aluno_video_uploader",
        )
        if uploaded_file_aluno is not None:
            st.session_state["video_aluno_bytes"] = uploaded_file_aluno.read()
            st.video(uploaded_file_aluno)  # Para pré-visualização do Streamlit
            logger.info(f"Vídeo do Aluno carregado: {uploaded_file_aluno.name}")

    with col2:
        st.header("Vídeo do Mestre")
        uploaded_file_mestre = st.file_uploader(
            "Upload Vídeo de Ref (Mestre)",
            type=["mp4", "mov", "avi"],
            key="mestre_video_uploader",
        )
        if uploaded_file_mestre is not None:
            st.session_state["video_mestre_bytes"] = uploaded_file_mestre.read()
            st.video(uploaded_file_mestre)  # Para pré-visualização do Streamlit
            logger.info(f"Vídeo do Mestre carregado: {uploaded_file_mestre.name}")

    if (
        st.session_state["video_aluno_bytes"] is None
        or st.session_state["video_mestre_bytes"] is None
    ):
        st.warning("Por favor, carregue ambos os vídeos para iniciar a análise.")
        analyze_button_disabled = True
    else:
        analyze_button_disabled = False

    if st.button(
        "Analisar Movimentos", disabled=analyze_button_disabled, key="analyze_button"
    ):
        st.info(
            "Iniciando a análise dos vídeos. Isso pode levar alguns minutos, por favor aguarde..."
        )
        logger.info("Botão 'Analisar Movimentos' clicado. Iniciando análise.")

        analyzer = VideoAnalyzer()

        try:
            # Converte bytes para BytesIO para o analyzer
            aluno_video_io = io.BytesIO(st.session_state["video_aluno_bytes"])
            mestre_video_io = io.BytesIO(st.session_state["video_mestre_bytes"])

            # Processamento do vídeo do Aluno
            st.session_state["processed_frames_aluno"] = []
            st.text("Processando vídeo do Aluno...")
            progress_aluno = st.progress(0)
            aluno_frames_gen = analyzer.analyze_video(aluno_video_io)

            # Precisamos estimar o número de frames para a barra de progresso.
            # Em uma aplicação real, isso seria feito lendo o cabeçalho do vídeo.
            # Aqui, para simplificar, vamos iterar uma vez para contar se necessário
            # ou apenas atualizar o progresso com base nos frames processados.
            # Para este exemplo, vamos apenas iterar e atualizar.

            frame_counter_aluno = 0
            for frame, l_data in aluno_frames_gen:
                st.session_state["processed_frames_aluno"].append(frame)
                frame_counter_aluno += 1
                progress_aluno.progress(
                    min(100, int((frame_counter_aluno / 500) * 100))
                )  # Exemplo: assume max 500 frames
            progress_aluno.empty()
            logger.info("Processamento do vídeo do Aluno concluído.")

            # Processamento do vídeo do Mestre
            st.session_state["processed_frames_mestre"] = []
            st.text("Processando vídeo do Mestre...")
            progress_mestre = st.progress(0)
            mestre_frames_gen = analyzer.analyze_video(mestre_video_io)

            frame_counter_mestre = 0
            for frame, l_data in mestre_frames_gen:
                st.session_state["processed_frames_mestre"].append(frame)
                frame_counter_mestre += 1
                progress_mestre.progress(
                    min(100, int((frame_counter_mestre / 500) * 100))
                )  # Exemplo: assume max 500 frames
            progress_mestre.empty()
            logger.info("Processamento do vídeo do Mestre concluído.")

            st.success("Ambos os vídeos processados! Exibindo resultados...")
            st.session_state["current_frame_aluno_index"] = 0
            st.session_state["current_frame_mestre_index"] = 0

        except Exception as e:
            logger.error(f"Erro durante o processamento do vídeo: {e}", exc_info=True)
            st.error(f"Ocorreu um erro durante a análise do vídeo: {e}")
        finally:
            if analyzer:
                del analyzer

    # Exibição dos frames processados e controles
    if (
        st.session_state["processed_frames_aluno"]
        and st.session_state["processed_frames_mestre"]
    ):
        st.subheader("Visualização dos Movimentos Analisados")

        # Controles de frame (Sliders)
        max_frames_aluno = len(st.session_state["processed_frames_aluno"]) - 1
        max_frames_mestre = len(st.session_state["processed_frames_mestre"]) - 1

        st.session_state["current_frame_aluno_index"] = st.slider(
            "Frame Vídeo Aluno",
            0,
            max_frames_aluno,
            st.session_state["current_frame_aluno_index"],
            key="aluno_frame_slider",
        )
        st.session_state["current_frame_mestre_index"] = st.slider(
            "Frame Vídeo Mestre",
            0,
            max_frames_mestre,
            st.session_state["current_frame_mestre_index"],
            key="mestre_frame_slider",
        )

        # Exibe os frames atuais
        current_frame_aluno = st.session_state["processed_frames_aluno"][
            st.session_state["current_frame_aluno_index"]
        ]
        current_frame_mestre = st.session_state["processed_frames_mestre"][
            st.session_state["current_frame_mestre_index"]
        ]

        # Converte frames OpenCV (numpy array) para bytes para exibição no Streamlit
        _, buffer_aluno = cv2.imencode(
            ".png", cv2.cvtColor(current_frame_aluno, cv2.COLOR_BGR2RGB)
        )
        _, buffer_mestre = cv2.imencode(
            ".png", cv2.cvtColor(current_frame_mestre, cv2.COLOR_BGR2RGB)
        )

        st.image(
            [buffer_aluno.tobytes(), buffer_mestre.tobytes()],
            caption=["Vídeo do Aluno (Processado)", "Vídeo do Mestre (Processado)"],
            width=450,
            use_column_width=False,
        )

        st.text("Resultados da Análise e Feedback: (Em breve)")
        st.info("Funcionalidades de comparação e feedback estão em desenvolvimento.")

    else:
        st.info("Carregue e analise os vídeos acima para ver os resultados aqui.")


if __name__ == "__main__":
    streamlit_main()
