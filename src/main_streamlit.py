import streamlit as st
import logging
import os
import cv2
import numpy as np
import time
from src.video_analyzer import VideoAnalyzer
from src.utils import setup_logging

# Configuração do logging usando a função utilitária.
logger = setup_logging()


def main():
    """
    Função principal da aplicação Streamlit.
    Esta função configura a interface do usuário e orquestra o fluxo principal.
    """
    logger.info("Iniciando a aplicação Streamlit...")

    st.set_page_config(
        page_title="Analisador de Movimentos de Krav Maga (Streamlit)",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🥋 Analisador de Movimentos de Krav Maga (Streamlit - Protótipo)")
    logger.info("Título da aplicação Streamlit exibido.")

    st.markdown(
        """
        Bem-vindo ao Analisador de Movimentos de Krav Maga!
        Esta é a versão de protótipo/web da ferramenta, utilizando Visão Computacional (OpenCV e MediaPipe)
        para comparar seus movimentos com os de mestres, oferecendo feedback detalhado
        para aprimorar sua técnica.
    """
    )
    logger.info("Descrição da aplicação Streamlit exibida.")

    st.header("Upload de Vídeos")
    logger.info("Seção de upload de vídeos iniciada.")

    video_aluno = st.file_uploader(
        "Selecione o vídeo do Aluno (MP4, MOV, AVI)",
        type=["mp4", "mov", "avi"],
        key="aluno_video_uploader",
    )
    logger.info(
        f"Widget de upload de vídeo do aluno criado. Vídeo carregado: {video_aluno is not None}"
    )

    video_mestre = st.file_uploader(
        "Selecione o vídeo de Referência do Mestre (MP4, MOV, AVI)",
        type=["mp4", "mov", "avi"],
        key="mestre_video_uploader",
    )
    logger.info(
        f"Widget de upload de vídeo do mestre criado. Vídeo carregado: {video_mestre is not None}"
    )

    if video_aluno and video_mestre:
        st.success("Ambos os vídeos foram carregados com sucesso! 🎉")
        logger.info("Ambos os vídeos foram carregados. Pronto para análise.")

        if st.button("Analisar Movimentos"):
            logger.info("Botão 'Analisar Movimentos' clicado.")

            analyzer = VideoAnalyzer()

            st.info(
                "Iniciando a análise dos vídeos. Isso pode levar alguns minutos, por favor aguarde..."
            )
            progress_bar = st.progress(0)
            status_text = st.empty()

            col1, col2 = st.columns(2)

            annotated_frames_aluno = []
            landmarks_aluno = []
            annotated_frames_mestre = []
            landmarks_mestre = []

            # --- Processamento do Vídeo do Aluno ---
            status_text.text("Processando vídeo do Aluno...")
            logger.info("Iniciando processamento do vídeo do aluno.")
            try:
                # O método analyze_video retorna um gerador, processando frame a frame.
                for i, (frame, l_data) in enumerate(
                    analyzer.analyze_video(video_aluno)
                ):
                    annotated_frames_aluno.append(frame)
                    landmarks_aluno.append(l_data)
                    progress = min(int((i / 500) * 50), 49)
                    progress_bar.progress(progress)
                    if i % 50 == 0:
                        status_text.text(f"Processando vídeo do Aluno: Frame {i}...")

                progress_bar.progress(50)
                status_text.text(
                    "Vídeo do Aluno processado. Iniciando vídeo do Mestre..."
                )
                logger.info("Processamento do vídeo do aluno concluído.")

                # --- Processamento do Vídeo do Mestre ---
                logger.info("Iniciando processamento do vídeo do mestre.")
                for i, (frame, l_data) in enumerate(
                    analyzer.analyze_video(video_mestre)
                ):
                    annotated_frames_mestre.append(frame)
                    landmarks_mestre.append(l_data)
                    progress = min(50 + int((i / 500) * 50), 99)
                    progress_bar.progress(progress)
                    if i % 50 == 0:
                        status_text.text(f"Processando vídeo do Mestre: Frame {i}...")

                progress_bar.progress(100)
                status_text.text("Ambos os vídeos processados! Exibindo resultados...")
                logger.info(
                    "Processamento do vídeo do mestre concluído. Análise completa."
                )

                st.success("Análise de pose concluída! Exibindo vídeos processados. ✨")

                # Exibe os vídeos processados (com landmarks).
                with col1:
                    st.write("### Vídeo do Aluno (Com Detecção de Pose)")
                    if annotated_frames_aluno:
                        st.image(
                            cv2.cvtColor(annotated_frames_aluno[0], cv2.COLOR_BGR2RGB),
                            caption="Primeiro frame do vídeo do Aluno com pose detectada.",
                            use_column_width=True,
                        )
                        st.markdown(
                            f"**Total de Frames Processados (Aluno):** {len(annotated_frames_aluno)}"
                        )
                        logger.info(
                            f"Primeiro frame do vídeo do aluno exibido. Total frames: {len(annotated_frames_aluno)}"
                        )
                    else:
                        st.warning("Nenhum frame processado para o vídeo do Aluno.")
                        logger.warning("Nenhum frame processado para o vídeo do Aluno.")

                with col2:
                    st.write("### Vídeo do Mestre (Com Detecção de Pose)")
                    if annotated_frames_mestre:
                        st.image(
                            cv2.cvtColor(annotated_frames_mestre[0], cv2.COLOR_BGR2RGB),
                            caption="Primeiro frame do vídeo do Mestre com pose detectada.",
                            use_column_width=True,
                        )
                        st.markdown(
                            f"**Total de Frames Processados (Mestre):** {len(annotated_frames_mestre)}"
                        )
                        logger.info(
                            f"Primeiro frame do vídeo do mestre exibido. Total frames: {len(annotated_frames_mestre)}"
                        )
                    else:
                        st.warning("Nenhum frame processado para o vídeo do Mestre.")
                        logger.warning(
                            "Nenhum frame processado para o vídeo do Mestre."
                        )

                st.session_state["landmarks_aluno"] = landmarks_aluno
                st.session_state["landmarks_mestre"] = landmarks_mestre
                logger.info("Landmarks de aluno e mestre armazenados em session_state.")

            except Exception as e:
                logger.error(
                    f"Erro durante o processamento do vídeo: {e}", exc_info=True
                )
                st.error(f"Ocorreu um erro durante a análise do vídeo: {e}")
                status_text.text("Erro durante o processamento.")
            finally:
                if "analyzer" in locals() and analyzer is not None:
                    del analyzer

            st.subheader("Resultados da Análise e Feedback")
            st.write(
                "A comparação detalhada dos movimentos e o feedback serão exibidos aqui em breve."
            )
            logger.info("Seção de resultados da análise e feedback exibida.")

    else:
        st.warning("Por favor, carregue ambos os vídeos para iniciar a análise.")
        logger.info("Aguardando o upload de ambos os vídeos.")

    logger.info("Aplicação Streamlit encerrando o ciclo principal.")


if __name__ == "__main__":
    main()
