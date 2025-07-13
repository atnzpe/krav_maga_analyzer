# src/video_analyzer.py

import cv2
import numpy as np

import tempfile
import os
from typing import Generator, Tuple, Optional
from src.pose_estimator import PoseEstimator  # Importa a classe PoseEstimator
from src.utils import setup_logging

# Configura o logger para este módulo específico.
logger = setup_logging()


class VideoAnalyzer:
    """
    Classe responsável por carregar vídeos, extrair frames e aplicar a detecção de pose.

    Esta classe gerencia o ciclo de vida de processamento de um vídeo,
    desde a leitura dos frames até a aplicação do modelo de pose
    e a coleta dos resultados.
    """

    def __init__(self):
        """
        Inicializa o VideoAnalyzer, preparando a instância do PoseEstimator.
        """
        logger.info("Inicializando VideoAnalyzer.")
        self.pose_estimator = PoseEstimator()  # Instancia o PoseEstimator.

    def analyze_video(
        self, video_file_buffer
    ) -> Generator[Tuple[np.ndarray, Optional[list]], None, None]:
        """
        Processa um vídeo, extrai frames, aplica detecção de pose e retorna os resultados.

        Args:
            video_file_buffer: O buffer do arquivo de vídeo (e.g., de st.file_uploader ou flet.FilePicker).

        Yields:
            Tuple[np.ndarray, Optional[list]]: Uma tupla contendo:
                - frame (np.ndarray): O frame do vídeo com os landmarks desenhados.
                - landmarks_data (Optional[list]): Uma lista de dicionários, onde cada dicionário
                                                    representa um landmark com suas coordenadas (x, y, z)
                                                    e visibilidade. Retorna None se nenhum landmark for detectado.
        """
        # Aplicação do Zen of Python: "Simple is better than complex."
        # Usamos tempfile para lidar com o buffer de arquivo de forma segura.
        # Isso garante que o vídeo possa ser lido pelo OpenCV.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            # Escreve o conteúdo do buffer em um arquivo temporário.
            # Se video_file_buffer for um objeto de arquivo (como de st.file_uploader), use .read()
            # Se for um path (como de flet.FilePickerResult), precisará de lógica diferente.
            # Por agora, assumimos um objeto com método .read().
            if hasattr(video_file_buffer, "read"):
                tmp_file.write(video_file_buffer.read())
            else:  # Assume que é um caminho de arquivo se não tiver .read()
                with open(video_file_buffer, "rb") as f:
                    tmp_file.write(f.read())

            temp_video_path = tmp_file.name
            logger.info(f"Vídeo temporário salvo em: {temp_video_path}")

        cap = cv2.VideoCapture(
            temp_video_path
        )  # Abre o arquivo de vídeo usando OpenCV.

        if not cap.isOpened():
            logger.error(f"Erro ao abrir o arquivo de vídeo: {temp_video_path}")
            # Limpa o arquivo temporário mesmo em caso de erro.
            os.remove(temp_video_path)
            # Adapta a mensagem de erro para o contexto, caso o buffer não tenha 'name'.
            file_name_display = getattr(
                video_file_buffer, "name", "arquivo desconhecido"
            )
            raise IOError(
                f"Não foi possível abrir o arquivo de vídeo: {file_name_display}"
            )

        frame_count = 0
        try:
            while cap.isOpened():  # Loop enquanto o vídeo estiver aberto.
                ret, frame = cap.read()  # Lê o próximo frame.
                if not ret:  # Se não há mais frames, sai do loop.
                    logger.info("Fim do vídeo ou erro de leitura de frame.")
                    break

                frame_count += 1
                logger.debug(f"Processando Frame #{frame_count}")

                # Processa o frame para detecção de pose.
                annotated_frame, results = self.pose_estimator.process_frame(frame)

                landmarks_data = None
                if results.pose_landmarks:
                    # Converte os landmarks do objeto MediaPipe para um formato mais fácil de usar (lista de dicts).
                    landmarks_data = [
                        {
                            "name": self.pose_estimator.mp_pose.PoseLandmark(i).name,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility,
                        }
                        for i, landmark in enumerate(results.pose_landmarks.landmark)
                    ]
                    logger.debug(f"Landmarks extraídos para o Frame #{frame_count}")
                else:
                    logger.debug(
                        f"Nenhum landmark detectado para o Frame #{frame_count}"
                    )

                # Retorna o frame anotado e os dados dos landmarks.
                yield annotated_frame, landmarks_data

        finally:
            cap.release()  # Libera o objeto VideoCapture.
            os.remove(temp_video_path)  # Remove o arquivo temporário.
            logger.info(
                f"Recursos de vídeo liberados e arquivo temporário {temp_video_path} removido."
            )
            self.pose_estimator.close()  # Garante que o modelo de pose seja fechado.

    def __del__(self):
        """
        Destrutor para garantir que o modelo de pose seja fechado se a instância
        for coletada pelo garbage collector.
        """
        logger.info("Destrutor do VideoAnalyzer chamado.")
        self.pose_estimator.close()


# Exemplo de uso (apenas para demonstração, não será executado na aplicação principal)
if __name__ == "__main__":
    logger.info("Executando exemplo de VideoAnalyzer com vídeo dummy.")
    # Para testar VideoAnalyzer, forneça um caminho para um arquivo de vídeo real.
    # Exemplo conceitual:
    # analyzer = VideoAnalyzer()
    # for annotated_frame, landmarks in analyzer.analyze_video(real_video_path_or_buffer):
    #     cv2.imshow("Frame", annotated_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    logger.info("Exemplo de VideoAnalyzer finalizado.")
