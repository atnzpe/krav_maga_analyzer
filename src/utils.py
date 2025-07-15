# src/utils.py

import logging
import os
import sys  # Importado para a configuração do StreamHandler
import numpy as np  # Importa a biblioteca NumPy para operações matemáticas com arrays


def setup_logging():
    """
    Configura o sistema de logging para a aplicação.
    Cria um diretório 'logs' se não existir e configura o logger
    para gravar em 'app.log' e exibir no console.

    Esta função garante que todas as mensagens importantes do sistema
    sejam registradas para depuração e monitoramento, seguindo as melhores práticas.
    """
    log_dir = "logs"  # Define o nome do diretório para os logs
    if not os.path.exists(log_dir):  # Verifica se o diretório de logs existe
        os.makedirs(log_dir)  # Se não existir, cria o diretório

    log_file_path = os.path.join(
        log_dir, "app.log"
    )  # Define o caminho completo do arquivo de log

    # Remove handlers existentes para evitar duplicação em re-configurações.
    # Isso é importante em ambientes como Streamlit ou Flet onde a função pode ser chamada
    # múltiplas vezes durante o ciclo de vida da aplicação.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configura a base do sistema de logging
    logging.basicConfig(
        level=logging.INFO,  # Define o nível mínimo de log a ser registrado
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato das mensagens de log.
        handlers=[
            logging.FileHandler(
                log_file_path
            ),  # Handler para gravar logs em um arquivo.
            logging.StreamHandler(
                sys.stdout
            ),  # Handler para exibir logs no console (saída padrão).
        ],
    )
    # Configura o nível de logging para o módulo 'flet' para WARNING ou ERROR
    # para evitar logs muito verbosos do framework que podem poluir o console.
    logging.getLogger("flet").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )  # Para requests HTTP de flet/streamlit, que também podem ser verbosos.

    # Retorna um logger específico para o módulo que o chamou (__name__).
    # Isso permite que as mensagens de log incluam o nome do módulo de origem.
    return logging.getLogger(__name__)


def calculate_angle(p1: dict, p2: dict, p3: dict) -> float:
    """
    Calcula o ângulo em 3D entre três pontos (landmarks).

    Args:
        p1 (dict): Dicionário contendo as coordenadas (x, y, z) do primeiro ponto.
        p2 (dict): Dicionário contendo as coordenadas (x, y, z) do ponto central (vértice do ângulo).
        p3 (dict): Dicionário contendo as coordenadas (x, y, z) do terceiro ponto.

    Returns:
        float: O ângulo em graus entre os três pontos. Retorna 0.0 se houver pontos coincidentes.
    """
    # Converte os dicionários de pontos para arrays NumPy para facilitar operações matemáticas.
    p1_array = np.array([p1["x"], p1["y"], p1["z"]])
    p2_array = np.array([p2["x"], p2["y"], p2["z"]])
    p3_array = np.array([p3["x"], p3["y"], p3["z"]])

    # Cria vetores dos lados do ângulo, com p2 como origem.
    # v1 aponta de p2 para p1, e v2 aponta de p2 para p3.
    v1 = p1_array - p2_array
    v2 = p3_array - p2_array

    # Calcula o produto escalar dos dois vetores.
    # O produto escalar está relacionado ao cosseno do ângulo entre os vetores.
    dot_product = np.dot(v1, v2)

    # Calcula a magnitude (comprimento) de cada vetor.
    # A magnitude é necessária para normalizar o produto escalar.
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Verifica se alguma magnitude é zero para evitar divisão por zero.
    # Isso pode ocorrer se os pontos forem coincidentes.
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        logging.warning(
            "Ponto(s) coincidente(s) ou vetor(es) nulo(s) detectado(s) ao calcular ângulo. Retornando 0.0."
        )
        return 0.0

    # Calcula o cosseno do ângulo.
    # Adiciona np.clip para garantir que o valor esteja no intervalo [-1, 1] antes de np.arccos.
    # Isso evita erros de domínio para valores ligeiramente fora devido a imprecisões de ponto flutuante.
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # Converte o ângulo de radianos para graus e retorna.
    angle_deg = np.degrees(angle_rad)
    return angle_deg
