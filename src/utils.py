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
        level=logging.INFO,  # Define o nível mínimo de log a ser registrado (INFO, DEBUG, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Define o formato das mensagens de log
        handlers=[
            logging.FileHandler(
                log_file_path
            ),  # Handler para gravar logs em um arquivo
            logging.StreamHandler(
                sys.stdout
            ),  # Handler para exibir logs no console (saída padrão)
        ],
    )
    # Configura o nível de logging para módulos específicos para evitar logs excessivamente verbosos
    logging.getLogger("flet").setLevel(
        logging.WARNING
    )  # Ignora logs de INFO/DEBUG de Flet
    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )  # Ignora logs de INFO/DEBUG de requisições HTTP

    return logging.getLogger(
        __name__
    )  # Retorna um logger específico para o módulo atual


def calculate_angle(p1: dict, p2: dict, p3: dict) -> float:
    """
    Calcula o ângulo em 3D entre três pontos (p1-p2-p3) onde p2 é o vértice.
    Esta função é crucial para a análise de movimento, permitindo quantificar
    a posição relativa das articulações do corpo.

    Argumentos:
        p1 (dict): Primeiro ponto (ex: Ombro), com chaves 'x', 'y', 'z' representando coordenadas 3D.
        p2 (dict): Ponto do vértice (ex: Cotovelo), com chaves 'x', 'y', 'z'. Este é o ponto central do ângulo.
        p3 (dict): Terceiro ponto (ex: Punho), com chaves 'x', 'y', 'z'.

    Retorna:
        float: O ângulo em graus, entre 0 e 180. Retorna 0.0 se houver pontos coincidentes
               ou vetores nulos para evitar erros matemáticos.

    Bibliotecas Utilizadas:
        numpy: Para operações de vetorização e cálculo de produto escalar e norma.
    """
    # Converte os dicionários de pontos em arrays NumPy.
    # Isso permite realizar operações matemáticas de vetor de forma eficiente.
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
    # Adiciona np.clip para garantir que o valor esteja no intervalo [-1, 1] devido a pequenas imprecisões de ponto flutuante.
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Converte o cosseno do ângulo de volta para graus.
    angle_rad = np.arccos(cosine_angle)  # Ângulo em radianos
    angle_deg = np.degrees(angle_rad)  # Converte radianos para graus

    return angle_deg
