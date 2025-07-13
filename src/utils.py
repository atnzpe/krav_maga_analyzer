# src/utils.py

import logging
import os

def setup_logging(log_dir="logs", log_file_name="app.log", level=logging.INFO):
    """
    Configura o sistema de logging da aplicação.

    Args:
        log_dir (str): Diretório onde o arquivo de log será salvo.
        log_file_name (str): Nome do arquivo de log.
        level (int): Nível mínimo de log (e.g., logging.INFO, logging.DEBUG).
    """
    # Garante que o diretório de logs exista.
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)

    # Remove handlers existentes para evitar duplicação em re-configurações (útil em testes ou reloads).
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configura o logger para escrever em um arquivo e no console.
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path), # Salva logs em um arquivo.
            logging.StreamHandler() # Exibe logs no console.
        ]
    )
    # Retorna um logger específico para quem chamou a função, embora setup_logging
    # configure o logger raiz. Isso é um padrão comum para modules específicos.
    return logging.getLogger(__name__)

# Exemplo de uso (não será executado diretamente quando importado, apenas ilustrativo)
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Teste de log do módulo utils.")
