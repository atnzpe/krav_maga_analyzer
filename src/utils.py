import logging
import os


def setup_logging():
    """
    Configura o sistema de logging para a aplicação.
    Cria um diretório 'logs' se não existir e configura o logger
    para gravar em 'app.log' e exibir no console.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, "app.log")

    # Remove handlers existentes para evitar duplicação em re-configurações
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,  # Nível padrão de logging
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(
                sys.stdout
            ),  # sys.stdout para garantir saída no console
        ],
    )
    # Configura o nível de logging para o módulo 'flet' para WARNING ou ERROR
    # para evitar logs muito verbosos do framework.
    logging.getLogger("flet").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )  # Para requests HTTP de flet/streamlit

    return logging.getLogger(__name__)


# Adicione esta importação para o StreamHandler funcionar corretamente.
# Isso já estaria implícito se você estivesse usando este arquivo como um módulo,
# mas é bom ser explícito.
import sys
