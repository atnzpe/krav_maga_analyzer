# src/utils.py

import logging
import flet as ft # Importar flet é necessário para a classe FeedbackManager

# Configuração básica do logger
def setup_logging():
    """
    Configura o sistema de logging para a aplicação, definindo o formato
    e o nível de saída para INFO.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Configuração de logging inicializada.")

def get_logger(name: str):
    """
    Retorna uma instância de logger com o nome especificado.

    Args:
        name (str): O nome do logger (geralmente __name__ do módulo).

    Returns:
        logging.Logger: A instância do logger.
    """
    return logging.getLogger(name)

class FeedbackManager:
    """
    Gerencia a exibição de feedback e mensagens de status na interface do usuário (UI).
    Usa um controle `ft.Text` para exibir mensagens, atualizando-o conforme necessário.
    """

    def __init__(self, feedback_text_control: ft.Text = None):
        """
        Inicializa o FeedbackManager.

        Args:
            feedback_text_control (ft.Text): O controle ft.Text onde as mensagens serão exibidas.
                                             Se não for fornecido, um novo ft.Text será criado.
        """
        if feedback_text_control is None:
            # Cria um controle ft.Text padrão se nenhum for fornecido
            self.feedback_text_control = ft.Text(
                value="Aguardando ações...",
                color=ft.colors.BLACK,
                size=16,
                weight=ft.FontWeight.NORMAL,
                text_align=ft.TextAlign.CENTER,
            )
        else:
            self.feedback_text_control = feedback_text_control
        get_logger(__name__).info("FeedbackManager inicializado.") # Usando get_logger aqui

    def set_feedback_control(self, control: ft.Text):
        """
        Define ou atualiza o controle ft.Text que será usado para exibir o feedback.
        Isso permite que o controle seja criado na função `main` do Flet e depois
        passado para o FeedbackManager.
        """
        self.feedback_text_control = control
        get_logger(__name__).info("Controle de feedback na UI atualizado no FeedbackManager.")

    def update_feedback(self, page: ft.Page, message: str, is_error: bool = False):
        """
        Atualiza o texto de feedback na UI e a página.

        Args:
            page (ft.Page): A instância da página Flet para atualização.
            message (str): A mensagem a ser exibida.
            is_error (bool): Se `True`, a mensagem será formatada como um erro.
        """
        get_logger(__name__).info(f"Atualizando feedback: {message} (Erro: {is_error})")
        self.feedback_text_control.value = message
        self.feedback_text_control.color = ft.colors.RED if is_error else ft.colors.BLACK
        page.update() # Atualiza a página para refletir a mudança.
        get_logger(__name__).debug("Página da UI atualizada com novo feedback.")

# Outras classes ou funções utilitárias podem vir aqui

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
