# src/main_flet.py

import flet as ft
import logging
from src.utils import setup_logging
import os

# Configuração do logging para a aplicação Flet.
logger = setup_logging()


def main(page: ft.Page):
    """
    Função principal da aplicação Flet.
    Esta função configura a interface do usuário Flet.
    """
    logger.info("Iniciando a aplicação Flet...")

    page.title = "Analisador de Movimentos de Krav Maga (Flet)"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.window_width = 1200
    page.window_height = 800
    page.scroll = "adaptive"  # Adiciona scroll para conteúdo que exceda a tela.

    logger.info("Configurações da página Flet aplicadas.")

    def pick_files_result(e: ft.FilePickerResultEvent):
        """
        Callback para o resultado da seleção de arquivos.
        """
        if e.files:
            for f in e.files:
                logger.info(f"Arquivo selecionado: {f.name} ({f.path})")
                # Aqui você pode adicionar lógica para diferenciar vídeo do aluno/mestre
                # Por enquanto, apenas exibe o caminho.
                page.add(ft.Text(f"Arquivo selecionado: {f.name}"))
        else:
            logger.info("Seleção de arquivo cancelada.")
            page.add(ft.Text("Seleção de arquivo cancelada."))
        page.update()

    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)  # Adiciona o FilePicker ao overlay da página.

    page.add(
        ft.Column(
            [
                ft.Text(
                    "🥋 Analisador de Movimentos de Krav Maga (Flet - Em Desenvolvimento)",
                    size=24,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Text(
                    """
                    Bem-vindo ao Analisador de Movimentos de Krav Maga!
                    Esta é a versão desktop/APK da ferramenta, utilizando Visão Computacional
                    para comparar seus movimentos com os de mestres.
                    Funcionalidades de upload e análise serão implementadas aqui.
                """,
                    size=16,
                ),
                ft.ElevatedButton(
                    "Selecionar Vídeo do Aluno",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: file_picker.pick_files(
                        allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"]
                    ),
                ),
                ft.ElevatedButton(
                    "Selecionar Vídeo do Mestre",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: file_picker.pick_files(
                        allow_multiple=False, allowed_extensions=["mp4", "mov", "avi"]
                    ),
                ),
                ft.Text("Status: Aguardando upload de vídeos...", key="status_text"),
                # Placeholder para exibição de vídeos e resultados
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Text("Vídeo do Aluno"),
                            alignment=ft.alignment.center,
                            width=400,
                            height=300,
                            bgcolor=ft.colors.BLUE_GREY_100,
                            border_radius=10,
                        ),
                        ft.Container(
                            content=ft.Text("Vídeo do Mestre"),
                            alignment=ft.alignment.center,
                            width=400,
                            height=300,
                            bgcolor=ft.colors.BLUE_GREY_100,
                            border_radius=10,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                ),
                ft.Text(
                    "Resultados da Análise e Feedback: (Em breve)",
                    size=18,
                    weight=ft.FontWeight.MEDIUM,
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        )
    )
    logger.info("Elementos da UI Flet adicionados à página.")
    page.update()
    logger.info("Página Flet atualizada.")


# Ponto de entrada da aplicação Flet.
if __name__ == "__main__":
    ft.app(target=main)
