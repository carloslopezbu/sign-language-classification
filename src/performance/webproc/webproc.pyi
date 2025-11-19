"""Type stubs for webproc module."""

from typing import List

class Trie:
    """Estructura de datos Trie implementada en Rust."""

    def __init__(self) -> None:
        """Crea un nuevo Trie vacío."""
        ...

    def insert(self, word: str) -> None:
        """Inserta una palabra en el Trie."""
        ...

    def contains(self, word: str) -> bool:
        """Verifica si una palabra existe en el Trie."""
        ...

    def to_str(self) -> str:
        """Convierte el Trie a string para debugging."""
        ...

def get_the_meat_balls(metadata_path: str, languages: List[str]) -> None:
    """
    Extrae videos de SignTheSpread para los idiomas especificados.

    Args:
        metadata_path: Ruta al archivo CSV de metadata
        languages: Lista de códigos de idioma (ej: ["fr.fr", "de.at"])
    """
    ...
