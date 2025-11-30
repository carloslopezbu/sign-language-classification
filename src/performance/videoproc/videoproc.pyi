# videoproc.pyi
from typing import overload

class Rect:
    """A Region of Interest, representing a rectangle in an Image."""

    # Propiedades expuestas con .def_readwrite
    x: int
    y: int
    width: int
    height: int

    @overload
    def __init__(self) -> None:
        """Initialize an empty rectangle (0,0,0,0)."""
        ...

    @overload
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """Initialize a rectangle with specific coordinates and dimensions."""
        ...

    def area(self) -> int:
        """Computes the area of the ROI (width x height)."""
        ...

def align_sign_text_sequences(
    video_path: str,
    text_detector_path: str,
    text_recognitor_path: str,
    roi_ratio: float,
) -> None:
    """
    Procesa el video buscando texto en la parte inferior de la pantalla.

    Args:
        video_path: Ruta al video.
        text_detector_path: Ruta al modelo EAST (.pb).
        text_recognitor_path: Ruta a la carpeta tessdata.
        roi_ratio: Porcentaje de altura desde abajo (0.0 a 1.0).
                   Ej: 0.3 usa el 30% inferior de la imagen.
    """
    ...
