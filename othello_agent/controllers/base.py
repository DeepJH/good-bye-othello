from abc import ABC, abstractmethod
from PIL import Image
from ..protocols import GridGeometry

class BaseController(ABC):
    @abstractmethod
    def capture_screen(self) -> Image.Image:
        pass

    @abstractmethod
    def tap_screen(self, x: int, y: int):
        pass

    def tap_cell(self, row: int, col: int, geometry: GridGeometry):
        cx, cy = geometry.get_cell_center(row, col)
        self.tap_screen(cx, cy)

    @abstractmethod
    def is_connected(self) -> bool:
        pass
