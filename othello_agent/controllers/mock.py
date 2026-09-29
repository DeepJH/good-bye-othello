from typing import List, Tuple, Optional
from PIL import Image
from .base import BaseController

class MockController(BaseController):
    def __init__(self, screen_image: Optional[Image.Image] = None):
        self.screen_image = screen_image or Image.new("RGB", (500, 500), (0, 0, 0))
        self.tap_history: List[Tuple[int, int]] = []

    def capture_screen(self) -> Image.Image:
        return self.screen_image.copy()

    def tap_screen(self, x: int, y: int):
        self.tap_history.append((x, y))

    def is_connected(self) -> bool:
        return True
