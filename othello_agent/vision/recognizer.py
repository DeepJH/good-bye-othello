from typing import Tuple
from PIL import Image
import numpy as np
from ..protocols import GridGeometry, CellState
from .calibration import BoardConfig

class BoardRecognizer:
    def __init__(self, config: BoardConfig):
        self.config = config
        self.geometry = config.to_grid_geometry()

    def recognize(self, image: Image.Image) -> Tuple[np.ndarray, GridGeometry]:
        # Convert to grayscale for robust luminance sampling
        gray = image.convert('L')
        board = np.zeros((self.config.rows, self.config.cols), dtype=int)
        
        left, top = self.config.top_left
        right, bottom = self.config.bottom_right
        cell_w = (right - left) / self.config.cols
        cell_h = (bottom - top) / self.config.rows
        
        sample_w = cell_w * self.config.sample_ratio
        sample_h = cell_h * self.config.sample_ratio
        
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                cx, cy = self.geometry.get_cell_center(r, c)
                box = (
                    int(cx - sample_w / 2),
                    int(cy - sample_h / 2),
                    int(cx + sample_w / 2),
                    int(cy + sample_h / 2)
                )
                cell_crop = gray.crop(box)
                avg_luma = float(np.mean(np.array(cell_crop)))
                
                if avg_luma <= self.config.black_threshold:
                    board[r, c] = CellState.BLACK.value
                elif avg_luma >= self.config.white_threshold:
                    board[r, c] = CellState.WHITE.value
                else:
                    board[r, c] = CellState.EMPTY.value

        return board, self.geometry
