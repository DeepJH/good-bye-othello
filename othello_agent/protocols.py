from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Optional
import numpy as np

class Player(IntEnum):
    BLACK = 1
    WHITE = -1

    @property
    def opponent(self) -> 'Player':
        return Player.WHITE if self == Player.BLACK else Player.BLACK

class CellState(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

@dataclass
class GridGeometry:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    rows: int = 8
    cols: int = 8

    def get_cell_center(self, row: int, col: int) -> Tuple[int, int]:
        left, top = self.top_left
        right, bottom = self.bottom_right
        cell_w = (right - left) / self.cols
        cell_h = (bottom - top) / self.rows
        center_x = int(left + (col + 0.5) * cell_w)
        center_y = int(top + (row + 0.5) * cell_h)
        return (center_x, center_y)

@dataclass
class BoardState:
    board: np.ndarray  # shape (8, 8) with values in {-1, 0, 1}
    current_player: Player
    last_move: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.board.ndim != 2 or self.board.shape[0] != self.board.shape[1]:
            raise ValueError(f"Board must be 2D square matrix, got shape {self.board.shape}")

    def count_pieces(self) -> Tuple[int, int]:
        black = int(np.sum(self.board == Player.BLACK.value))
        white = int(np.sum(self.board == Player.WHITE.value))
        return (black, white)
