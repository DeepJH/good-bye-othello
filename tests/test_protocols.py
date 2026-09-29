import numpy as np
import pytest
from othello_agent.protocols import Player, CellState, BoardState, GridGeometry

def test_player_and_cell_state_values():
    assert Player.BLACK.value == 1
    assert Player.WHITE.value == -1
    assert CellState.EMPTY.value == 0
    assert CellState.BLACK.value == 1
    assert CellState.WHITE.value == -1

def test_board_state_validation():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1
    board[3, 4] = board[4, 3] = 1
    state = BoardState(board=board, current_player=Player.BLACK)
    assert state.board.shape == (8, 8)
    assert state.current_player == Player.BLACK
    assert state.count_pieces() == (2, 2)  # (black, white)

def test_grid_geometry_cell_center():
    geo = GridGeometry(top_left=(100, 200), bottom_right=(900, 1000), rows=8, cols=8)
    cx, cy = geo.get_cell_center(0, 0)
    assert cx == 150
    assert cy == 250
