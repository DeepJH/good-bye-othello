import numpy as np
import pytest
from othello_agent.protocols import Player, BoardState
from othello_agent.model_adapter import AlphaZeroModelAdapter

def test_model_adapter_initial_board_prediction():
    adapter = AlphaZeroModelAdapter(board_size=8, num_sims=10)
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1
    board[3, 4] = board[4, 3] = 1
    state = BoardState(board=board, current_player=Player.BLACK)
    
    move = adapter.predict_move(state)
    assert move is not None
    r, c = move
    # Initial legal moves for Black in standard Othello
    assert (r, c) in {(2, 3), (3, 2), (4, 5), (5, 4)}

def test_model_adapter_white_perspective():
    adapter = AlphaZeroModelAdapter(board_size=8, num_sims=10)
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1
    board[3, 4] = board[4, 3] = 1
    # After Black plays (2, 3), pieces flipped: (3, 3) becomes 1
    board[2, 3] = 1
    board[3, 3] = 1
    state = BoardState(board=board, current_player=Player.WHITE)
    move = adapter.predict_move(state)
    assert move is not None
    # Legal moves for White
    assert (move[0], move[1]) in {(2, 2), (2, 4), (4, 2)}
