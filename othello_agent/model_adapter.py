import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Ensure alpha-zero-general is accessible
AZG_DIR = Path(__file__).resolve().parent.parent / "alpha-zero-general"
if str(AZG_DIR) not in sys.path:
    sys.path.insert(0, str(AZG_DIR))

from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict
from .protocols import BoardState, Player

class AlphaZeroModelAdapter:
    def __init__(self, board_size: int = 8, num_sims: int = 25, model_dir: Optional[str] = None, model_file: Optional[str] = None):
        self.board_size = board_size
        self.num_sims = num_sims
        self.game = OthelloGame(board_size)
        self.nnet = NNet(self.game)

        if model_dir is None:
            model_dir = str(AZG_DIR / 'pretrained_models' / 'othello' / 'pytorch')
        if model_file is None:
            model_file = '6x100x25_best.pth.tar' if board_size == 6 else '8x8_100checkpoints_best.pth.tar'

        self.nnet.load_checkpoint(model_dir, model_file)
        self.mcts_args = dotdict({'numMCTSSims': num_sims, 'cpuct': 1.0})
        self.mcts = MCTS(self.game, self.nnet, self.mcts_args)

    def predict_move(self, state: BoardState) -> Optional[Tuple[int, int]]:
        # Canonical form: AlphaZero model always evaluates from player 1 perspective.
        # If current_player is WHITE (-1), multiply board by -1.
        player_val = state.current_player.value
        canonical_board = self.game.getCanonicalForm(state.board, player_val)
        
        valids = self.game.getValidMoves(canonical_board, 1)
        if sum(valids) == 0 or (valids[-1] == 1 and sum(valids) == 1):
            return None  # No valid moves or only pass

        action_probs = self.mcts.getActionProb(canonical_board, temp=0)
        action = int(np.argmax(action_probs))
        
        if action == self.board_size * self.board_size:
            return None  # Pass action
            
        row = action // self.board_size
        col = action % self.board_size
        return (row, col)
