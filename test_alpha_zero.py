"""
Comprehensive automated test suite for AlphaZero Othello inference and gameplay.
"""
import sys
from pathlib import Path
import numpy as np

AZG_DIR = Path(__file__).resolve().parent / "alpha-zero-general"
if str(AZG_DIR) not in sys.path:
    sys.path.insert(0, str(AZG_DIR))

from Arena import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import RandomPlayer, GreedyOthelloPlayer
from othello.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict

MODEL_DIR = str(AZG_DIR / 'pretrained_models' / 'othello' / 'pytorch')


def test_6x6_model_inference():
    print("[1/3] Testing 6x6 pretrained model inference (4 turns)...")
    game = OthelloGame(6)
    nnet = NNet(game)
    nnet.load_checkpoint(MODEL_DIR, '6x100x25_best.pth.tar')
    mcts = MCTS(game, nnet, dotdict({'numMCTSSims': 10, 'cpuct': 1.0}))
    ai = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    rp = RandomPlayer(game).play

    board = game.getInitBoard()
    curPlayer = 1
    for turn in range(1, 5):
        canonical = game.getCanonicalForm(board, curPlayer)
        action = ai(canonical) if curPlayer == 1 else rp(canonical)
        assert game.getValidMoves(canonical, 1)[action] > 0, f"Turn {turn}: Invalid move chosen"
        board, curPlayer = game.getNextState(board, curPlayer, action)
    print("  -> 6x6 model inference: OK")


def test_8x8_model_inference():
    print("[2/3] Testing 8x8 pretrained model inference (4 turns)...")
    game = OthelloGame(8)
    nnet = NNet(game)
    nnet.load_checkpoint(MODEL_DIR, '8x8_100checkpoints_best.pth.tar')
    mcts = MCTS(game, nnet, dotdict({'numMCTSSims': 10, 'cpuct': 1.0}))
    ai = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    rp = RandomPlayer(game).play

    board = game.getInitBoard()
    curPlayer = 1
    for turn in range(1, 5):
        canonical = game.getCanonicalForm(board, curPlayer)
        action = ai(canonical) if curPlayer == 1 else rp(canonical)
        assert game.getValidMoves(canonical, 1)[action] > 0, f"Turn {turn}: Invalid move chosen"
        board, curPlayer = game.getNextState(board, curPlayer, action)
    print("  -> 8x8 model inference: OK")


def test_complete_game():
    print("[3/3] Testing complete game play-through (6x6 AI vs Greedy)...")
    game = OthelloGame(6)
    nnet = NNet(game)
    nnet.load_checkpoint(MODEL_DIR, '6x100x25_best.pth.tar')
    mcts = MCTS(game, nnet, dotdict({'numMCTSSims': 15, 'cpuct': 1.0}))
    ai = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    greedy = GreedyOthelloPlayer(game).play

    arena = Arena(ai, greedy, game)
    result = arena.playGame(verbose=False)
    assert result in (1, -1, 0), f"Unexpected game result: {result}"
    print(f"  -> Complete game finished with valid termination result ({result}): OK")


if __name__ == "__main__":
    test_6x6_model_inference()
    test_8x8_model_inference()
    test_complete_game()
    print("\nAll tests PASSED successfully!")
