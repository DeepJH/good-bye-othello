import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add alpha-zero-general directory to sys.path
AZG_DIR = Path(__file__).resolve().parent / "alpha-zero-general"
if str(AZG_DIR) not in sys.path:
    sys.path.insert(0, str(AZG_DIR))

from Arena import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import RandomPlayer, GreedyOthelloPlayer, HumanOthelloPlayer
from othello.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict

MODEL_DIR = str(AZG_DIR / 'pretrained_models' / 'othello' / 'pytorch')

def create_player(player_type, game, model_file=None, num_sims=50):
    if player_type == 'human':
        return HumanOthelloPlayer(game).play
    elif player_type == 'random':
        return RandomPlayer(game).play
    elif player_type == 'greedy':
        return GreedyOthelloPlayer(game).play
    elif player_type in ('alpha', 'cpu'):
        nnet = NNet(game)
        if model_file is None:
            model_file = '6x100x25_best.pth.tar' if game.n == 6 else '8x8_100checkpoints_best.pth.tar'
        nnet.load_checkpoint(MODEL_DIR, model_file)
        mcts_args = dotdict({'numMCTSSims': num_sims, 'cpuct': 1.0})
        mcts = MCTS(game, nnet, mcts_args)
        return lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def main():
    parser = argparse.ArgumentParser(description="good-bye-othello: AlphaZero Othello Play & Demo")
    parser.add_argument('--board-size', type=int, default=8, choices=[6, 8], help='Board dimension (6 or 8)')
    parser.add_argument('--player1', type=str, default='alpha', choices=['alpha', 'random', 'greedy', 'human'])
    parser.add_argument('--player2', type=str, default='random', choices=['alpha', 'random', 'greedy', 'human'])
    parser.add_argument('--sims', type=int, default=25, help='MCTS simulations per move')
    parser.add_argument('--games', type=int, default=1, help='Number of games (playGames plays games*2 rounds alternating sides)')
    parser.add_argument('--silent', action='store_true', help='Disable move-by-move board printing')
    args = parser.parse_args()

    game = OthelloGame(args.board_size)
    p1 = create_player(args.player1, game, num_sims=args.sims)
    p2 = create_player(args.player2, game, num_sims=args.sims)

    arena = Arena(p1, p2, game, display=OthelloGame.display)
    verbose = not args.silent

    if args.games == 1:
        print(f"Starting match: {args.player1} (Player 1, X) vs {args.player2} (Player -1, O) on {args.board_size}x{args.board_size} board...")
        result = arena.playGame(verbose=verbose)
        if result == 1:
            print(f"\nGame Over! Result: Player 1 ({args.player1}) won!")
        elif result == -1:
            print(f"\nGame Over! Result: Player 2 ({args.player2}) won!")
        else:
            print("\nGame Over! Result: Draw!")
    else:
        print(f"Starting tournament: {args.player1} vs {args.player2} ({args.games * 2} games with alternating sides)...")
        results = arena.playGames(args.games * 2, verbose=verbose)
        print(f"\nTournament results: {args.player1} won {results[0]}, {args.player2} won {results[1]}, draws: {results[2]}")


if __name__ == "__main__":
    main()
