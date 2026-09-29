from dataclasses import dataclass
from enum import Enum
import time
from typing import Optional, Tuple
import numpy as np
from .protocols import BoardState, Player
from .vision.recognizer import BoardRecognizer
from .controllers.base import BaseController
from .model_adapter import AlphaZeroModelAdapter

class StepStatus(Enum):
    PLAYED = "PLAYED"
    WAITING_OPPONENT = "WAITING_OPPONENT"
    PASS = "PASS"
    GAME_OVER = "GAME_OVER"
    ERROR = "ERROR"

@dataclass
class StepReport:
    status: StepStatus
    move: Optional[Tuple[int, int]] = None
    board: Optional[np.ndarray] = None
    message: str = ""

class GameOrchestrator:
    def __init__(
        self,
        controller: BaseController,
        recognizer: BoardRecognizer,
        model_adapter: AlphaZeroModelAdapter,
        ai_player: Optional[Player] = None,
    ):
        self.controller = controller
        self.recognizer = recognizer
        self.model = model_adapter
        self.ai_player = ai_player
        self.last_board: Optional[np.ndarray] = None

    def infer_current_player(self, board: np.ndarray) -> Player:
        # Standard Othello: Black starts, piece count is even => Black's turn; odd => White's turn
        black_cnt = int(np.sum(board == Player.BLACK.value))
        white_cnt = int(np.sum(board == Player.WHITE.value))
        total = black_cnt + white_cnt
        return Player.BLACK if total % 2 == 0 else Player.WHITE

    def step(self) -> StepReport:
        # 1. Capture screen
        img = self.controller.capture_screen()
        # 2. Recognize board
        board, geometry = self.recognizer.recognize(img)

        # 3. Infer turn and AI color if not set
        current_turn = self.infer_current_player(board)
        if self.ai_player is None:
            self.ai_player = current_turn

        if current_turn != self.ai_player:
            return StepReport(status=StepStatus.WAITING_OPPONENT, board=board, message="Waiting for opponent move...")

        # 4. Predict move
        state = BoardState(board=board, current_player=self.ai_player)
        move = self.model.predict_move(state)
        
        if move is None:
            return StepReport(status=StepStatus.PASS, board=board, message="No legal moves, passing.")

        # 5. Execute move
        self.controller.tap_cell(move[0], move[1], geometry)
        self.last_board = board
        return StepReport(status=StepStatus.PLAYED, move=move, board=board, message=f"Played move {move}")
