# Othello AI Controller & Agent Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, decoupled Othello bot system connecting an AlphaZero AI model adapter to a device controller adapter (starting with ADB Android automation) through a reusable vision board recognizer and an orchestrator loop.

**Architecture:** Four decoupled layers:
1. `protocols.py`: Canonical data contracts (`BoardState`, `Player`, `CellState`, `GridGeometry`, `Move`).
2. `model_adapter.py`: Translates `BoardState` into AlphaZero MCTS input and outputs standardized `(row, col)` moves.
3. `vision/`: Standalone, reusable board recognition layer converting raw screenshots into `BoardState` and clickable `GridGeometry` with semi-automatic ROI calibration.
4. `controllers/`: Hardware/environment abstraction (`BaseController`, `AdbController`, `MockController`) providing screen capture and touch dispatch.
5. `orchestrator.py` & `run_bot.py`: Main game loop tying perception, decision, execution, and turn-inference together.

**Tech Stack:** Python 3.12, PyTorch 2.x, NumPy, Pillow, Pytest, ADB (`adb shell input tap`, `adb exec-out screencap -p`).

**Spec:** Architectural design approved in chat turn 2026-09-29.

## Global Constraints
- Standard 8x8 Othello/Reversi board (with extensible support for 6x6).
- Black = 1, White = -1, Empty = 0 throughout internal representations.
- Model adapter must handle perspective canonicalization automatically (if AI plays White, canonicalize board representation before calling MCTS).
- Vision recognition must be self-contained (takes image, returns `BoardState` + `GridGeometry`) without depending on ADB or network.
- Every task must be testable offline (using mock image generator or mock controller) before physical device connection.
- Commit to git after every completed task.

---

### Task 1: Standard Protocols & Data Contracts

**Files:**
- Create: `othello_agent/__init__.py`
- Create: `othello_agent/protocols.py`
- Test: `tests/test_protocols.py`

**Interfaces:**
- Consumes: Standard library dataclasses, enum, numpy.
- Produces: `Player`, `CellState`, `BoardState`, `GridGeometry`, `MoveResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocols.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_protocols.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'othello_agent'

- [ ] **Step 3: Write minimal implementation**

```python
# othello_agent/__init__.py
"""Othello AI Agent and Controller Adapter Package."""
```

```python
# othello_agent/protocols.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_protocols.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add othello_agent/ tests/test_protocols.py
git commit -m "feat(protocols): define standard interfaces and data contracts"
```

---

### Task 2: Model Adapter Layer

**Files:**
- Create: `othello_agent/model_adapter.py`
- Test: `tests/test_model_adapter.py`

**Interfaces:**
- Consumes: `othello_agent.protocols.BoardState`, `alpha-zero-general.othello.OthelloGame`, `alpha-zero-general.othello.pytorch.NNet`.
- Produces: `ModelAdapter.predict_move(board_state: BoardState) -> Optional[Tuple[int, int]]` (returns None if PASS).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_adapter.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_model_adapter.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'othello_agent.model_adapter'

- [ ] **Step 3: Write minimal implementation**

```python
# othello_agent/model_adapter.py
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
        if sum(valids) == 0 or valids[-1] == 1 and sum(valids) == 1:
            return None  # No valid moves or only pass

        action_probs = self.mcts.getActionProb(canonical_board, temp=0)
        action = int(np.argmax(action_probs))
        
        if action == self.board_size * self.board_size:
            return None  # Pass action
            
        row = action // self.board_size
        col = action % self.board_size
        return (row, col)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_model_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add othello_agent/model_adapter.py tests/test_model_adapter.py
git commit -m "feat(adapter): add AlphaZero model adapter translating standard BoardState"
```

---

### Task 3: Vision Module & Board Recognizer

**Files:**
- Create: `othello_agent/vision/__init__.py`
- Create: `othello_agent/vision/recognizer.py`
- Create: `othello_agent/vision/calibration.py`
- Test: `tests/test_vision.py`

**Interfaces:**
- Consumes: PIL Image / NumPy array, `protocols.GridGeometry`.
- Produces: `BoardRecognizer.recognize(image: Image) -> Tuple[np.ndarray, GridGeometry]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vision.py
import numpy as np
from PIL import Image, ImageDraw
import pytest
from othello_agent.protocols import GridGeometry, Player, CellState
from othello_agent.vision.recognizer import BoardRecognizer
from othello_agent.vision.calibration import BoardConfig

def create_synthetic_board_image():
    # Create 1000x1000 green board at (100, 100) to (900, 900)
    img = Image.new("RGB", (1000, 1000), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 900, 900], fill=(34, 139, 34))  # ForestGreen
    
    geo = GridGeometry(top_left=(100, 100), bottom_right=(900, 900), rows=8, cols=8)
    # Draw initial pieces
    for r, c, col in [(3, 3, (240, 240, 240)), (4, 4, (240, 240, 240)), 
                      (3, 4, (20, 20, 20)), (4, 3, (20, 20, 20))]:
        cx, cy = geo.get_cell_center(r, c)
        rad = 35
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)
    return img, geo

def test_board_recognizer_synthetic_image():
    img, geo = create_synthetic_board_image()
    config = BoardConfig(top_left=geo.top_left, bottom_right=geo.bottom_right, rows=8, cols=8)
    recognizer = BoardRecognizer(config)
    
    board, out_geo = recognizer.recognize(img)
    assert board.shape == (8, 8)
    assert board[3, 3] == CellState.WHITE.value
    assert board[4, 4] == CellState.WHITE.value
    assert board[3, 4] == CellState.BLACK.value
    assert board[4, 3] == CellState.BLACK.value
    assert board[0, 0] == CellState.EMPTY.value
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_vision.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'othello_agent.vision'

- [ ] **Step 3: Write minimal implementation**

```python
# othello_agent/vision/__init__.py
from .recognizer import BoardRecognizer
from .calibration import BoardConfig

__all__ = ["BoardRecognizer", "BoardConfig"]
```

```python
# othello_agent/vision/calibration.py
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Tuple, Optional
from ..protocols import GridGeometry

@dataclass
class BoardConfig:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    rows: int = 8
    cols: int = 8
    black_threshold: int = 70    # Max luminance for black piece
    white_threshold: int = 180   # Min luminance for white piece
    sample_ratio: float = 0.5    # Cell central area sample ratio

    def to_grid_geometry(self) -> GridGeometry:
        return GridGeometry(top_left=self.top_left, bottom_right=self.bottom_right, rows=self.rows, cols=self.cols)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'BoardConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['top_left'] = tuple(data['top_left'])
        data['bottom_right'] = tuple(data['bottom_right'])
        return cls(**data)
```

```python
# othello_agent/vision/recognizer.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_vision.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add othello_agent/vision/ tests/test_vision.py
git commit -m "feat(vision): implement reusable board recognizer and ROI calibration config"
```

---

### Task 4: Device Controller Abstraction & AdbController

**Files:**
- Create: `othello_agent/controllers/__init__.py`
- Create: `othello_agent/controllers/base.py`
- Create: `othello_agent/controllers/mock.py`
- Create: `othello_agent/controllers/adb.py`
- Test: `tests/test_controllers.py`

**Interfaces:**
- Consumes: `othello_agent.protocols.GridGeometry`, subprocess / mock.
- Produces: `BaseController` interface (`capture_screen() -> Image`, `tap_cell(r, c, geometry)`, `is_connected() -> bool`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_controllers.py
from PIL import Image
import pytest
from othello_agent.protocols import GridGeometry
from othello_agent.controllers.mock import MockController
from othello_agent.controllers.adb import AdbController

def test_mock_controller_capture_and_tap():
    dummy_img = Image.new("RGB", (100, 100), (0, 0, 0))
    ctrl = MockController(screen_image=dummy_img)
    assert ctrl.is_connected()
    
    img = ctrl.capture_screen()
    assert img.size == (100, 100)
    
    geo = GridGeometry(top_left=(0, 0), bottom_right=(80, 80), rows=8, cols=8)
    ctrl.tap_cell(2, 3, geo)
    assert len(ctrl.tap_history) == 1
    assert ctrl.tap_history[0] == (25, 35)  # center of (2, 3)

def test_adb_controller_instantiation():
    adb = AdbController(serial=None)
    # Method signatures exist and are callable
    assert hasattr(adb, "capture_screen")
    assert hasattr(adb, "tap_cell")
    assert hasattr(adb, "is_connected")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_controllers.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'othello_agent.controllers'

- [ ] **Step 3: Write minimal implementation**

```python
# othello_agent/controllers/base.py
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
```

```python
# othello_agent/controllers/mock.py
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
```

```python
# othello_agent/controllers/adb.py
import io
import subprocess
from typing import Optional
from PIL import Image
from .base import BaseController

class AdbController(BaseController):
    def __init__(self, serial: Optional[str] = None):
        self.serial = serial
        self._cmd_prefix = ["adb"]
        if self.serial:
            self._cmd_prefix.extend(["-s", self.serial])

    def _run_cmd(self, args, check=True):
        return subprocess.run(self._cmd_prefix + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

    def is_connected(self) -> bool:
        try:
            res = self._run_cmd(["get-state"], check=False)
            return b"device" in res.stdout
        except Exception:
            return False

    def capture_screen(self) -> Image.Image:
        # High speed binary screencap via exec-out
        res = self._run_cmd(["exec-out", "screencap", "-p"])
        if not res.stdout:
            raise RuntimeError(f"ADB screencap failed: {res.stderr.decode('utf-8', errors='ignore')}")
        return Image.open(io.BytesIO(res.stdout))

    def tap_screen(self, x: int, y: int):
        self._run_cmd(["shell", "input", "tap", str(int(x)), str(int(y))])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_controllers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add othello_agent/controllers/ tests/test_controllers.py
git commit -m "feat(controllers): add BaseController, MockController and AdbController"
```

---

### Task 5: Game Orchestrator & CLI Runner

**Files:**
- Create: `othello_agent/orchestrator.py`
- Create: `run_bot.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `ModelAdapter`, `BoardRecognizer`, `BaseController`.
- Produces: `GameOrchestrator.step() -> StepReport`, `run_bot.py` command line interface.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
from PIL import Image, ImageDraw
import numpy as np
import pytest
from othello_agent.protocols import GridGeometry, Player
from othello_agent.vision.calibration import BoardConfig
from othello_agent.vision.recognizer import BoardRecognizer
from othello_agent.controllers.mock import MockController
from othello_agent.model_adapter import AlphaZeroModelAdapter
from othello_agent.orchestrator import GameOrchestrator, StepStatus

def test_orchestrator_step_execution():
    # Create initial board image
    img = Image.new("RGB", (800, 800), (34, 139, 34))
    draw = ImageDraw.Draw(img)
    geo = GridGeometry(top_left=(0, 0), bottom_right=(800, 800), rows=8, cols=8)
    for r, c, col in [(3, 3, (240, 240, 240)), (4, 4, (240, 240, 240)), 
                      (3, 4, (20, 20, 20)), (4, 3, (20, 20, 20))]:
        cx, cy = geo.get_cell_center(r, c)
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=col)

    ctrl = MockController(screen_image=img)
    config = BoardConfig(top_left=(0, 0), bottom_right=(800, 800), rows=8, cols=8)
    recognizer = BoardRecognizer(config)
    adapter = AlphaZeroModelAdapter(board_size=8, num_sims=5)

    orchestrator = GameOrchestrator(
        controller=ctrl,
        recognizer=recognizer,
        model_adapter=adapter,
        ai_player=Player.BLACK  # AI plays Black
    )

    report = orchestrator.step()
    assert report.status == StepStatus.PLAYED
    assert report.move is not None
    assert len(ctrl.tap_history) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'othello_agent.orchestrator'

- [ ] **Step 3: Write minimal implementation**

```python
# othello_agent/orchestrator.py
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
        ai_player: Optional[Player] = None, # If None, inferred automatically
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
```

```python
# run_bot.py
import argparse
import time
from othello_agent.controllers.adb import AdbController
from othello_agent.vision.calibration import BoardConfig
from othello_agent.vision.recognizer import BoardRecognizer
from othello_agent.model_adapter import AlphaZeroModelAdapter
from othello_agent.orchestrator import GameOrchestrator, StepStatus
from othello_agent.protocols import Player

def main():
    parser = argparse.ArgumentParser(description="good-bye-othello: ADB AI Bot Runner")
    parser.add_argument('--config', type=str, default='config/board_config.json', help='Path to board ROI config')
    parser.add_argument('--color', type=str, default='auto', choices=['auto', 'black', 'white'], help='AI player color')
    parser.add_argument('--sims', type=int, default=25, help='MCTS simulation iterations per move')
    parser.add_argument('--interval', type=float, default=1.0, help='Polling interval (seconds)')
    parser.add_argument('--serial', type=str, default=None, help='ADB device serial')
    args = parser.parse_args()

    controller = AdbController(serial=args.serial)
    if not controller.is_connected():
        print("Error: No ADB device connected. Please connect your phone/emulator and enable USB debugging.")
        return

    config = BoardConfig.load(args.config)
    recognizer = BoardRecognizer(config)
    model = AlphaZeroModelAdapter(board_size=config.rows, num_sims=args.sims)

    ai_player = None
    if args.color == 'black':
        ai_player = Player.BLACK
    elif args.color == 'white':
        ai_player = Player.WHITE

    orchestrator = GameOrchestrator(controller, recognizer, model, ai_player=ai_player)
    print("Bot started! Press Ctrl+C to stop.")
    try:
        while True:
            report = orchestrator.step()
            print(f"[{time.strftime('%H:%M:%S')}] {report.status.value}: {report.message}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add othello_agent/orchestrator.py run_bot.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add GameOrchestrator loop and run_bot CLI entrypoint"
```

---

### Task 6: Documentation, Dependencies & Default Config Scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `config/default_8x8_config.json`
- Modify: `README.md`

- [ ] **Step 1: Update requirements.txt to include pillow**
- [ ] **Step 2: Provide default sample config and calibration guide in README**
- [ ] **Step 3: Run full pytest suite across all tests to ensure 100% green**
- [ ] **Step 4: Commit**

```bash
git add requirements.txt config/ README.md
git commit -m "docs: add bot usage guide, config presets and complete test suite verification"
```
