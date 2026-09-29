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

def test_orchestrator_waiting_opponent():
    # Board where it is White's turn (total pieces = 5, odd)
    img = Image.new("RGB", (800, 800), (34, 139, 34))
    draw = ImageDraw.Draw(img)
    geo = GridGeometry(top_left=(0, 0), bottom_right=(800, 800), rows=8, cols=8)
    # 5 pieces
    for r, c, col in [
        (3, 3, (20, 20, 20)), (4, 4, (240, 240, 240)), 
        (3, 4, (20, 20, 20)), (4, 3, (20, 20, 20)),
        (2, 3, (20, 20, 20))
    ]:
        cx, cy = geo.get_cell_center(r, c)
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=col)

    ctrl = MockController(screen_image=img)
    config = BoardConfig(top_left=(0, 0), bottom_right=(800, 800), rows=8, cols=8)
    recognizer = BoardRecognizer(config)
    adapter = AlphaZeroModelAdapter(board_size=8, num_sims=5)

    # AI is Black, but it's White's turn
    orchestrator = GameOrchestrator(
        controller=ctrl,
        recognizer=recognizer,
        model_adapter=adapter,
        ai_player=Player.BLACK
    )

    report = orchestrator.step()
    assert report.status == StepStatus.WAITING_OPPONENT
    assert len(ctrl.tap_history) == 0
