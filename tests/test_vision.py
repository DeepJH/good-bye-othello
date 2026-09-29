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

def test_board_config_save_load(tmp_path):
    config_path = str(tmp_path / "board_config.json")
    config = BoardConfig(top_left=(10, 20), bottom_right=(300, 400), rows=8, cols=8)
    config.save(config_path)
    
    loaded = BoardConfig.load(config_path)
    assert loaded.top_left == (10, 20)
    assert loaded.bottom_right == (300, 400)
    assert loaded.rows == 8
