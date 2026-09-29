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
    # Center of row 2, col 3: left=0, cell_w=10 -> (3 + 0.5)*10 = 35; top=0, cell_h=10 -> (2 + 0.5)*10 = 25
    assert ctrl.tap_history[0] == (35, 25)

def test_adb_controller_instantiation():
    adb = AdbController(serial=None)
    # Method signatures exist and are callable
    assert hasattr(adb, "capture_screen")
    assert hasattr(adb, "tap_cell")
    assert hasattr(adb, "is_connected")
