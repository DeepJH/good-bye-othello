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
