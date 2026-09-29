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
