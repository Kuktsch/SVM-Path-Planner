import numpy as np
from collections import deque

from .svm_model import SVMModel
from .config import PlannerConfig


class ConnectivityResult:
    def __init__(self, connected: bool, mask: np.ndarray, xs: np.ndarray, ys: np.ndarray):
        self.connected = connected
        self.mask = mask
        self.xs = xs
        self.ys = ys


class CorridorChecker:
    """Проверка связности коридора SVM"""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
    
    def build_grid(self, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Строит сетку для проверки связности"""
        x_min, x_max, y_min, y_max = bounds
        xs = np.arange(x_min, x_max + 1e-9, self.config.grid_resolution)
        ys = np.arange(y_min, y_max + 1e-9, self.config.grid_resolution)
        XX, YY = np.meshgrid(xs, ys)
        XY = np.stack([XX.ravel(), YY.ravel()], axis=1)
        return XY, xs, ys
    
    def check(self, model: SVMModel, bounds: tuple[float, float, float, float], start: tuple[float, float], goal: tuple[float, float]) -> ConnectivityResult:
        """Проверяет связность коридора между стартом и целью"""
        XY, xs, ys = self.build_grid(bounds)
        f = model.boundary_mask(XY, margin=1.0)
        mask = f.reshape(ys.shape[0], xs.shape[0])

        def to_idx(p: tuple[float, float]) -> tuple[int, int]:
            x, y = p
            ix = int(round((x - xs[0]) / (xs[1] - xs[0])))
            iy = int(round((y - ys[0]) / (ys[1] - ys[0])))
            ix = max(0, min(ix, xs.shape[0] - 1))
            iy = max(0, min(iy, ys.shape[0] - 1))
            return iy, ix

        sy, sx = to_idx(start)
        gy, gx = to_idx(goal)

        if not mask[sy, sx] or not mask[gy, gx]:
            return ConnectivityResult(False, mask, xs, ys)

        H, W = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        dq = deque([(sy, sx)])
        visited[sy, sx] = True
        neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while dq:
            y, x = dq.popleft()
            if y == gy and x == gx:
                return ConnectivityResult(True, mask, xs, ys)
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    dq.append((ny, nx))

        return ConnectivityResult(False, mask, xs, ys)

