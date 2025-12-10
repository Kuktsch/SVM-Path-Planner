import math
import numpy as np

from .config import PlannerConfig
from .environment import Environment
from .geometry import GeometryUtils, Polygon
from .svm_model import SVMModel


class CollisionChecker:
    """Проверка коллизий пути"""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
    
    def path_within_corridor(self, model: SVMModel, path: list[tuple[float, float]], margin: float = 1.0) -> bool:
        """Проверяет, находится ли путь в коридоре SVM"""
        XY = np.array(path, dtype=float)
        f = model.decision(XY)
        return bool(np.all(np.abs(f) < margin))
    
    def vehicle_collision_free(self, env: Environment, path: list[tuple[float, float]]) -> bool:
        """Проверяет, свободен ли путь от столкновений с препятствиями"""
        if len(path) < 2:
            return True
        for i in range(1, len(path)):
            p_prev = path[i - 1]
            p_curr = path[i]
            heading = math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
            rect = Polygon(GeometryUtils.rectangle_corners(p_curr, self.config.half_length, self.config.half_width, heading))
            for obs in env.obstacles:
                if rect.intersects(obs.polygon):
                    return False
        return True
