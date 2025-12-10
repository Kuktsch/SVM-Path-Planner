import math
import warnings
import numpy as np
from scipy.interpolate import splprep, splev

from .svm_model import SVMModel
from .config import PlannerConfig


class PathResult:
    def __init__(self, points: list[tuple[float, float]], reached: bool):
        self.points = points
        self.reached = reached


class PathExtractor:
    """Извлечение пути из SVM модели"""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
    
    def smooth(self, points: list[tuple[float, float]], s: float = 0.2, num: int = 200) -> list[tuple[float, float]]:
        """Сглаживает путь с помощью сплайнов"""
        if len(points) < 4:
            return points
        xy = np.array(points)
        
        # Вычисляем адаптивный параметр сглаживания на основе длины пути
        path_length = self.length(points)
        if path_length > 0:
            # Адаптивный s: увеличиваем для длинных путей
            adaptive_s = max(s, path_length * 0.01)
        else:
            adaptive_s = s
        
        # Подавляем предупреждение о слишком малом s, если оно не критично
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*s too small.*", category=UserWarning)
            try:
                tck, _ = splprep([xy[:, 0], xy[:, 1]], s=adaptive_s)
            except (ValueError, TypeError):
                # Если сплайн не работает, пробуем с увеличенным s
                adaptive_s = max(adaptive_s * 2, 1.0)
                tck, _ = splprep([xy[:, 0], xy[:, 1]], s=adaptive_s)
        
        ts = np.linspace(0.0, 1.0, num)
        xs, ys = splev(ts, tck)
        return list(zip(xs, ys))
    
    def length(self, points: list[tuple[float, float]]) -> float:
        """Вычисляет длину пути"""
        if len(points) < 2:
            return 0.0
        xy = np.array(points)
        diffs = np.diff(xy, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        return float(seg.sum())
    
    def _find_zero_cross_along_ray(self, model: SVMModel, origin: np.ndarray, direction: np.ndarray, ds: float, max_dist: float, margin: float = 1.0) -> tuple[bool, np.ndarray]:
        """Находит пересечение нуля вдоль луча"""
        f_origin = model.decision(origin.reshape(1, -1))[0]
        if abs(f_origin) < 0.1:
            return True, origin
        
        prev_f = f_origin
        p_prev = origin
        steps = max(1, int(max_dist / ds))
        
        for i in range(1, steps + 1):
            p = origin + direction * (i * ds)
            f = model.decision(p.reshape(1, -1))[0]
            
            if abs(f) >= margin:
                if abs(prev_f) < margin:
                    if abs(prev_f) < abs(f):
                        return True, p_prev
                    else:
                        t = abs(prev_f) / (abs(prev_f) + abs(f) + 1e-12)
                        z = p_prev * (1 - t) + p * t
                        return True, z
                return False, origin
            
            if prev_f is not None and (prev_f <= 0.0 < f or prev_f >= 0.0 > f):
                t = abs(prev_f) / (abs(prev_f) + abs(f) + 1e-12)
                z = p_prev * (1 - t) + p * t
                return True, z
            
            if abs(f) < 0.05:
                return True, p
            
            prev_f = f
            p_prev = p
        
        if abs(prev_f) < margin * 0.8:
            return True, p_prev
        
        return False, origin
    
    def extract(self, model: SVMModel, start: tuple[float, float], goal: tuple[float, float]) -> PathResult:
        """Извлекает путь из модели SVM"""
        p = np.array(start, dtype=float)
        goal_np = np.array(goal, dtype=float)

        path: list[tuple[float, float]] = [tuple(p)]
        sector = math.radians(self.config.sector_angle_deg)

        for _ in range(self.config.max_steps):
            to_goal = goal_np - p
            dist_goal = float(np.linalg.norm(to_goal))
            if dist_goal < self.config.step_ds * 1.2:
                path.append((float(goal_np[0]), float(goal_np[1])))
                return PathResult(points=path, reached=True)

            heading = to_goal / (dist_goal + 1e-12)
            best_point = None
            best_cost = float("inf")
            
            f_current = model.decision(p.reshape(1, -1))[0]
            if abs(f_current) < 0.5:
                direct_step = min(self.config.step_ds * 1.5, dist_goal)
                p_direct = p + heading * direct_step
                f_direct = model.decision(p_direct.reshape(1, -1))[0]
                if abs(f_direct) < 1.0:
                    best_point = p_direct
                    best_cost = float(np.linalg.norm(goal_np - p_direct))
            
            num_dirs = 17
            for k in range(num_dirs):
                ang = -sector + (2 * sector) * (k / (num_dirs - 1))
                d = np.array([
                    heading[0] * math.cos(ang) - heading[1] * math.sin(ang),
                    heading[0] * math.sin(ang) + heading[1] * math.cos(ang),
                ])
                ok, z = self._find_zero_cross_along_ray(model, p, d, self.config.step_ds, max_dist=min(3.0, dist_goal))
                if not ok:
                    continue
                dist_to_goal = float(np.linalg.norm(goal_np - z))
                f_z = model.decision(z.reshape(1, -1))[0]
                cost = dist_to_goal + 0.3 * abs(f_z)
                if cost < best_cost:
                    best_cost = cost
                    best_point = z

            if best_point is None:
                if abs(f_current) < 1.0:
                    small_step = min(self.config.step_ds * 0.5, dist_goal)
                    p_fallback = p + heading * small_step
                    f_fallback = model.decision(p_fallback.reshape(1, -1))[0]
                    if abs(f_fallback) < 1.5: 
                        best_point = p_fallback
                    else:
                        return PathResult(points=path, reached=False)
                else:
                    return PathResult(points=path, reached=False)

            p = best_point
            path.append((float(p[0]), float(p[1])))

        return PathResult(points=path, reached=False)
