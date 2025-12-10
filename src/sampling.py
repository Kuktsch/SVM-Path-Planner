import numpy as np

from .environment import Environment
from .geometry import Point, GeometryUtils, Polygon
from .config import PlannerConfig


class Samples:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y


class SampleGenerator:
    """Генератор обучающих выборок для SVM"""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
    
    def _interpolate_edge(self, a: Point, b: Point, n: int) -> list[Point]:
        """Интерполирует точки вдоль ребра"""
        return [(
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
        ) for t in np.linspace(0.0, 1.0, n, endpoint=True)]
    
    def _polygon_distance_to_segment(self, poly: Polygon, seg_start: Point, seg_end: Point) -> float:
        """Вычисляет минимальное расстояние от многоугольника до отрезка линии"""
        min_dist = float('inf')
        intersects = False
        
        if poly.contains_point(seg_start) or poly.contains_point(seg_end):
            intersects = True
        
        for t in np.linspace(0.0, 1.0, 10):
            seg_pt = (
                seg_start[0] + t * (seg_end[0] - seg_start[0]),
                seg_start[1] + t * (seg_end[1] - seg_start[1])
            )
            if poly.contains_point(seg_pt):
                intersects = True
                break
        
        seg_dir = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        for edge_start, edge_end in poly.edges():
            edge_dir = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
            det = seg_dir[0] * edge_dir[1] - seg_dir[1] * edge_dir[0]
            if abs(det) < 1e-10:
                continue
            t1 = ((edge_start[0] - seg_start[0]) * edge_dir[1] - (edge_start[1] - seg_start[1]) * edge_dir[0]) / det
            t2 = ((edge_start[0] - seg_start[0]) * seg_dir[1] - (edge_start[1] - seg_start[1]) * seg_dir[0]) / det
            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                intersects = True
                break
        
        for v in poly.vertices:
            dist = GeometryUtils.point_to_segment_distance(v, seg_start, seg_end)
            if dist < min_dist:
                min_dist = dist
        
        if intersects:
            return -min_dist
        return min_dist
    
    def _find_obstacle_projection_on_line(self, obs: Polygon, line_start: np.ndarray, line_dir: np.ndarray, line_length: float) -> tuple[float, np.ndarray]:
        """Находит точку на линии, ближайшую к препятствию"""
        verts = obs.vertices
        centroid = np.array([
            sum(v[0] for v in verts) / len(verts),
            sum(v[1] for v in verts) / len(verts)
        ])
        rel = centroid - line_start
        t = np.dot(rel, line_dir) / line_length
        t = max(0.0, min(1.0, t))
        proj_point = line_start + t * line_dir * line_length
        return t, proj_point
    
    def _check_point_in_bounds(self, p: np.ndarray, bounds: tuple[float, float, float, float]) -> bool:
        """Проверяет, находится ли точка в пределах рабочей области"""
        x_min, x_max, y_min, y_max = bounds
        return x_min <= p[0] <= x_max and y_min <= p[1] <= y_max
    
    def _shift_point_away_from_obstacles(
        self,
        p: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        obstacles: list[Polygon],
        bounds: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, bool]:
        """Сдвигает точку от препятствий, если она сталкивается"""
        p_shifted = p.copy()
        
        for obs in obstacles:
            dist = obs.distance_to_point(tuple(p_shifted))
            if dist < 0:
                penetration = abs(dist)
                obs_centroid = np.array([
                    sum(v[0] for v in obs.vertices) / len(obs.vertices),
                    sum(v[1] for v in obs.vertices) / len(obs.vertices)
                ])
                rel_obs = obs_centroid - center
                side = np.sign(np.dot(rel_obs, normal))
                shift_magnitude = min(self.config.max_shift, penetration + self.config.shift_eps)
                p_shifted = p_shifted - side * normal * shift_magnitude
                dist_after = obs.distance_to_point(tuple(p_shifted))
                if dist_after < 0:
                    p_shifted = p_shifted - side * normal * self.config.max_shift
                    dist_after = obs.distance_to_point(tuple(p_shifted))
                    if dist_after < 0:
                        return p, False
        
        if not self._check_point_in_bounds(p_shifted, bounds):
            return p, False
        
        return p_shifted, True
    
    def get_initial_obstacle_labels(self, env: Environment, start: np.ndarray, normal: np.ndarray) -> list[int]:
        """Получает начальные метки препятствий"""
        labels = []
        for obs in env.obstacles:
            verts = obs.polygon.vertices
            centroid = np.array([
                sum(v[0] for v in verts) / len(verts),
                sum(v[1] for v in verts) / len(verts)
            ])
            rel = centroid - start
            side = np.sign(np.dot(rel, normal))
            labels.append(1 if side >= 0 else -1)
        return labels
    
    def generate(self, env: Environment, obstacle_labels: list[int] | None = None) -> Samples:
        """Генерирует обучающие выборки"""
        start = np.array(env.start)
        goal = np.array(env.goal)
        dir_vec = goal - start
        line_length = np.linalg.norm(dir_vec)
        if line_length < 1e-12:
            dir_vec = np.array([1.0, 0.0])
            line_length = 1.0
        else:
            dir_vec = dir_vec / line_length
        normal = np.array([-dir_vec[1], dir_vec[0]])

        X_list: list[tuple[float, float]] = []
        y_list: list[int] = []

        if obstacle_labels is None:
            obstacle_labels = self.get_initial_obstacle_labels(env, start, normal)
        
        for obs_idx, obs in enumerate(env.obstacles):
            obs_label = obstacle_labels[obs_idx]
            verts = obs.polygon.vertices
            n = len(verts)
            for i in range(n):
                a = verts[i]
                b = verts[(i + 1) % n]
                pts = self._interpolate_edge(a, b, self.config.border_samples_per_edge)
                for p in pts:
                    X_list.append((p[0], p[1]))
                    y_list.append(obs_label)

        dg = self.config.guide_sample_distance
        dp = self.config.guide_sample_spacing
        n_start = self.config.guide_samples_near_points
        all_obstacles = [obs.polygon for obs in env.obstacles]
        L_local = self.config.obstacle_proximity_threshold
        
        obstacle_zones: list[tuple[float, Polygon, bool]] = []
        for obs in env.obstacles:
            dist = self._polygon_distance_to_segment(obs.polygon, tuple(start), tuple(goal))
            intersects = dist < 0
            if dist <= L_local:
                t, proj = self._find_obstacle_projection_on_line(obs.polygon, start, dir_vec, line_length)
                obstacle_zones.append((t, obs.polygon, intersects))
        
        obstacle_zones.sort(key=lambda x: x[0])
        
        if len(obstacle_zones) == 0:
            num_guides = max(4, int(line_length / dp))
            for i in range(num_guides + 1):
                t = i / max(1, num_guides)
                center = start + dir_vec * (t * line_length)
                for sgn in (-1.0, 1.0):
                    p = center + sgn * normal * dg
                    p_shifted, is_valid = self._shift_point_away_from_obstacles(
                        p, center, normal, all_obstacles, env.bounds
                    )
                    if is_valid:
                        X_list.append((float(p_shifted[0]), float(p_shifted[1])))
                        y_list.append(1 if sgn > 0 else -1)
        else:
            for i in range(n_start):
                offset_along = i * dp
                if offset_along >= line_length:
                    break
                center = start + dir_vec * offset_along
                for sgn in (-1.0, 1.0):
                    p = center + sgn * normal * dg
                    p_shifted, is_valid = self._shift_point_away_from_obstacles(
                        p, center, normal, all_obstacles, env.bounds
                    )
                    if is_valid:
                        X_list.append((float(p_shifted[0]), float(p_shifted[1])))
                        y_list.append(1 if sgn > 0 else -1)
            
            for i in range(n_start):
                offset_along = i * dp
                if offset_along >= line_length:
                    break
                center = goal - dir_vec * offset_along
                for sgn in (-1.0, 1.0):
                    p = center + sgn * normal * dg
                    p_shifted, is_valid = self._shift_point_away_from_obstacles(
                        p, center, normal, all_obstacles, env.bounds
                    )
                    if is_valid:
                        X_list.append((float(p_shifted[0]), float(p_shifted[1])))
                        y_list.append(1 if sgn > 0 else -1)
            
            for t, obs_poly, intersects in obstacle_zones:
                proj_point = start + t * dir_vec * line_length
                
                if intersects:
                    L_local_adj = L_local * 1.5
                    dg_adj = dg * 1.3
                    n_local_base = max(4, int(L_local_adj / dp))
                else:
                    L_local_adj = L_local
                    dg_adj = dg
                    n_local_base = max(2, int(L_local_adj / dp))
                
                max_points_per_cluster = int(np.ceil(L_local_adj / dp))
                n_local = min(max_points_per_cluster, n_local_base)
                
                cluster_points: list[tuple[np.ndarray, int]] = []
                
                for i in range(-n_local // 2, n_local // 2 + 1):
                    offset_along = i * dp
                    center = proj_point + dir_vec * offset_along
                    rel_center = center - start
                    t_center = np.dot(rel_center, dir_vec) / (line_length + 1e-12)
                    if t_center < 0.0 or t_center > 1.0:
                        continue
                    
                    for sgn in (-1.0, 1.0):
                        p = center + sgn * normal * dg_adj
                        p_shifted, is_valid = self._shift_point_away_from_obstacles(
                            p, center, normal, all_obstacles, env.bounds
                        )
                        if is_valid:
                            cluster_points.append((p_shifted, 1 if sgn > 0 else -1))
                
                if len(cluster_points) < 2:
                    L_expanded = L_local_adj * 1.5
                    n_expanded = max(2, min(max_points_per_cluster, int(L_expanded / dp)))
                    cluster_points = []
                    for i in range(-n_expanded // 2, n_expanded // 2 + 1):
                        offset_along = i * dp
                        center = proj_point + dir_vec * offset_along
                        rel_center = center - start
                        t_center = np.dot(rel_center, dir_vec) / (line_length + 1e-12)
                        if t_center < 0.0 or t_center > 1.0:
                            continue
                        
                        for sgn in (-1.0, 1.0):
                            p = center + sgn * normal * dg_adj
                            p_shifted, is_valid = self._shift_point_away_from_obstacles(
                                p, center, normal, all_obstacles, env.bounds
                            )
                            if is_valid:
                                cluster_points.append((p_shifted, 1 if sgn > 0 else -1))
                    
                    if len(cluster_points) < 2:
                        for edge_offset in [-L_local_adj * 0.5, L_local_adj * 0.5]:
                            center_edge = proj_point + dir_vec * edge_offset
                            rel_edge = center_edge - start
                            t_edge = np.dot(rel_edge, dir_vec) / (line_length + 1e-12)
                            if 0.0 <= t_edge <= 1.0:
                                for sgn in (-1.0, 1.0):
                                    p = center_edge + sgn * normal * dg_adj * 1.5
                                    p_shifted, is_valid = self._shift_point_away_from_obstacles(
                                        p, center_edge, normal, all_obstacles, env.bounds
                                    )
                                    if is_valid:
                                        cluster_points.append((p_shifted, 1 if sgn > 0 else -1))
                
                for p_shifted, label in cluster_points:
                    X_list.append((float(p_shifted[0]), float(p_shifted[1])))
                    y_list.append(label)

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)
        return Samples(X=X, y=y)
