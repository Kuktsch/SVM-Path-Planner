import random

from .geometry import Polygon, Point
from .config import PlannerConfig


class Obstacle:
    def __init__(self, polygon: Polygon):
        self.polygon = polygon


class Environment:
    def __init__(self, start: Point, goal: Point, obstacles: list[Obstacle], bounds: tuple[float, float, float, float]):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.bounds = bounds  # x_min, x_max, y_min, y_max

    @staticmethod
    def random(config: PlannerConfig, seed: int | None = None) -> "Environment":
        rng = random.Random(seed)
        x_min, x_max, y_min, y_max = config.x_min, config.x_max, config.y_min, config.y_max

        # Старт/цель вдали от границ
        start = (rng.uniform(x_min + 1.0, x_min + 3.0), rng.uniform(y_min + 1.0, y_max - 1.0))
        goal = (rng.uniform(x_max - 3.0, x_max - 1.0), rng.uniform(y_min + 1.0, y_max - 1.0))

        obstacles = Environment.generate_obstacles(config, start, goal, seed=seed)
        return Environment(start=start, goal=goal, obstacles=obstacles, bounds=(x_min, x_max, y_min, y_max))

    @staticmethod
    def generate_obstacles(config: PlannerConfig, start: Point, goal: Point, seed: int | None = None) -> list[Obstacle]:
        rng = random.Random(seed)
        # Фиксированная область генерации препятствий: от -6 до 6
        obstacle_x_min, obstacle_x_max = -6.0, 6.0
        obstacle_y_min, obstacle_y_max = -6.0, 6.0
        
        obstacles: list[Obstacle] = []
        attempts = 0
        max_attempts = config.num_obstacles * 20
        
        while len(obstacles) < config.num_obstacles and attempts < max_attempts:
            cx = rng.uniform(obstacle_x_min, obstacle_x_max)
            cy = rng.uniform(obstacle_y_min, obstacle_y_max)
            sx = rng.uniform(config.obstacle_min_size, config.obstacle_max_size) * 0.5
            sy = rng.uniform(config.obstacle_min_size, config.obstacle_max_size) * 0.5
            
            poly = Polygon([
                (cx - sx, cy - sy),
                (cx + sx, cy - sy),
                (cx + sx, cy + sy),
                (cx - sx, cy + sy),
            ])
            
            # Проверка что препятствие не содержит start/goal
            if not poly.contains_point(start) and not poly.contains_point(goal):
                obstacles.append(Obstacle(poly))
            
            attempts += 1
        
        return obstacles

    @staticmethod
    def create(start: Point, goal: Point, num_obstacles: int, config: PlannerConfig, seed: int | None = None) -> "Environment":
        """Создает окружение с заданными start/goal и количеством препятствий"""
        # Временно обновляем количество препятствий в config для этой генерации
        old_num = config.num_obstacles
        config.num_obstacles = num_obstacles
        obstacles = Environment.generate_obstacles(config, start, goal, seed=seed)
        config.num_obstacles = old_num
        return Environment(start=start, goal=goal, obstacles=obstacles, bounds=(config.x_min, config.x_max, config.y_min, config.y_max))
