import numpy as np

class PlanningPhaseTiming:
    """Время выполнения различных фаз планирования"""
    def __init__(self):
        self.sampling_ms = 0.0
        self.svm_training_ms = 0.0
        self.path_extraction_ms = 0.0
        self.smoothing_ms = 0.0


class SamplingStatistics:
    """Статистика процесса выборки"""
    def __init__(self):
        self.total_samples = 0
        self.border_samples = 0
        self.guide_samples = 0


class SVMStatistics:
    """Статистика обучения SVM"""
    def __init__(self):
        self.num_support_vectors = 0
        self.support_vector_ratio = 0.0  # Процент опорных векторов
        self.C = 1.0
        self.gamma = 1.0


class PathQualityMetrics:
    """Метрики качества пути"""
    def __init__(self):
        self.path_length = 0.0
        self.straight_line_distance = 0.0
        self.path_efficiency = 0.0  # straight_line / path_length
        self.steps = 0
    
    def compute(self, path, start, goal):
        """Вычисляет метрики качества пути"""
        from .path_extractor import PathExtractor
        from .config import PlannerConfig
        
        extractor = PathExtractor(PlannerConfig())
        L = extractor.length(path)
        self.path_length = L
        
        # Прямая дистанция
        straight_dist = np.linalg.norm(np.array(goal) - np.array(start))
        self.straight_line_distance = straight_dist
        
        # Эффективность
        self.path_efficiency = straight_dist / L if L > 0 else 0.0
        
        self.steps = len(path)


class EnvironmentStatistics:
    """Статистика окружения"""
    def __init__(self):
        self.num_obstacles = 0
        self.obstacle_density = 0.0
        self.start_goal_distance = 0.0
    
    def compute(self, env):
        """Вычисляет статистику окружения"""
        self.num_obstacles = len(env.obstacles)
        
        # Вычисление площади препятствий (для прямоугольников)
        total_obstacle_area = 0.0
        for obs in env.obstacles:
            verts = obs.vertices
            if len(verts) >= 3:
                # Shoelace formula для площади многоугольника
                area = 0.0
                for i in range(len(verts)):
                    j = (i + 1) % len(verts)
                    area += verts[i][0] * verts[j][1]
                    area -= verts[j][0] * verts[i][1]
                total_obstacle_area += abs(area) / 2.0
        
        # Площадь сцены
        x_min, x_max, y_min, y_max = env.bounds
        scene_area = (x_max - x_min) * (y_max - y_min)
        
        # Плотность препятствий
        self.obstacle_density = total_obstacle_area / scene_area if scene_area > 0 else 0.0
        
        # Расстояние старт-цель
        start = np.array(env.start)
        goal = np.array(env.goal)
        self.start_goal_distance = float(np.linalg.norm(goal - start))


class Statistics:
    """Расширенный класс статистики для планирования пути"""
    
    def __init__(
        self,
        path_length,
        time_ms,
        steps,
        num_support_vectors,
        planning_timing=None,
        sampling_stats=None,
        svm_stats=None,
        path_quality=None,
        env_stats=None,
        in_corridor=False,
        collision_free=False,
        num_patterns=0,
    ):
        # Базовые метрики (для обратной совместимости)
        self.path_length = path_length
        self.time_ms = time_ms
        self.steps = steps
        self.num_support_vectors = num_support_vectors
        self.in_corridor = in_corridor
        self.collision_free = collision_free
        self.num_patterns = num_patterns
        
        # Расширенные метрики
        self.planning_timing = planning_timing or PlanningPhaseTiming()
        self.sampling_stats = sampling_stats or SamplingStatistics()
        self.svm_stats = svm_stats or SVMStatistics()
        self.path_quality = path_quality or PathQualityMetrics()
        self.env_stats = env_stats or EnvironmentStatistics()

    def to_dict(self):
        """Преобразует статистику в словарь для сохранения"""
        result = {
            # Базовые метрики
            "path_length": self.path_length,
            "time_ms": self.time_ms,
            "steps": self.steps,
            "num_support_vectors": self.num_support_vectors,
            "in_corridor": self.in_corridor,
            "collision_free": self.collision_free,
            
            # Время выполнения фаз
            "timing_sampling_ms": self.planning_timing.sampling_ms,
            "timing_svm_training_ms": self.planning_timing.svm_training_ms,
            "timing_path_extraction_ms": self.planning_timing.path_extraction_ms,
            "timing_smoothing_ms": self.planning_timing.smoothing_ms,
            
            # Статистика выборки
            "samples_total": self.sampling_stats.total_samples,
            "samples_border": self.sampling_stats.border_samples,
            "samples_guide": self.sampling_stats.guide_samples,
            
            # Статистика SVM
            "svm_support_vector_ratio": self.svm_stats.support_vector_ratio,
            "svm_C": self.svm_stats.C,
            "svm_gamma": self.svm_stats.gamma,
            
            # Качество пути
            "path_straight_distance": self.path_quality.straight_line_distance,
            "path_efficiency": self.path_quality.path_efficiency,
            
            # Статистика окружения
            "env_num_obstacles": self.env_stats.num_obstacles,
            "env_obstacle_density": self.env_stats.obstacle_density,
            "env_start_goal_distance": self.env_stats.start_goal_distance,
            
            # Процесс планирования
            "num_patterns": self.num_patterns,
        }
        return result

    def get_summary_text(self):
        """Возвращает текстовое резюме статистики для вывода в окне"""
        return Statistics._format_text(self)
    
    @staticmethod
    def _format_text(stats):
        """Форматирует текст статистики из объекта или словаря"""
        # Если это словарь
        if isinstance(stats, dict):
            d = stats
            path_length = d.get('path_length', 0)
            straight_distance = d.get('path_straight_distance', 0)
            efficiency = d.get('path_efficiency', 0)
            steps = d.get('steps', 0)
            time_ms = d.get('time_ms', 0)
            timing_sampling = d.get('timing_sampling_ms', 0)
            timing_svm = d.get('timing_svm_training_ms', 0)
            timing_path = d.get('timing_path_extraction_ms', 0)
            timing_smooth = d.get('timing_smoothing_ms', 0)
            num_sv = d.get('num_support_vectors', 0)
            sv_ratio = d.get('svm_support_vector_ratio', 0)
            svm_c = d.get('svm_C', 0)
            svm_gamma = d.get('svm_gamma', 0)
            samples_total = d.get('samples_total', 0)
            samples_border = d.get('samples_border', 0)
            samples_guide = d.get('samples_guide', 0)
            env_obstacles = d.get('env_num_obstacles', 0)
            env_density = d.get('env_obstacle_density', 0)
            env_distance = d.get('env_start_goal_distance', 0)
            num_patterns = d.get('num_patterns', 0)
        else:
            # Это объект Statistics
            path_length = stats.path_length
            straight_distance = stats.path_quality.straight_line_distance
            efficiency = stats.path_quality.path_efficiency
            steps = stats.steps
            time_ms = stats.time_ms
            timing_sampling = stats.planning_timing.sampling_ms
            timing_svm = stats.planning_timing.svm_training_ms
            timing_path = stats.planning_timing.path_extraction_ms
            timing_smooth = stats.planning_timing.smoothing_ms
            num_sv = stats.num_support_vectors
            sv_ratio = stats.svm_stats.support_vector_ratio
            svm_c = stats.svm_stats.C
            svm_gamma = stats.svm_stats.gamma
            samples_total = stats.sampling_stats.total_samples
            samples_border = stats.sampling_stats.border_samples
            samples_guide = stats.sampling_stats.guide_samples
            env_obstacles = stats.env_stats.num_obstacles
            env_density = stats.env_stats.obstacle_density
            env_distance = stats.env_stats.start_goal_distance
            num_patterns = stats.num_patterns
        
        lines = [
            "СТАТИСТИКА ПЛАНИРОВАНИЯ ПУТИ\n",
            "Качество пути",
            f"  - Длина пути: {path_length:.3f}",
            f"  - Прямая дистанция: {straight_distance:.3f}",
            f"  - Эффективность пути: {efficiency:.2%}",
            f"  - Количество шагов: {steps}",
            "",
            "Производительность",
            f"  - Общее время: {time_ms:.1f} мс",
            f"  - Выборка: {timing_sampling:.1f} мс",
            f"  - Обучение SVM: {timing_svm:.1f} мс",
            f"  - Извлечение пути: {timing_path:.1f} мс",
            f"  - Сглаживание: {timing_smooth:.1f} мс",
            f"  - Скорость планирования: {path_length / (time_ms / 1000.0):.2f} ед/с" if time_ms > 0 else "  - Скорость планирования: N/A",
            "",
            "SVM",
            f"  - Опорных векторов: {num_sv}",
            f"  - Процент опорных векторов: {sv_ratio:.1%}",
            f"  - Параметры: C={svm_c}, γ={svm_gamma}",
            "",
            "Выборка",
            f"  - Всего примеров: {samples_total}",
            f"  - Границы препятствий: {samples_border}",
            f"  - Guide samples: {samples_guide}",
            "",
            "Окружение",
            f"  - Препятствий: {env_obstacles}",
            f"  - Плотность препятствий: {env_density:.2%}",
            f"  - Расстояние старт-цель: {env_distance:.3f}",
            "",
            "Процесс планирования",
            f"  - Количество паттернов: {num_patterns}",
        ]
        return "\n".join(lines)
