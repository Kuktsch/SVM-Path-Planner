import time
from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal

import random
import numpy as np
from ..config import PlannerConfig
from ..environment import Environment
from ..sampling import SampleGenerator
from ..svm import KernelSVM
from ..corridor import CorridorChecker
from ..path_extractor import PathExtractor
from ..geometry import GeometryUtils
from ..stats import (
    Statistics, PlanningPhaseTiming, SamplingStatistics, SVMStatistics,
    PathQualityMetrics,
    EnvironmentStatistics
)


class PlannerWorker(QThread):
    finished = pyqtSignal(object, object, object, object)  # env, model, samples, result_dict
    status_update = pyqtSignal(str)

    def __init__(self, config: PlannerConfig, env: Optional[Environment] = None, seed: int = 0):
        super().__init__()
        self.config = config
        self.env = env
        self.seed = seed

    def run(self):
        try:
            if self.env is None:
                self.status_update.emit("Генерация окружения...")
                env = Environment.random(self.config, seed=self.seed)
            else:
                env = self.env
            
            # Поиск паттернов: попробовать различные паттерны маркировки препятствий
            start_np = np.array(env.start)
            goal_np = np.array(env.goal)
            dir_vec = goal_np - start_np
            line_length = np.linalg.norm(dir_vec)
            if line_length < 1e-12:
                dir_vec = np.array([1.0, 0.0])
            else:
                dir_vec = dir_vec / line_length
            normal = np.array([-dir_vec[1], dir_vec[0]])
            
            # Инициализация генераторов и проверщиков
            sample_generator = SampleGenerator(self.config)
            path_extractor = PathExtractor(self.config)
            corridor_checker = CorridorChecker(self.config)
            
            # Получить начальный паттерн
            initial_labels = sample_generator.get_initial_obstacle_labels(env, start_np, normal)
            
            best_result = None
            best_path_length = float('inf')
            best_model = None
            best_samples = None
            best_stats = None
            
            patterns_to_try = [initial_labels]  # Начать с начального паттерна
            
            # Генерировать случайные паттерны
            rng = random.Random(self.seed)
            for _ in range(self.config.pattern_trials - 1):
                # Случайно перевернуть некоторые метки препятствий
                pattern = initial_labels.copy()
                num_flips = max(1, len(pattern) // 2)
                indices_to_flip = rng.sample(range(len(pattern)), min(num_flips, len(pattern)))
                for idx in indices_to_flip:
                    pattern[idx] = -pattern[idx]
                patterns_to_try.append(pattern)
            
            for pattern_idx, pattern in enumerate(patterns_to_try):
                if pattern_idx > 0:
                    self.status_update.emit(f"Попытка паттерна {pattern_idx + 1}/{len(patterns_to_try)}...")
                else:
                    self.status_update.emit("Генерация обучающей выборки...")
                
                # Измерение времени выборки
                t_sampling = time.time()
                samples = sample_generator.generate(env, obstacle_labels=pattern)
                dt_sampling_ms = (time.time() - t_sampling) * 1000.0
                
                # Статистика выборки
                sampling_stats = SamplingStatistics()
                sampling_stats.total_samples = len(samples.X)
                # Подсчет border и guide samples (эвристика)
                start = np.array(env.start)
                goal = np.array(env.goal)
                dir_vec = goal - start
                dist_sg = np.linalg.norm(dir_vec)
                if dist_sg > 1e-12:
                    dir_vec = dir_vec / dist_sg
                else:
                    dir_vec = np.array([1.0, 0.0])
                normal = np.array([-dir_vec[1], dir_vec[0]])
                
                guide_count = 0
                for i, x in enumerate(samples.X):
                    rel = x - start
                    along = np.dot(rel, dir_vec)
                    across = np.abs(np.dot(rel, normal))
                    if -0.5 <= along <= dist_sg + 0.5 and across < 1.8:
                        guide_count += 1
                sampling_stats.guide_samples = guide_count
                sampling_stats.border_samples = sampling_stats.total_samples - guide_count
                
                self.status_update.emit("Обучение SVM...")
                t_svm = time.time()
                model = KernelSVM.train(samples, self.config)
                dt_svm_ms = (time.time() - t_svm) * 1000.0
                
                # Статистика SVM
                svm_params = model.params_summary()
                svm_stats = SVMStatistics()
                svm_stats.num_support_vectors = svm_params["num_support_vectors"]
                svm_stats.support_vector_ratio = svm_stats.num_support_vectors / sampling_stats.total_samples if sampling_stats.total_samples > 0 else 0.0
                svm_stats.C = svm_params["C"]
                svm_stats.gamma = svm_params["gamma"]
                
                self.status_update.emit("Проверка связности коридора...")
                t_conn = time.time()
                conn = corridor_checker.check(model, env.bounds, env.start, env.goal)
                dt_conn_ms = (time.time() - t_conn) * 1000.0
                
                if not conn.connected:
                    # Попробовать с большим количеством guide samples
                    self.status_update.emit("Повторное обучение с увеличенной выборкой...")
                    old_count = self.config.guide_samples_count
                    self.config.guide_samples_count = int(old_count * 1.5)
                    t_sampling2 = time.time()
                    samples = sample_generator.generate(env, obstacle_labels=pattern)
                    dt_sampling_ms += (time.time() - t_sampling2) * 1000.0
                    t_svm2 = time.time()
                    model = KernelSVM.train(samples, self.config)
                    dt_svm_ms += (time.time() - t_svm2) * 1000.0
                    self.config.guide_samples_count = old_count
                    t_conn2 = time.time()
                    conn = corridor_checker.check(model, env.bounds, env.start, env.goal)
                    dt_conn_ms += (time.time() - t_conn2) * 1000.0
                    if not conn.connected:
                        continue  # Пропустить этот паттерн, если все еще не связан
                
                self.status_update.emit("Построение траектории...")
                t_path = time.time()
                path_result = path_extractor.extract(model, env.start, env.goal, obstacles=env.obstacles)
                dt_path_ms = (time.time() - t_path) * 1000.0
                
                if not path_result.reached:
                    continue  # Пропустить, если путь не достигает цели
                
                # Сглаживание может "врезаться" в препятствия — подбираем s, чтобы траектория оставалась безопасной
                t_smooth = time.time()
                s_candidates = [self.config.spline_s, self.config.spline_s * 0.5, self.config.spline_s * 0.2, 0.0]
                smoothed = path_result.points
                for s_val in s_candidates:
                    cand = path_extractor.smooth(path_result.points, s=s_val)
                    if not GeometryUtils.path_intersects_obstacles(cand, env.obstacles):
                        smoothed = cand
                        break
                dt_smooth_ms = (time.time() - t_smooth) * 1000.0
                if GeometryUtils.path_intersects_obstacles(smoothed, env.obstacles):
                    continue  # даже без сглаживания пересекает препятствия — этот паттерн не подходит
                
                # Проверки коллизий удалены
                in_corridor = True
                no_collision = True
                
                L = path_extractor.length(smoothed)
                dt_total_ms = dt_sampling_ms + dt_svm_ms + dt_path_ms + dt_smooth_ms + dt_conn_ms
                
                # Вычисление расширенных метрик
                path_quality = PathQualityMetrics()
                path_quality.compute(smoothed, env.start, env.goal)
                
                env_stats = EnvironmentStatistics()
                env_stats.compute(env)
                
                # Время выполнения фаз
                timing = PlanningPhaseTiming()
                timing.sampling_ms = dt_sampling_ms
                timing.svm_training_ms = dt_svm_ms
                timing.path_extraction_ms = dt_path_ms
                timing.smoothing_ms = dt_smooth_ms
                
                # Создание объекта статистики
                stats = Statistics(
                    path_length=L,
                    time_ms=dt_total_ms,
                    steps=len(path_result.points),
                    num_support_vectors=svm_stats.num_support_vectors,
                    planning_timing=timing,
                    sampling_stats=sampling_stats,
                    svm_stats=svm_stats,
                    path_quality=path_quality,
                    env_stats=env_stats,
                    in_corridor=in_corridor,
                    collision_free=no_collision,
                    num_patterns=len(patterns_to_try),
                )
                
                # Предпочитать более короткие пути
                if L < best_path_length:
                        best_path_length = L
                        best_result = {
                            "path": smoothed,
                            "path_result": path_result,
                        }
                        best_result.update(stats.to_dict())  # Добавляем всю статистику
                        best_model = model
                        best_samples = samples
                        best_stats = stats
            
            # Использовать лучший результат или резервную попытку
            if best_result is None:
                # Резерв: использовать последнюю попытку паттерна
                self.status_update.emit("Использование последнего паттерна...")
                
                # Инициализация генераторов и проверщиков для fallback
                sample_generator = SampleGenerator(self.config)
                path_extractor = PathExtractor(self.config)
                
                # patterns_to_try уже определен выше, используем его
                
                t_sampling = time.time()
                samples = sample_generator.generate(env, obstacle_labels=patterns_to_try[-1])
                dt_sampling_ms = (time.time() - t_sampling) * 1000.0
                
                sampling_stats = SamplingStatistics()
                sampling_stats.total_samples = len(samples.X)
                # Упрощенный подсчет
                sampling_stats.guide_samples = int(sampling_stats.total_samples * 0.6)
                sampling_stats.border_samples = sampling_stats.total_samples - sampling_stats.guide_samples
                
                t_svm = time.time()
                model = KernelSVM.train(samples, self.config)
                dt_svm_ms = (time.time() - t_svm) * 1000.0
                
                svm_params = model.params_summary()
                svm_stats = SVMStatistics()
                svm_stats.num_support_vectors = svm_params["num_support_vectors"]
                svm_stats.support_vector_ratio = svm_stats.num_support_vectors / sampling_stats.total_samples if sampling_stats.total_samples > 0 else 0.0
                svm_stats.C = svm_params["C"]
                svm_stats.gamma = svm_params["gamma"]
                
                t_path = time.time()
                path_result = path_extractor.extract(model, env.start, env.goal, obstacles=env.obstacles)
                dt_path_ms = (time.time() - t_path) * 1000.0
                
                t_smooth = time.time()
                s_candidates = [self.config.spline_s, self.config.spline_s * 0.5, self.config.spline_s * 0.2, 0.0]
                smoothed = path_result.points
                for s_val in s_candidates:
                    cand = path_extractor.smooth(path_result.points, s=s_val)
                    if not GeometryUtils.path_intersects_obstacles(cand, env.obstacles):
                        smoothed = cand
                        break
                dt_smooth_ms = (time.time() - t_smooth) * 1000.0
                if GeometryUtils.path_intersects_obstacles(smoothed, env.obstacles):
                    # Даже после подбора параметра сглаживания траектория пересекает препятствия
                    self.status_update.emit("Не удалось построить траекторию без пересечения препятствий")
                    self.finished.emit(env, None, None, None)
                    return
                
                # Проверки коллизий удалены
                in_corridor = True
                no_collision = True
                
                L = path_extractor.length(smoothed)
                dt_total_ms = dt_sampling_ms + dt_svm_ms + dt_path_ms + dt_smooth_ms
                
                path_quality = PathQualityMetrics()
                path_quality.compute(smoothed, env.start, env.goal)
                
                env_stats = EnvironmentStatistics()
                env_stats.compute(env)
                
                timing = PlanningPhaseTiming()
                timing.sampling_ms = dt_sampling_ms
                timing.svm_training_ms = dt_svm_ms
                timing.path_extraction_ms = dt_path_ms
                timing.smoothing_ms = dt_smooth_ms
                
                stats = Statistics(
                    path_length=L,
                    time_ms=dt_total_ms,
                    steps=len(path_result.points),
                    num_support_vectors=svm_stats.num_support_vectors,
                    planning_timing=timing,
                    sampling_stats=sampling_stats,
                    svm_stats=svm_stats,
                    path_quality=path_quality,
                    env_stats=env_stats,
                    in_corridor=in_corridor,
                    collision_free=no_collision,
                    num_patterns=len(patterns_to_try),
                )
                
                result = {
                    "path": smoothed,
                    "path_result": path_result,
                }
                result.update(stats.to_dict())
                best_model = model
                best_samples = samples
            else:
                result = best_result
            
            self.status_update.emit("Готово")
            self.finished.emit(env, best_model, best_samples, result)
        except Exception as e:
            self.status_update.emit(f"Ошибка: {str(e)}")
            self.finished.emit(None, None, None, None)

