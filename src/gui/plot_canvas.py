from typing import Optional, List, Tuple
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..environment import Environment
from ..svm import KernelSVM
from ..sampling import Samples
from ..corridor import CorridorChecker


class PlotCanvas(FigureCanvas):
    """Холст для визуализации сцены планирования"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, color="gray")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.fig.tight_layout()

    def clear(self):
        """Очищает холст"""
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, color="gray")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

    def plot_scene(
        self,
        env: Environment,
        model: Optional[KernelSVM],
        samples: Optional[Samples],
        path: Optional[List[Tuple[float, float]]],
    ):
        """Отрисовывает сцену планирования"""
        self.clear()

        # 1) Препятствия - светло-серое заполнение, один проход с одним элементом легенды
        first_obs = True
        for obs in env.obstacles:
            xs = [v[0] for v in obs.vertices] + [obs.vertices[0][0]]
            ys = [v[1] for v in obs.vertices] + [obs.vertices[0][1]]
            label = "Препятствия" if first_obs else None
            self.ax.fill(xs, ys, color="#c0c0c0", alpha=0.5, label=label)
            first_obs = False

        # 2) Выборки (классовые точки и guide)
        if samples is not None:
            pos_mask = samples.y == 1
            neg_mask = samples.y == -1
            pos_X = samples.X[pos_mask]
            neg_X = samples.X[neg_mask]

            # Эвристическое обнаружение guide возле линии старт->цель
            start = np.array(env.start)
            goal = np.array(env.goal)
            dir_vec = goal - start
            dist_sg = np.linalg.norm(dir_vec)
            if dist_sg > 1e-12:
                dir_vec = dir_vec / dist_sg
            else:
                dir_vec = np.array([1.0, 0.0])
            normal = np.array([-dir_vec[1], dir_vec[0]])

            guide_mask = np.zeros(len(samples.X), dtype=bool)
            for i, x in enumerate(samples.X):
                rel = x - start
                along = np.dot(rel, dir_vec)
                across = np.abs(np.dot(rel, normal))
                if -0.5 <= along <= dist_sg + 0.5 and across < 1.8:
                    guide_mask[i] = True

            # Построить классовые точки
            if len(pos_X) > 0:
                self.ax.scatter(pos_X[:, 0], pos_X[:, 1], c="r", marker=".", s=12, alpha=0.7, label="Класс +1")
            if len(neg_X) > 0:
                self.ax.scatter(neg_X[:, 0], neg_X[:, 1], c="b", marker=".", s=12, alpha=0.7, label="Класс -1")

            # Построить guide samples как маленькие желтые точки
            guide_X = samples.X[guide_mask]
            if len(guide_X) > 0:
                self.ax.scatter(guide_X[:, 0], guide_X[:, 1], c="y", s=8, alpha=0.5, label=None)

        # 3) Границы SVM f=+-1
        if model is not None:
            from ..config import PlannerConfig
            config = PlannerConfig()
            config.grid_resolution = 0.15
            checker = CorridorChecker(config)
            XY, xs, ys = checker.build_grid(env.bounds)
            f = model.decision(XY).reshape(ys.shape[0], xs.shape[0])
            cs_pos = self.ax.contour(xs, ys, f, levels=[1.0], colors=["r"], linestyles=["--"], linewidths=1.2, alpha=0.8)
            cs_neg = self.ax.contour(xs, ys, f, levels=[-1.0], colors=["b"], linestyles=["--"], linewidths=1.2, alpha=0.8)

        # 4) Траектория f=0 (путь) - толстая черная линия
        if path is not None and len(path) > 1:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            self.ax.plot(px, py, "-", color="k", linewidth=3.2, label="Траектория", zorder=10)

        # 5) Старт и Цель - большие маркеры
        self.ax.scatter([env.start[0]], [env.start[1]], c="g", marker="o", s=60, edgecolors="darkgreen", linewidths=1.2, label="Старт")
        self.ax.scatter([env.goal[0]], [env.goal[1]], c="r", marker="o", s=60, edgecolors="darkred", linewidths=1.2, label="Цель")

        self.ax.set_xlim(env.bounds[0], env.bounds[1])
        self.ax.set_ylim(env.bounds[2], env.bounds[3])

        # 6) Легенда - внизу справа, упрощенная и дедуплицированная
        handles, labels = self.ax.get_legend_handles_labels()
        seen = set()
        new_handles = []
        new_labels = []
        for h, l in zip(handles, labels):
            if l and l not in seen and l in {"Препятствия", "Класс +1", "Класс -1", "Траектория", "Старт", "Цель"}:
                seen.add(l)
                new_handles.append(h)
                new_labels.append(l)
        if new_handles:
            self.ax.legend(new_handles, new_labels, loc="lower right", fontsize=9, framealpha=0.9)

        self.draw()

