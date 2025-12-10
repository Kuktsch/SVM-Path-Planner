import numpy as np
import matplotlib.pyplot as plt

from .environment import Environment
from .svm_model import SVMModel
from .corridor import CorridorChecker


class SceneVisualizer:
    """Визуализатор сцены планирования"""
    
    def __init__(self, config=None):
        self.config = config
    
    def plot(self, env: Environment, model: SVMModel | None, path: list[tuple[float, float]] | None, save_path: str | None = None, samples: np.ndarray | None = None, labels: np.ndarray | None = None):
        """Строит график сцены"""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

        first_obs = True
        for obs in env.obstacles:
            xs = [v[0] for v in obs.polygon.vertices] + [obs.polygon.vertices[0][0]]
            ys = [v[1] for v in obs.polygon.vertices] + [obs.polygon.vertices[0][1]]
            label = "Obstacles" if first_obs else None
            ax.fill(xs, ys, color="#c0c0c0", alpha=0.5, label=label)
            first_obs = False

        if samples is not None and labels is not None:
            pos = labels == 1
            neg = labels == -1
            if np.any(pos):
                ax.scatter(samples[pos, 0], samples[pos, 1], c="r", marker=".", s=12, alpha=0.7, label="Class +1")
            if np.any(neg):
                ax.scatter(samples[neg, 0], samples[neg, 1], c="b", marker=".", s=12, alpha=0.7, label="Class -1")

        if model is not None:
            from .config import PlannerConfig
            if self.config is None:
                self.config = PlannerConfig()
            checker = CorridorChecker(self.config)
            XY, xs, ys = checker.build_grid(env.bounds)
            f = model.decision(XY).reshape(ys.shape[0], xs.shape[0])
            ax.contour(xs, ys, f, levels=[1.0], colors=["r"], linestyles=["--"], linewidths=1.2, alpha=0.8)
            ax.contour(xs, ys, f, levels=[-1.0], colors=["b"], linestyles=["--"], linewidths=1.2, alpha=0.8)
            ax.contour(xs, ys, f, levels=[0.0], colors=["k"], linestyles=["-"], linewidths=3.0, alpha=0.95)

        if path is not None and len(path) > 1:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, "-", color="k", linewidth=3.2, label="Trajectory")

        ax.scatter([env.start[0]], [env.start[1]], c="g", marker="o", s=60, edgecolors="darkgreen", linewidths=1.2, label="Start")
        ax.scatter([env.goal[0]], [env.goal[1]], c="r", marker="o", s=60, edgecolors="darkred", linewidths=1.2, label="Goal")

        ax.set_xlim(env.bounds[0], env.bounds[1])
        ax.set_ylim(env.bounds[2], env.bounds[3])

        handles, labels_list = ax.get_legend_handles_labels()
        seen = set()
        new_h, new_l = [], []
        wanted = {"Obstacles", "Class +1", "Class -1", "Trajectory", "Start", "Goal"}
        for h, l in zip(handles, labels_list):
            if l and l in wanted and l not in seen:
                seen.add(l)
                new_h.append(h)
                new_l.append(l)
        if new_h:
            ax.legend(new_h, new_l, loc="lower right", fontsize=9, framealpha=0.9)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
