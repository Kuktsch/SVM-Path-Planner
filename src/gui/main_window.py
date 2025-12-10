from typing import Optional, List, Tuple
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QStatusBar, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ..config import PlannerConfig
from ..environment import Environment
from ..svm_model import SVMModel

from ..sampling import Samples
from ..corridor import CorridorChecker
from .dialogs import ParametersDialog, StatisticsDialog, EnvironmentDialog
from .planner_worker import PlannerWorker


class PlotCanvas(FigureCanvas):
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
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, color="gray")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

    def plot_scene(
        self,
        env: Environment,
        model: Optional[SVMModel],
        samples: Optional[Samples],
        path: Optional[List[Tuple[float, float]]],
    ):
        self.clear()

        # 1) Препятствия - светло-серое заполнение, один проход с одним элементом легенды
        first_obs = True
        for obs in env.obstacles:
            xs = [v[0] for v in obs.polygon.vertices] + [obs.polygon.vertices[0][0]]
            ys = [v[1] for v in obs.polygon.vertices] + [obs.polygon.vertices[0][1]]
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = PlannerConfig()
        self.env: Optional[Environment] = None
        self.model: Optional[SVMModel] = None
        self.samples: Optional[Samples] = None
        self.path: Optional[List[Tuple[float, float]]] = None
        self.stats: Optional[dict] = None
        self.worker: Optional[PlannerWorker] = None
        
        self.setWindowTitle("Планирование траектории движения обхода препятствий с помощью метода опорных векторов")
        self.setMinimumSize(900, 700)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout()
        central.setLayout(layout)
        
        # Строка состояния
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Готово")
        self.status_bar.addWidget(self.status_label)
        self.setStatusBar(self.status_bar)
        
        # Холст для построения графиков
        self.canvas = PlotCanvas(self)
        layout.addWidget(self.canvas)
        
        # Нижняя панель
        bottom_panel = QHBoxLayout()
        bottom_panel.setContentsMargins(10, 10, 10, 10)
        
        self.generate_btn = QPushButton("Генерация поля")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.clicked.connect(self.generate_environment)
        
        self.params_btn = QPushButton("Параметры")
        self.params_btn.setMinimumHeight(40)
        self.params_btn.clicked.connect(self.show_parameters)
        
        self.run_btn = QPushButton("Запуск алгоритма")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self.run_planner)
        self.run_btn.setEnabled(False)
        
        self.stats_btn = QPushButton("Статистика")
        self.stats_btn.setMinimumHeight(40)
        self.stats_btn.clicked.connect(self.show_statistics)
        self.stats_btn.setEnabled(False)
        
        bottom_panel.addWidget(self.generate_btn)
        bottom_panel.addWidget(self.params_btn)
        bottom_panel.addWidget(self.run_btn)
        bottom_panel.addWidget(self.stats_btn)
        bottom_panel.addStretch()
        
        layout.addLayout(bottom_panel)

    @pyqtSlot()
    def generate_environment(self):
        dialog = EnvironmentDialog(self.config, self.env, self)
        if dialog.exec_():
            new_env = dialog.get_environment()
            if new_env:
                self.env = new_env
                # Очищаем результаты предыдущего запуска
                self.model = None
                self.samples = None
                self.path = None
                self.stats = None
                self.stats_btn.setEnabled(False)
                
                # Отображаем только карту без траектории
                self.canvas.plot_scene(self.env, None, None, None)
                self.run_btn.setEnabled(True)
                self.status_label.setText("Поле сгенерировано. Готово к построению траектории")
            else:
                QMessageBox.warning(self, "Ошибка", "Неверные значения координат или параметров")

    @pyqtSlot()
    def show_parameters(self):
        dialog = ParametersDialog(self.config, self)
        if dialog.exec_():
            new_config = dialog.get_config()
            if new_config:
                self.config = new_config
                self.status_label.setText("Параметры обновлены")
            else:
                QMessageBox.warning(self, "Ошибка", "Неверные значения параметров")

    @pyqtSlot()
    def run_planner(self):
        if self.env is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала сгенерируйте поле!")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Информация", "Алгоритм уже выполняется")
            return
        
        self.run_btn.setEnabled(False)
        self.params_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.status_label.setText("Запуск алгоритма...")
        
        import random
        seed = random.randint(0, 10000)
        
        # Используем сохраненное окружение
        self.worker = PlannerWorker(self.config, env=self.env, seed=seed)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_planner_finished)
        self.worker.start()

    @pyqtSlot(object, object, object, object)
    def on_planner_finished(self, env, model, samples, result):
        self.run_btn.setEnabled(True)
        self.params_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        
        if result is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось построить траекторию")
            self.status_label.setText("Ошибка")
            return
        
        # env должен быть тот же что был сохранен, но на всякий случай обновим
        if env is not None:
            self.env = env
        self.model = model
        self.samples = samples
        self.path = result["path"]
        self.stats = result
        self.stats_btn.setEnabled(True)
        
        self.canvas.plot_scene(self.env, model, samples, self.path)
        self.status_label.setText("Траектория построена")

    @pyqtSlot()
    def show_statistics(self):
        if self.stats is None:
            QMessageBox.warning(self, "Предупреждение", "Статистика недоступна")
            return
        
        dialog = StatisticsDialog(self.stats, self)
        dialog.exec_()

