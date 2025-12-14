from typing import Optional, List, Tuple
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QStatusBar, QLabel, QMessageBox
)
from PyQt5.QtCore import pyqtSlot

from ..config import PlannerConfig
from ..environment import Environment
from ..svm import KernelSVM
from ..sampling import Samples
from .dialogs import ParametersDialog, StatisticsDialog, EnvironmentDialog
from .planner_worker import PlannerWorker
from .plot_canvas import PlotCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = PlannerConfig()
        self.env: Optional[Environment] = None
        self.model: Optional[KernelSVM] = None
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

