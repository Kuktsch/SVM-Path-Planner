from typing import Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QFormLayout, QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ..config import PlannerConfig
from ..environment import Environment
from ..geometry import Point


class ParametersDialog(QDialog):
    def __init__(self, config: PlannerConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Параметры алгоритма")
        self.setMinimumWidth(350)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        self.c_edit = QLineEdit(str(self.config.C))
        self.gamma_edit = QLineEdit(str(self.config.gamma))
        self.ds_edit = QLineEdit(str(self.config.step_ds))
        self.theta_edit = QLineEdit(str(self.config.sector_angle_deg))
        self.max_steps_edit = QLineEdit(str(self.config.max_steps))
        self.pattern_trials_edit = QLineEdit(str(self.config.pattern_trials))
        
        form.addRow("C (SVM):", self.c_edit)
        form.addRow("γ (gamma):", self.gamma_edit)
        form.addRow("Шаг (ds):", self.ds_edit)
        form.addRow("Угол сектора (θs, град):", self.theta_edit)
        form.addRow("Макс. шагов:", self.max_steps_edit)
        form.addRow("Количество паттернов:", self.pattern_trials_edit)
        
        layout.addLayout(form)
        
        buttons = QHBoxLayout()
        self.ok_btn = QPushButton("Применить")
        self.ok_btn.clicked.connect(self.accept)
        buttons.addWidget(self.ok_btn)
        layout.addLayout(buttons)
        
        self.setLayout(layout)

    def get_config(self) -> Optional[PlannerConfig]:
        try:
            config = PlannerConfig()
            config.C = float(self.c_edit.text())
            config.gamma = float(self.gamma_edit.text())
            config.step_ds = float(self.ds_edit.text())
            config.sector_angle_deg = float(self.theta_edit.text())
            config.max_steps = int(self.max_steps_edit.text())
            config.pattern_trials = int(self.pattern_trials_edit.text())
            return config
        except ValueError:
            return None


class StatisticsDialog(QDialog):
    def __init__(self, stats_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Статистика")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        self.stats_data = stats_data
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        info = QTextEdit()
        info.setReadOnly(True)
        info.setFont(QFont("Segoe UI", 9))
        
        # Используем единый метод форматирования
        from ..stats import Statistics
        if hasattr(self.stats_data, 'get_summary_text'):
            text = self.stats_data.get_summary_text()
        else:
            # Передаем словарь в статический метод
            text = Statistics._format_text(self.stats_data)
        
        info.setPlainText(text)
        layout.addWidget(info)
        
        buttons = QHBoxLayout()
        self.save_btn = QPushButton("Сохранить")
        self.close_btn = QPushButton("Закрыть")
        self.save_btn.clicked.connect(self.save_report)
        self.close_btn.clicked.connect(self.accept)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)
        
        self.setLayout(layout)

    def save_report(self):
        import csv
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            # Если stats_data это объект Statistics, используем to_dict()
            if hasattr(self.stats_data, 'to_dict'):
                data_to_save = self.stats_data.to_dict()
            else:
                data_to_save = self.stats_data
            
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data_to_save.keys()))
                writer.writeheader()
                writer.writerow(data_to_save)
            self.accept()


class EnvironmentDialog(QDialog):
    def __init__(self, config: PlannerConfig, current_env: Optional[Environment] = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Генерация поля")
        self.setMinimumWidth(400)
        self.current_env = current_env
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Start point
        start_layout = QHBoxLayout()
        self.start_x_edit = QLineEdit()
        self.start_y_edit = QLineEdit()
        if self.current_env:
            self.start_x_edit.setText(str(self.current_env.start[0]))
            self.start_y_edit.setText(str(self.current_env.start[1]))
        else:
            self.start_x_edit.setText(str(self.config.x_min + 2.0))
            self.start_y_edit.setText(str(self.config.y_min + 2.0))
        start_layout.addWidget(self.start_x_edit)
        start_layout.addWidget(QLabel(" , "))
        start_layout.addWidget(self.start_y_edit)
        form.addRow("Начальная точка (x, y):", start_layout)
        
        # Goal point
        goal_layout = QHBoxLayout()
        self.goal_x_edit = QLineEdit()
        self.goal_y_edit = QLineEdit()
        if self.current_env:
            self.goal_x_edit.setText(str(self.current_env.goal[0]))
            self.goal_y_edit.setText(str(self.current_env.goal[1]))
        else:
            self.goal_x_edit.setText(str(self.config.x_max - 2.0))
            self.goal_y_edit.setText(str(self.config.y_max - 2.0))
        goal_layout.addWidget(self.goal_x_edit)
        goal_layout.addWidget(QLabel(" , "))
        goal_layout.addWidget(self.goal_y_edit)
        form.addRow("Конечная точка (x, y):", goal_layout)
        
        # Number of obstacles
        self.num_obs_edit = QLineEdit()
        if self.current_env:
            self.num_obs_edit.setText(str(len(self.current_env.obstacles)))
        else:
            self.num_obs_edit.setText(str(self.config.num_obstacles))
        form.addRow("Количество препятствий:", self.num_obs_edit)
        
        layout.addLayout(form)

        
        buttons = QHBoxLayout()
        self.generate_btn = QPushButton("Применить")
        self.generate_btn.clicked.connect(self.accept)
        buttons.addWidget(self.generate_btn)
        layout.addLayout(buttons)
        
        self.setLayout(layout)

    def get_environment(self) -> Optional[Environment]:
        try:
            start_x = float(self.start_x_edit.text())
            start_y = float(self.start_y_edit.text())
            goal_x = float(self.goal_x_edit.text())
            goal_y = float(self.goal_y_edit.text())
            num_obs = int(self.num_obs_edit.text())
            
            start: Point = (start_x, start_y)
            goal: Point = (goal_x, goal_y)
            
            # Проверка что точки в пределах границ
            if not (self.config.x_min <= start_x <= self.config.x_max and 
                    self.config.y_min <= start_y <= self.config.y_max):
                return None
            if not (self.config.x_min <= goal_x <= self.config.x_max and 
                    self.config.y_min <= goal_y <= self.config.y_max):
                return None
            
            if num_obs < 0:
                return None
            
            # Импортируем Environment здесь чтобы избежать циклического импорта
            return Environment.create(start, goal, num_obs, self.config, seed=None)
        except ValueError:
            return None

