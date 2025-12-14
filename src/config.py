class PlannerConfig:
    def __init__(
        self,
        # Границы сцены
        x_min: float = -10.0,
        x_max: float = 10.0,
        y_min: float = -10.0,
        y_max: float = 10.0,
        # Прямоугольник транспортного средства (половины размеров)
        half_length: float = 0.5,
        half_width: float = 0.25,
        # Выборка
        border_samples_per_edge: int = 40,
        guide_samples_count: int = 60,
        virtual_margin: float = 1.0,  # Увеличено для более широкого коридора
        guide_sample_distance: float = 1.2,  # dg: расстояние от номинальной линии
        guide_sample_spacing: float = 0.4,   # dp: расстояние между guide samples
        guide_samples_near_points: int = 3,   # количество пар guide samples возле старта/цели
        obstacle_proximity_threshold: float = 2.5,  # L_local: локальная длина для кластеров препятствий
        max_shift: float = 2.0,  # максимальное расстояние сдвига для избежания коллизий
        shift_eps: float = 0.1,  # эпсилон для расчета глубины проникновения
        # SVM
        C: float = 100.0,  # Увеличено для лучшего разделения
        gamma: float = 0.5,  # Настроено для лучшей обобщающей способности и более плавных границ
        # Сетка / Поиск
        grid_resolution: float = 0.15,
        sector_angle_deg: float = 80.0,  # Более широкий сектор поиска
        step_ds: float = 0.15,  # Размер шага
        max_steps: int = 400,  # Больше шагов для более длинных путей
        # Сглаживание
        spline_s: float = 0.2,
        # Случайная генерация
        num_obstacles: int = 4,
        obstacle_min_size: float = 1.0,
        obstacle_max_size: float = 2.2,
        # Поиск паттернов для классификации препятствий
        pattern_trials: int = 5,  # Количество различных паттернов маркировки препятствий для попытки
        # Оптимизация SVM
        use_sklearn_svm: bool = True,  # Использовать оптимизированный scikit-learn вместо кастомного (быстрее)
        # Ввод/вывод
        output_dir: str = "outputs"
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.half_length = half_length
        self.half_width = half_width
        self.border_samples_per_edge = border_samples_per_edge
        self.guide_samples_count = guide_samples_count
        self.virtual_margin = virtual_margin
        self.guide_sample_distance = guide_sample_distance
        self.guide_sample_spacing = guide_sample_spacing
        self.guide_samples_near_points = guide_samples_near_points
        self.obstacle_proximity_threshold = obstacle_proximity_threshold
        self.max_shift = max_shift
        self.shift_eps = shift_eps
        self.C = C
        self.gamma = gamma
        self.grid_resolution = grid_resolution
        self.sector_angle_deg = sector_angle_deg
        self.step_ds = step_ds
        self.max_steps = max_steps
        self.spline_s = spline_s
        self.num_obstacles = num_obstacles
        self.obstacle_min_size = obstacle_min_size
        self.obstacle_max_size = obstacle_max_size
        self.pattern_trials = pattern_trials
        self.use_sklearn_svm = use_sklearn_svm
        self.output_dir = output_dir
