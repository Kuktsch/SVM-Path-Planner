import numpy as np

class KernelSVM:
    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None

    # RBF ядро 
    def kernel(self, x1, x2):
        # x1: (N, D), x2: (M, D)
        # returns kernel matrix NxM
        dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * dist)

    # Решатель SMO
    def fit(self, X, y, max_iter=500):
        y = y.astype(float)
        y[y == 0] = -1

        N, D = X.shape
        self.X = X
        self.y = y
        self.alphas = np.zeros(N)
        self.b = 0.0

        K = self.kernel(X, X)
        
        # Кэш для предсказаний f_i (изначально все alphas = 0, поэтому f_cache = b = 0)
        f_cache = np.zeros(N)
        
        # for i in range(N):
        #     f_cache[i] = np.sum(self.alphas * y * K[:, i]) + self.b

        # SMO обучение с оптимизациями
        num_changed = 0
        examine_all = True
        
        for iteration in range(max_iter):
            num_changed = 0
            if examine_all:
                for i in range(N):
                    num_changed += self._examine_example(i, K, y, f_cache)
            else:
                for i in range(N):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i, K, y, f_cache)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            if num_changed == 0 and not examine_all:
                break
        
        sv_mask = (self.alphas > 1e-6) & (self.alphas < self.C - 1e-6)
        if np.any(sv_mask):
            sv_indices = np.where(sv_mask)[0]
            b_values = []
            for i in sv_indices:
                f_i = np.sum(self.alphas * self.y * K[:, i])
                b_i = self.y[i] - f_i
                b_values.append(b_i)
            if len(b_values) > 0:
                self.b = np.mean(b_values)

        return self
    
    def _examine_example(self, i2, K, y, f_cache):
        """Оптимизированная проверка примера i2"""
        y2 = y[i2]
        alpha2 = self.alphas[i2]
        E2 = f_cache[i2] - y2
        
        r2 = E2 * y2
        
        # Проверяем условие KKT
        if (r2 < -1e-3 and alpha2 < self.C) or (r2 > 1e-3 and alpha2 > 0):
            # Сначала пробуем примеры с alphas на границах
            indices = np.where((self.alphas > 1e-6) & (self.alphas < self.C - 1e-6))[0]
            if len(indices) > 1:
                # Выбираем пример с максимальной ошибкой
                E1 = f_cache[indices] - y[indices]
                i1 = indices[np.argmax(np.abs(E1 - E2))]
            else:
                # Если нет примеров на границах, выбираем случайный
                i1 = np.random.randint(0, len(y))
                while i1 == i2:
                    i1 = np.random.randint(0, len(y))
            
            if self._take_step(i1, i2, K, y, f_cache):
                return 1
        
        return 0
    
    def _take_step(self, i1, i2, K, y, f_cache):
        """Оптимизация пары alphas"""
        if i1 == i2:
            return False
        
        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]
        y1 = y[i1]
        y2 = y[i2]
        E1 = f_cache[i1] - y1
        E2 = f_cache[i2] - y2
        s = y1 * y2
        
        # Границы
        if y1 != y2:
            L = max(0.0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0.0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        
        if L == H:
            return False
        
        # eta
        eta = 2.0 * K[i1, i2] - K[i1, i1] - K[i2, i2]
        if eta >= 0:
            return False
        
        # Новые значения alphas
        a2 = alpha2 - y2 * (E1 - E2) / eta
        a2 = np.clip(a2, L, H)
        
        if abs(a2 - alpha2) < 1e-5:
            return False
        
        a1 = alpha1 + s * (alpha2 - a2)
        
        # Обновляем смещение b
        b1 = self.b - E1 - y1 * (a1 - alpha1) * K[i1, i1] - y2 * (a2 - alpha2) * K[i1, i2]
        b2 = self.b - E2 - y1 * (a1 - alpha1) * K[i1, i2] - y2 * (a2 - alpha2) * K[i2, i2]
        
        if 0 < a1 < self.C:
            self.b = b1
        elif 0 < a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0
        
        # Обновляем alphas
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        
        # Обновляем кэш предсказаний
        delta1 = (a1 - alpha1) * y1
        delta2 = (a2 - alpha2) * y2
        f_cache += delta1 * K[:, i1] + delta2 * K[:, i2]
        
        return True

    def decision_function(self, Xtest):
        K = self.kernel(Xtest, self.X)
        return (K * (self.alphas * self.y)).sum(axis=1) + self.b

    def predict(self, Xtest):
        return np.sign(self.decision_function(Xtest)).astype(int)

    @property
    def support_vectors(self):
        mask = self.alphas > 1e-6
        return self.X[mask]