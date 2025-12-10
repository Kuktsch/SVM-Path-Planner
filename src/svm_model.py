import numpy as np
from .my_svm import KernelSVM

from .sampling import Samples
from .config import PlannerConfig

# СТАРАЯ ВЕРСИЯ (закомментировано): поддержка scikit-learn SVC
# import warnings
# from sklearn.svm import SVC
# from sklearn.exceptions import ConvergenceWarning
#
# class KernelSVMWrapper:
#     """Обертка для scikit-learn SVC, чтобы он работал как KernelSVM"""
#     def __init__(self, svc: SVC):
#         self.svc = svc
#         self.C = svc.C
#         self.gamma = svc.gamma
#         self.alphas = None  # Не используется для sklearn
#     
#     def decision_function(self, X):
#         return self.svc.decision_function(X)
#     
#     def predict(self, X):
#         return self.svc.predict(X)


class SVMModel:
    def __init__(self, svm: KernelSVM):
        self.svm = svm

    @staticmethod
    def train(samples: Samples, config: PlannerConfig) -> "SVMModel":
        # НОВАЯ ВЕРСИЯ: используем только кастомный KernelSVM
        svm = KernelSVM(C=config.C, gamma=config.gamma)
        svm.fit(samples.X, samples.y)
        return SVMModel(svm)
        
        # СТАРАЯ ВЕРСИЯ (закомментировано): поддержка выбора между sklearn и кастомным SVM
        # @staticmethod
        # def train(samples: Samples, config: PlannerConfig, use_sklearn: bool = False) -> "SVMModel":
        #     if use_sklearn:
        #         # Используем оптимизированный scikit-learn (значительно быстрее)
        #         with warnings.catch_warnings():
        #             warnings.filterwarnings("ignore", category=ConvergenceWarning)
        #             svm = SVC(C=config.C, kernel="rbf", gamma=config.gamma, max_iter=5000, tol=1e-3)
        #             svm.fit(samples.X, samples.y)
        #         wrapper = KernelSVMWrapper(svm)
        #         return SVMModel(wrapper)
        #     else:
        #         # Используем кастомный KernelSVM
        #         svm = KernelSVM(C=config.C, gamma=config.gamma)
        #         svm.fit(samples.X, samples.y)
        #         return SVMModel(svm)

    def decision(self, XY: np.ndarray) -> np.ndarray:
        # Прокси знакового расстояния: decision_function
        return self.svm.decision_function(XY)

    def params_summary(self) -> dict:
        # НОВАЯ ВЕРСИЯ: только для кастомного KernelSVM
        if hasattr(self.svm, 'alphas') and self.svm.alphas is not None:
            num_sv = int(np.sum(self.svm.alphas > 1e-6))
        else:
            num_sv = 0
        
        # СТАРАЯ ВЕРСИЯ (закомментировано): поддержка scikit-learn
        # if hasattr(self.svm, 'alphas') and self.svm.alphas is not None:
        #     num_sv = int(np.sum(self.svm.alphas > 1e-6))
        # elif hasattr(self.svm, 'svc'):
        #     # Для scikit-learn
        #     num_sv = len(self.svm.svc.support_vectors_)
        # else:
        #     num_sv = 0
        
        return {
            "C": self.svm.C,
            "gamma": self.svm.gamma,
            "num_support_vectors": num_sv,
        }

    def predict(self, XY: np.ndarray) -> np.ndarray:
        return self.svm.predict(XY)

    def boundary_mask(self, XY: np.ndarray, margin: float = 1.0) -> np.ndarray:
        f = self.decision(XY)
        return np.abs(f) < margin
