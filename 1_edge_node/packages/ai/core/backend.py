
from abc import ABC, abstractmethod
import numpy as np

class BaseBackend(ABC):
    """
    05082026 - KIET - Định nghĩa interface chung cho các backend chạy model AI
    """

    def __init__(self):
        """
        11082026 - KIET - Khởi tạo trạng thái mặc định của backend.
        """
        self.is_loaded = False

    @abstractmethod
    def load(self):
        """
        05082026 - KIET - Load model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray):
        """
        05082026 - KIET - run interface và trả raw output
        """
        raise NotImplementedError

    def close(self):
        """
        05082026 - KIET - Giai phong
        """
        self.is_loaded = False
