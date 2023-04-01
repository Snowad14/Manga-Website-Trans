import os
import numpy as np
from abc import abstractmethod

from ..utils import InfererModule, ModelWrapper

class CommonInpainter(InfererModule):

    def inpaint(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        return self._inpaint(image, mask, inpainting_size, verbose)

    @abstractmethod
    def _inpaint(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass

class OfflineInpainter(CommonInpainter, ModelWrapper):
    _MODEL_SUB_DIR = 'inpainting'

    def _inpaint(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    @abstractmethod
    def _infer(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass
