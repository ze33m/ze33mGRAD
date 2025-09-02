try:
    from Tensor import Tensor
except:
    from engine.Tensor import Tensor

import numpy as np
from typing import List

class GradientDescent:
    def __init__(self, params: List[Tensor], lr=1e-3):
        self.params = params
        self.lr = lr

    def step(self):
        for weight in self.params:
            weight.data -= self.lr * weight.grad
        