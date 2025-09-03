try:
    from Tensor import Tensor
except:
    from engine.Tensor import Tensor

import numpy as np


class GradientDescent:
    def __init__(self, params: list[Tensor], lr=1e-3):
        self.params = params
        self.lr = lr

    def step(self):
        for weight in self.params:
            weight.data -= self.lr * weight.grad

class MomentumGD:
    def __init__(self, params: list[Tensor], y, n):
        self.params = params
        self.y = y
        self.n = n
        self.velocities = [0 for i in range(len(params))]

    def step(self):
        for i, weight in enumerate(self.params):
            self.velocities[i] = self.y * self.velocities[i] + self.n * weight.grad
            weight.data -= self.velocities[i]
