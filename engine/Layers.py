from Tensor import Tensor
import numpy as np
from graph import draw_dot

class Linear:
    def __init__(self, features_in:int, features_out:int):
        self.W = Tensor(np.random.random((features_in,features_out)))
        self.b = Tensor(np.random.random(features_out))
    def __call__(self, X : Tensor):
        return X @ self.W + self.b
    
class ReLu:
    def __call__(self, X: Tensor):
        return X.relu()