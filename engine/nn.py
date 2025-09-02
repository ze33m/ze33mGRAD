try:
    from Tensor import Tensor
except:
    from engine.Tensor import Tensor

import numpy as np


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        pass
    
    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, features_in:int, features_out:int):
        self.W = self.W = Tensor(np.random.randn(features_in, features_out) * np.sqrt(2/features_in))
        self.b = Tensor(np.zeros(features_out))

    def forward(self, X : Tensor):
        X = X if isinstance(X, Tensor) else Tensor(X)
        return X @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b]
    
class ReLu(Module):
    def forward(self, X: Tensor):
        X = X if isinstance(X, Tensor) else Tensor(X)
        return X.relu()
    

    
class L1(Module) :
    def forward(self, output : Tensor, target : Tensor) -> Tensor:
        output = output if isinstance(output, Tensor) else Tensor(output)
        target = target if isinstance(target, Tensor) else Tensor(target)
        return (output - target).abs().sum() * (1 / output.data.shape[0])  
    

class Leaky_ReLU(Module):
    def __init__(self, slope = 1e-2):
        self.slope = slope
    def forward(self, X: Tensor):
        X = X if isinstance(X, Tensor) else Tensor(X)
        return X.leaky_relu(1e-2)

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def stop_on(self, X, n:int):
        for layer in self.layers[:n]:
            X = layer(X)
        return X

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
    
class MSELoss(Module):
    def forward(self, output : Tensor, target : Tensor) -> Tensor:
        output = output if isinstance(output, Tensor) else Tensor(output)
        target = target if isinstance(target, Tensor) else Tensor(target)
        return (output - target).sum() * (output - target).sum() * (1 / output.data.shape[0])