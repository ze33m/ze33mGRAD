import numpy as np
class Tensor():
    def __init__(self, data: np.ndarray, children = (), _op='', label = ''):
        data = np.array(data)
        self._op = _op 
        self.shape = data.shape
        self.data = data 
        self.label = label
        self._prev = children
        self.grad = np.zeros(self.shape)
        self._backward = lambda : None


    def __matmul__(self, other):
        out = Tensor(data=np.matmul(self.data,other.data), children=(self,other), _op='matmul')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        out = Tensor(data=self.data + other.data, children=(self,other), _op='add')

        def _backward():
            grad_self = out.grad
            while grad_self.ndim > self.data.ndim:
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)
            self.grad += grad_self

            grad_other = out.grad
            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)
            other.grad += grad_other
                
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, value):
        assert isinstance(value, (int,float))
        out = Tensor(data=self.data * value, children = tuple([self]), _op=f'*{value}')
        def _backward():
            self.grad += out.grad * value
        out._backward = _backward

        return out 
    
    def __rmul__(self,value):
        return self * value

    def relu(self):
        out = Tensor(self.data * (self.data > 0).astype(int), children = tuple([self]), _op='relu')

        def _backward():
            self.grad += (self.data > 0).astype(int) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        assert self.data.shape == ()
        topo = []
        visited = set()
        def build_topo(node):
            visited.add(node)
            for child in node._prev:
                build_topo(child)
            topo.append(node)
            
        build_topo(self)
        self.grad=1.
        for node in reversed(topo):
            node._backward()

    def sum(self):
        out = Tensor(self.data.sum(), children = tuple([self]), _op="sum")

        def _backward():
            self.grad += np.ones(self.grad.shape) * out.grad

        out._backward = _backward
        return out

    def __repr__(self):
        if len(self.shape) == 2:
            s = ''
            for row in self.data:
                s = s + str(row) + '\n'
            return s
        else:
            return str(self.data)