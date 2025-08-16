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
        assert self.data.shape == other.data.shape
        out = Tensor(data=self.data + other.data, children=(self,other), _op='add')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad 
        
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