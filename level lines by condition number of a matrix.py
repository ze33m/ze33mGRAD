import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import engine.nn as nn
from engine.Tensor import Tensor
from engine.Optims import GradientDescent

A = np.random.uniform(-2, 2, size=(200, 2)) 
Y = (A[:,0] + 3*A[:,1])
Y=Y.reshape(200,1)
w0 = np.linspace(-10, 10, 100)
w1 = np.linspace(-10, 10, 100)
xgrid, ygrid = np.meshgrid(w0, w1)

def compute_zgrid(A):
    zgrid = np.zeros_like(xgrid)
    for i in range(xgrid.shape[0]):
        for j in range(xgrid.shape[1]):
            W = np.array([xgrid[i, j], ygrid[i, j]])
            zgrid[i, j] = np.sum(np.abs(Y - A @ W))
    return zgrid


fig, ax = plt.subplots()
CS = ax.contour(xgrid, ygrid, compute_zgrid(A), 40)
ax.clabel(CS, inline=True, fontsize=8)
plt.xlabel("W0")
plt.ylabel("W1")
plt.title("Линии уровня")

class NN(nn.Module):
    def __init__(self, ):
        super().__init__()
        hidden1 = nn.Linear(2,2)
        out = nn.Linear(2,1)
        relu = nn.ReLu()
        self.seq = nn.Sequential(out)

    def parameters(self):
        return self.seq.parameters()
    
    def forward(self, X):
        return self.seq(X)


model = NN()

trajectory = []

epochs = 1000
L1 = nn.L1()
print(np.linalg.cond(A))
optim = GradientDescent(model.parameters(), lr=0.01)
for epoch in range(epochs):
    pred = model(A)
    loss = L1(pred, Y)
    loss.zero_grad()
    loss.backward()
    optim.step()
    w = model.parameters()[0].data.copy()
    trajectory.append(w)

trajectory = np.array(trajectory)

print(trajectory[0])
print(trajectory[-1])

def animate(frame):
    ax.clear()
    
    CS = ax.contour(xgrid, ygrid, compute_zgrid(A), 40)
    ax.clabel(CS, inline=True, fontsize=8)
    
    if frame > 0:
        ax.plot(trajectory[:frame, 0], trajectory[:frame, 1], 'r-', alpha=0.7, linewidth=2)

    if frame < len(trajectory):
        ax.plot(trajectory[frame, 0], trajectory[frame, 1], 'ro', markersize=8)
    
    ax.set_xlabel("W0")
    ax.set_ylabel("W1")
    ax.set_title(f"Градиентный спуск - Эпоха {frame}")

anim = FuncAnimation(fig, animate, frames=len(trajectory), interval=1, repeat=True)

plt.show()