import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
N = 200
A = np.array([[1,1],[1,1]])
X = np.random.uniform(-2, 2, size=(N, 2)) @ A # два признака
y = (X[:,0]**2 + 3*X[:,1] + 1 
     + 0.1*np.random.randn(N))             # таргет с шумом

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y)
ax.set_xlabel('X1')
ax.set_ylabel('X2') 
ax.set_zlabel('y')
plt.show()
print(np.linalg.cond(X))


