import numpy as np

class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, diff):
        dx = np.dot(diff, self.W.T)
        self.dW = np.dot(self.x.T, diff)
        self.db = np.mean(diff, axis=0)
        return dx

class ReLU():
    def __init__(self):
        self.mask = None

    def forward(self, x):
        out = x.copy()
        self.mask = out<=0
        out[self.mask] = 0
        return out

    def backward(self, diff):
        dx = diff
        dx[self.mask] = 0
        return dx

class MSE():
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y = y
        self.t = t
        out = np.sum((y-t)**2)/(2*y.shape[0])
        return out

    def backward(self, diff):
        dx = diff*(self.y-self.t)/(2*self.y.shape[0])
        return dx
