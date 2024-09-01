import numpy as np

from layers import *

class TwoLayerNet():
    def __init__(self, layer_size):
        input_size, hidden_size, output_size = layer_size
        self.params = {
            'W1': np.random.randn(input_size, hidden_size),
            'b1': np.random.randn(hidden_size),
            'W2': np.random.randn(hidden_size, output_size),
            'b2': np.random.randn(output_size)
        }
        self.grad = {key:np.zeros_like(self.params[key].size) for key in self.params.keys()}

        self.layer_names = ['affine1', 'relu1', 'affine2']
        self.layers = {
            self.layer_names[0]: Affine(self.params['W1'], self.params['b1']),
            self.layer_names[1]: ReLU(),
            self.layer_names[2]: Affine(self.params['W2'], self.params['b2'])
        }

        self.loss_fn = MSE()
        self.loss = None

    def predict(self, x):
        for layer_name in self.layer_names:
            x = self.layers[layer_name].forward(x)
        return x

    def numerical_gradient(self, x, t):
        init_params = self.params.copy()
        h = 0.001

        for key in self.params.keys():
            for index, _ in np.ndenumerate(self.params[key]):
                self.params[key][index] += h
                y = self.predict(x)
                loss_pl = self.loss_fn.forward(y, t)
                self.params = init_params

                self.params[key][index] -= h
                y = self.predict(x)
                loss_mi = self.loss_fn.forward(y, t)
                self.params = init_params

                grad = (loss_pl-loss_mi)/(2*h)
                self.grad[key][index] = grad

    def gradient(self, x, t):
        y = self.predict(x)
        self.loss = self.loss_fn.forward(y, t)

        diff = 1
        diff = self.loss_fn.backward(diff)

        for layer_name in reversed(self.layer_names):
            diff = self.layers[layer_name].backward(diff)
        
        self.grad['W1'] = self.layers[self.layer_names[0]].dW
        self.grad['b1'] = self.layers[self.layer_names[0]].db
        self.grad['W2'] = self.layers[self.layer_names[2]].dW
        self.grad['b2'] = self.layers[self.layer_names[2]].db

def train(x, t, hidden_size=5, num_iter=10000):
    layer_size = (x.shape[1], hidden_size, t.shape[1])
    net = TwoLayerNet(layer_size)

    lr = 0.001

    for i in range(num_iter):
        net.gradient(x, t)

        for key in net.params.keys():
            net.params[key] -= lr*net.grad[key]
        
        if i%(num_iter//10)==0:
            print('iter', i, ':', net.loss)

if __name__=='__main__':
    input_size = 4
    output_size = 3

    x = np.random.randn(100, input_size)
    W = np.random.randn(input_size, output_size)
    b = np.random.randn(output_size)
    t = np.dot(x, W) + b

    train(x, t)
