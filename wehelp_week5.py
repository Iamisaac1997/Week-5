import numpy as np

class NeuralNetwork:
    def __init__(self, weights, biases, learning_rate=0.01):
        self.weights = [np.array(w) for w in weights]                                 
        self.biases  = [np.array(b) for b in biases]
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0)

    def forward(self, inputs):

        activation = np.array(inputs)
        self.activations = [activation]                                                     # 保存每層激活值（含輸入層）
        self.z_values = []                                                                  # 保存各層線性組合結果

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activation) + b
            self.z_values.append(z)

            if i == 0:                                                                      # 只有第一層 (i == 0) 用 ReLU，其餘層用線性
                activation = self.relu(z)
            else:
                activation = z

            self.activations.append(activation)

        return activation

    def mse_loss(self, predicted, expected):
        """
        MSE 損失:
          L = mean((predicted - expected)^2)
        """
        predicted = np.array(predicted)
        expected = np.array(expected)
        return np.mean((predicted - expected)**2)

    def backward(self, expected):

        expected = np.array(expected)
        n = expected.size                                                                   # 輸出層元素數
        L = len(self.weights)                                                               # 總層數(不含輸入層)

        # 建立梯度儲存
        self.gradients = [np.zeros_like(w) for w in self.weights]
        self.bias_gradients = [np.zeros_like(b) for b in self.biases]

        # -- 第 2 層(輸出層, index = L-1) 為 linear
        delta = (2 / n) * (self.activations[-1] - expected)
        self.gradients[-1] = np.dot(delta, self.activations[-2].T)
        self.bias_gradients[-1] = delta

        # -- 第 1 層(第二隱藏層, index = L-2) 為 linear
        #   delta = (W_out^T * delta_out)
        delta = np.dot(self.weights[-1].T, delta)  # 不乘 ReLU'
        self.gradients[-2] = np.dot(delta, self.activations[-3].T)
        self.bias_gradients[-2] = delta

        # -- 第 0 層(第一隱藏層, index = L-3) 為 ReLU 
        #   delta = (W_lin^T * delta_lin) * ReLU'(z^(1))
        delta = np.dot(self.weights[-2].T, delta) * self.relu_derivative(self.z_values[0])
        self.gradients[-3] = np.dot(delta, self.activations[0].T)
        self.bias_gradients[-3] = delta

    def update_weights(self):
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients[i]
            self.biases[i]  -= self.learning_rate * self.bias_gradients[i]

weights = [
    [[0.5, 0.2], [0.6, -0.6]],                                                              # 第一隱藏層shape (2,2)
    [[0.8, -0.5]],                                                                          # 第二隱藏層shape (1,2)
    [[0.6], [-0.3]]                                                                         # 輸出層shape (2,1)
]

biases = [
    [[0.3], [0.25]],                                                                        # 第一隱藏層shape (2,1)
    [[0.6]],                                                                                # 第二隱藏層shape (1,1)
    [[0.4], [0.75]]                                                                         # 輸出層shape   (2,1)
]

nn = NeuralNetwork(weights, biases, learning_rate=0.01)

inputs = [[1.5], [0.5]]                                                                     # shape: (2,1)
expected_output = [[0.8], [1]]                                                              # shape: (2,1)


init_pred = nn.forward(inputs)
init_loss = nn.mse_loss(init_pred, expected_output)
print("Task 1: Training 1 time")
print("O1,O2:", init_pred.flatten())
print("Total loss:", init_loss)

for times in range(1000):
    nn.forward(inputs)
    nn.backward(expected_output)
    nn.update_weights()

final_pred = nn.forward(inputs)
final_loss = nn.mse_loss(final_pred, expected_output)

print("\nTraining 1000 times")
print("Final O1,O2:", final_pred.flatten())
print("Final total loss:", final_loss)
