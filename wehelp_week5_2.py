import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

class NeuralNetwork:
    def __init__(self, weights, biases, learning_rate=0.1):
        self.weights = [np.array(w) for w in weights]
        self.biases  = [np.array(b) for b in biases]
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0)

    def forward(self, inputs):
        self.x_input = np.array(inputs)
        
                                                                                            # 隱藏層
        self.z1 = np.dot(self.weights[0], self.x_input) + self.biases[0]                    # shape (2,1)
        self.a1 = self.relu(self.z1)                                                        # shape (2,1)
        
                                                                                            # 輸出層
        self.z2 = np.dot(self.weights[1], self.a1) + self.biases[1]                         # shape (1,1)
        self.a2 = sigmoid(self.z2)                                                          # shape (1,1)
        
        return self.a2

    def bce_loss(self, predicted, expected):
        y = np.array(expected)
        return -(y * np.log(predicted) + (1 - y) * np.log(1 - predicted))

    def backward(self, expected):

        y = np.array(expected)
        a2 = self.a2                                                                        # 輸出層結果, shape (1,1)
        
        
        delta2 = (a2 - y)                                                                   # 輸出層 delta (梯度簡化為 a2 - y)     
        self.dW2 = np.dot(delta2, self.a1.T)                                                # 輸出層權重與偏差的梯度
        self.db2 = delta2                                                                   # W2: shape (1,2), a1: shape (2,1)
        
        
        d_hidden = np.dot(self.weights[1].T, delta2) * self.relu_derivative(self.z1)        # 隱藏層的梯度
        
        self.dW1 = np.dot(d_hidden, self.x_input.T)                                         # 隱藏層權重與偏差的梯度
        self.db1 = d_hidden                                                                 # W1: shape (2,2), x_input: shape (2,1)                                                                

    def update_weights(self):
    
        self.weights[1] -= self.learning_rate * self.dW2
        self.biases[1]  -= self.learning_rate * self.db2

        self.weights[0] -= self.learning_rate * self.dW1
        self.biases[0]  -= self.learning_rate * self.db1

# 第一層: W1 shape (2,2), b1 shape (2,1)
# 第二層: W2 shape (1,2), b2 shape (1,1)

weights = [
    [[0.5, 0.2],
     [0.6, -0.6]],
    [[0.8, 0.4]]
]
biases = [
    [[0.3],
     [0.25]],
    [[-0.5]]
]

nn = NeuralNetwork(weights, biases, learning_rate=0.1)

inputs = [[0.75],
          [1.25]]                                                                             # shape (2,1)
expected_output = [[1]]                                                                       # shape (1,1)

initial_predicted = nn.forward(inputs)
initial_loss = nn.bce_loss(initial_predicted, expected_output)
print("Task 1: Training 1 time")
print("O1,O2:", initial_predicted.tolist())
print("Total loss:", initial_loss)

for times in range(1000):
    nn.forward(inputs)
    nn.backward(expected_output)
    nn.update_weights()

final_predicted = nn.forward(inputs)
final_loss = nn.bce_loss(final_predicted, expected_output)

print("\nTask 2: Training 1000 times")
print("Final O1,O2:", final_predicted.tolist())
print("Final total loss:", final_loss)
