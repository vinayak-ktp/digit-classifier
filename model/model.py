import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
from utils.Layer import Dense, Dropout
from utils.Activation import ReLU
from utils.Loss import CategoricalCrossEntropy
# from utils.Optimizer import Adam

class Model:
    def __init__(self):
        self.dense1 = Dense(784, 128, weight_regularizer_l2=1e-4, bias_regularizer_l2=1e-4)
        self.activation1 = ReLU()
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(128, 64, weight_regularizer_l2=1e-4, bias_regularizer_l2=1e-4)
        self.activation2 = ReLU()
        self.dropout2 = Dropout(0.1)
        self.dense3 = Dense(64, 10, weight_regularizer_l2=1e-4, bias_regularizer_l2=1e-4)
        self.loss_activation = CategoricalCrossEntropy()

    def forward(self, X, y, type='train'):
        if type == 'train':
            self.dropout1.train()
            self.dropout2.train()
        else:
            self.dropout1.eval()
            self.dropout2.eval()
        
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)
        self.dropout1.forward(self.activation1.output)
        self.dense2.forward(self.dropout1.output)
        self.activation2.forward(self.dense2.output)
        self.dropout2.forward(self.activation2.output)
        self.dense3.forward(self.dropout2.output)
        self.loss_activation.forward(self.dense3.output, y)

        return self.loss_activation.output

    def backward(self, y_pred, y_true):
        self.loss_activation.backward(y_pred, y_true)
        self.dense3.backward(self.loss_activation.dinputs)
        self.dropout2.backward(self.dense3.dinputs)
        self.activation2.backward(self.dropout2.dinputs)
        self.dense2.backward(self.activation2.dinputs)
        self.dropout1.backward(self.dense2.dinputs)
        self.activation1.backward(self.dropout1.dinputs)
        self.dense1.backward(self.activation1.dinputs)