import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

#linear regression
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

sample_size = 3 #samples in dataset

w = Variable(torch.Tensor([1.0]), requires_grad=True)

#model
def forward(x):
    return x * w

#loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

""" gradient is calculated automatically with pytorch
def gradient(x, y):
    return 2 * x * (x*w - y)
"""

print("Predict before training", 4, forward(4))
print("\n")
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward() #computes the gradient for each value
        print("\tx:{} y:{} grad:{}".format(x_val, y_val, w.grad.data[0]))
        w.data = w.data - 0.01 * w.grad.data

        #zero gradient after updating weights
        w.grad.data.zero_()
        
    print("progress:", epoch, "w=", w.data[0], "loss=", l.data[0])

print("Predict after training", 4, forward(4).data[0])