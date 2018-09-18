import torch

class Model(torch.nn.Module):
    def __init__(self): #constructor for initiating the nn.Linear module
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1) #input size is only one x value(feature), only one y output

    def forward(self, x): #Variable of input
        y_pred = self.linear(x)
        return y_pred #output as Variable