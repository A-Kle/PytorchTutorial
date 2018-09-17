import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) #3x1 matrix
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self): #constructor for initiating the nn.Linear module
        super(Model, self).__init__()
        self.Linear = torch.nn.Linear(1,1)