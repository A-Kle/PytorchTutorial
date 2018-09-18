import torch
from torch.autograd import Variable
import model as m

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) #3x1 matrix
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

model = m.Model()
criterion = torch.nn.MSELoss(size_average=False) #computes squared error
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #parameters to be updated + learning rate

for epoch in range(500):
    y_pred = model(x_data)

    #computes loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    optimizer.zero_grad() #resets gradient from previous values
    loss.backward() #stochastic gradient descent computes new gradient for batch(not just one value)
    optimizer.step() #performs optimization step(updates the value for whole batch for our function)

hour_var = Variable(torch.Tensor([[4.0]]))
print("Prediction after training:", 4, model.forward(hour_var).data[0][0])