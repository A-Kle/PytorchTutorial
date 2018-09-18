import torch
from torch.autograd import Variable
import model as m

#logistic regression, binary prediction

device = torch.device('cuda')

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]], device=device))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]], device=device))

model = m.Model()

criterion = torch.nn.BCELoss(reduction='elementwise_mean') #loss for binary values
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #parameters to be updated + learning rate

for epoch in range(1000):
    y_pred = model(x_data)

    #computes loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad() #resets gradient from previous values
    loss.backward() #stochastic gradient descent computes new gradient for batch(not just one value)
    optimizer.step() #performs optimization step(updates the value for whole batch for our function)

hour_var = Variable(torch.Tensor([[1.0]]))
print("Prediction after training:", 1, "hour", model.forward(hour_var).item(), model.forward(hour_var).item() > 0.5)

hour_var = Variable(torch.Tensor([[7.0]]))
print("Prediction after training:", 7, "hours", model.forward(hour_var).item(), model.forward(hour_var).item() > 0.5)