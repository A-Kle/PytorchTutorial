import torch
from torch.autograd import Variable
import model as m

device = torch.device('cuda') #to run on gpu
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]], device=device)) #3x1 matrix
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]], device=device))

model = m.Model()
criterion = torch.nn.MSELoss(size_average=False) #computes squared error
optimizer = torch.optim.SGD(model.parameters(), lr=0.03) #parameters to be updated + learning rate

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

for epoch in range(500):
    y_pred = model(x_data)

    #computes loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad() #resets gradient from previous values
    loss.backward() #stochastic gradient descent computes new gradient for batch(not just one value)
    optimizer.step() #performs optimization step(updates the value for whole batch for our function)

hour_var = Variable(torch.Tensor([[4.0]]))
print("Prediction after training:", 4, model.forward(hour_var).item())