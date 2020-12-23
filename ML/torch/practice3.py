import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# y = x^2
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # torch只能處理二維,用unsqueeze將一維變二維
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # 第一層Linear神經網路
        self.predict = torch.nn.Linear(n_hidden, n_output) # output layer

    # 正向傳遞的過程
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# feature數量, hidden layer數量, output數量
net = Net(1, 10, 1)
print(net)

plt.ion() # realtime print plot
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5) # parameters提供所有神經網路參數,lr-> learning rate
loss_func = torch.nn.MSELoss() # Regression使用Mean Square Error來計算loss

# Training

Iter = 200

for t in range(Iter):
    prediction = net(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad() # 將所有參數的梯度歸0再做反向傳播
    loss.backward() # Backpropagation算出gradient
    optimizer.step() #以learning rate來更新weight,優化梯度

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

