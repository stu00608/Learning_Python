import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# y0表示x0的標籤,以此類推
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
x, y = Variable(x), Variable(y)


# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
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

# feature有x,y兩個特徵,輸出[0,1] or [1,0]分別代表不同分類
net = Net(2, 10, 2)
print(net)

plt.ion() # realtime print plot
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02) # parameters提供所有神經網路參數,lr-> learning rate
loss_func = torch.nn.CrossEntropyLoss() # CrossEntropyLoss用來計算分類問題的機率,[0.8,0.2]代表分在第一類的機率有80%

# Training

Iter = 200

for t in range(Iter):
    out = net(x) # [-2,-12,20] -> softmax(out) -> [0.1,0.2,0.7]
    loss = loss_func(out,y)

    optimizer.zero_grad() # 將所有參數的梯度歸0再做反向傳播
    loss.backward() # Backpropagation算出gradient
    optimizer.step() #以learning rate來更新weight,優化梯度

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()