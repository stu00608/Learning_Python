# Machine Learning : Pytorch

```python
import torch
```

## tensor v.s. numpy

> tensor 和 numpy 都是處理矩陣運算時好用的工具,但Pytorch內提供的tensor也可以替代掉numpy的角色.

```python
data = np.arange(6).reshape([2,3])
tensor = torch.from_numpy(data)

print(
    '\nnumpy\n',data,
    '\ntorch\n',tensor,
)
```

```
numpy
[[0 1 2]
 [3 4 5]] 
torch
 tensor([[0, 1, 2],
        [3, 4, 5]])
```

### 矩陣運算

* numpy當中使用np.matmul(),torch中使用torch.mm()

```python
data = [ [1,2],[3,4] ]
tensor = torch.FloatTensor(data) # 32-bit 浮點數型態之tensor

print(
    '\nnumpy\n',np.matmul(data,data),
    '\ntorch\n',torch.mm(tensor,tensor),
)
```

```
numpy
 [[ 7 10]
 [15 22]] 
torch
 tensor([[ 7., 10.],
        [15., 22.]])
```

---

* numpy和torch中都有dot()的函式,但兩者會有不同

```python
data = [ [1,2],[3,4] ]
tensor = torch.FloatTensor(data) # 32-bit 浮點數型態之tensor
data = np.array(data)

print(
    '\nnumpy\n',data.dot(data),
    '\ntorch\n',torch.mm(tensor,tensor),
)
```

```
numpy
 [[ 7 10]
 [15 22]] 
torch
 tensor([[ 7., 10.],
        [15., 22.]])
```

* tensor當中的dot()函式會將矩陣flat過後算內積,會是一個值

* torch的tensor當中也有許多數學運算和numpy相同,例如：mean(),sin()...

## Variable

---

```python
import numpy as np
import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad=True) #requires_grad會自動計算gradient(Backpropagation)

print(tensor)

print(variable)
```

```
tensor([[1., 2.],
        [3., 4.]])
tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
```

---

```python
t_out = torch.mean(tensor*tensor) # tensor不能反向傳播
v_out = torch.mean(variable*variable)
```

```
tensor(7.5000) 
 tensor(7.5000, grad_fn=<MeanBackward0>)
```

---

* 計算variable的backpropagation
* v_out = variable平方取平均值 = `1/4*sum(variable*variable)`
* 對v_out做微分得到`1/4*2*varialble`=`variable/2`

```python
v_out.backward()
print(variable.grad)
```

```
tensor([[0.5000, 1.0000],
        [1.5000, 2.0000]])
```

## Activation Function 激勵函數

* 將y=wx+b之類的線性函數做非線性轉換

---

```python
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)  # 在-5~5(包含)之間取得200個點
x = Variable(x)
x_np = x.data.numpy()   # matplotlib不接受torch形式,轉換成numpy

# popular activation functions
y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy() # there's no softplus in torch
# y_softmax = torch.softmax(x, dim=0).data.numpy() softmax is a special kind of activation function, it is about probability

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()

```

* `softmax()`函式使用在分類問題中的機率計算,不是用在本例的線性轉非線性的激勵函數,但還是屬於一種激勵函數.

---

## Regression & Classification

* 搭建第一個Neural Network

```python
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
```

```
Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
```

---

* 更新w參數.
* 建立Optimizer,常見的有準確率梯度下降法SGD(stochastic gradient descent),Momentum(類慣性),Adagrad,Adam(Momentum+Adagrad)
* 建立loss function,常見的有MSELoss(Mean Square Error),計算分類問題使用CrossEntropy.
* Iter代表迭代次數
* 建立訓練迴圈.

```python

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

```

---

* 分類問題

```python
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
```

---

### 快速搭建

```python
# method 1 
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
net1 = Net(2, 10, 2)
print(net1)

# method 2

net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)

print(net2)
```

```
Net(
  (hidden): Linear(in_features=2, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=2, bias=True)
)
Sequential(
  (0): Linear(in_features=2, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=2, bias=True)
)
```

* 在torch.nn內的ReLU類別,視為一個Neuron,和functional內的relu()是一樣的功能,但後者是純函數

---

## 分批訓練

* 運用DataLoader實現分批訓練.

```python
import torch
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # 每一個epoch都會打亂訓練的順序
    # num_workers=2, # 每次提取2個worker
)

def show_batch():
    for epoch in range(3):
        for step, (batch_x,batch_y) in enumerate(loader):
            # train your data...
                print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                    batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()
```

```

import torch...
Epoch:  0 | Step:  0 | batch x:  [10.  7.  4.  2.  9.  8.  5.  6.] | batch y:  [1. 4. 7. 9. 2. 3. 6. 5.]
Epoch:  0 | Step:  1 | batch x:  [3. 1.] | batch y:  [ 8. 10.]
Epoch:  1 | Step:  0 | batch x:  [ 5.  3. 10.  8.  6.  7.  9.  2.] | batch y:  [6. 8. 1. 3. 5. 4. 2. 9.]
Epoch:  1 | Step:  1 | batch x:  [4. 1.] | batch y:  [ 7. 10.]
Epoch:  2 | Step:  0 | batch x:  [ 6.  9.  7. 10.  1.  3.  5.  4.] | batch y:  [ 5.  2.  4.  1. 10.  8.  6.  7.]
Epoch:  2 | Step:  1 | batch x:  [2. 8.] | batch y:  [9. 3.]
```

* 可以看到我們將BATH_SIZE設為8,在Epoch 0的時候第一次提取了8個參數,但第二次因為不足8個,所以自動return剩下的參數來train.

---

```python

```

```

```

---

```python

```

```

```

---

```python

```

```

```

---

```python

```

```

```

---

```python

```

```

```
