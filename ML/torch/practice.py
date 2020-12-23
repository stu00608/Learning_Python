import numpy as np
import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad=True) #requires_grad會自動計算gradient(Backpropagation)

t_out = torch.mean(tensor*tensor) # tensor不能反向傳播
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()
print(variable)