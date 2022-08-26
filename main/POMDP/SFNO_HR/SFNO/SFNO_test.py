import torch
import torch.nn as nn
from model import FNO2d, Heat_forward

'''

'''

batch_size = 20
x = 10
y = 10
n = 9
mode1 = 5
mode2 = 5
width = 10
res = 4*(x-1)+1
model = FNO2d(mode1, mode2,  width,res,n)
data  = torch.zeros(batch_size,x,y,7)
one = model(data)
