import torch
import torch.nn as nn
import os
from torchFewShot.models.conv4 import word_encoder1,conv4_image,map,cpn
from torchFewShot.models.propagation import Propagation
from torchFewShot.models.resnet12 import resnet12
from thop import profile
# model1 = word_encoder1(312,640)
# input1 = torch.randn(5, 312) 
# macs1, params1 = profile(model1, inputs=(input1, 0))


model2 = word_encoder1(312,640)
input2 = torch.randn(5, 312) 
macs2, params2 = profile(model2, inputs=(input2, 0))

model4 = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5)
input5 = torch.randn(80, 3,84,84) 
macs4, params4 = profile(model4, inputs=(input5, ))

# print(macs1,params1)
# model2 = conv4_image()
# input2 = torch.randn(80, 3,84,84) 
# macs2, params2 = profile(model2, inputs=(input2, ))
# print(macs2,params2)

model3 = Propagation(0,640)
input3 = torch.randn(75, 5,640) 
input4 = torch.randn(75, 1,640)
macs3, params3 = profile(model3, inputs=(input3,input3,input4 ))

model6 = cpn()
input6 = torch.randn(75, 5,640) 
macs6, params6 = profile(model6, inputs=(input3,))

print('ours:',(macs2+macs4+macs3)*2/(1000**3),(params2+params3+params4)/(1000**2))
print('cpn:',(macs3+macs4+macs6)*2/(1000**3),(params3+params4+params6)/(1000**2))

model7 = conv4_image()
input7 = torch.randn(80, 3,84,84) 
macs7, params7 = profile(model7, inputs=(input7, ))
print(macs7,params7/(1000**2))
