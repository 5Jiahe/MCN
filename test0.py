# import os, sys
# import numpy as np
# import cv2
import torch

a = torch.randn((312,640))
print(a.shape)
a = torch.pinverse(a)
print(a.shape)
