from __future__ import print_function
from __future__ import division

import os
import sys
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
sys.path.append('./torchFewShot')

from args_xent import argument_parser
from torchFewShot.models.trainer import ModelTrainer
from torchFewShot.models.propagation import Propagation
from torchFewShot.models.resnet12 import resnet12
from torchFewShot.models.conv4 import word_encoder1,word_encoder2,conv4_image,cross_attention,cross_attention1
from torchFewShot.data_manager import DataManager
from torchFewShot.utils.logger import Logger

from torch.utils.tensorboard import SummaryWriter 

import numpy as np
import pandas as pd
# import matplotlib.pyplot.plt
import seaborn as sns
import matplotlib
sns.set(context='notebook',style='whitegrid')
# 设置风格尺度和显示中文

import warnings
warnings.filterwarnings('ignore')  
from scipy.stats import norm #使用直方图和最大似然高斯分布拟合绘制分布

def main(word_mask_rate,image_mask_rate):
    parser = argument_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    args.save_dir = os.path.join(args.save_dir, args.dataset, args.backbone, '%dway_%dshot'%(args.nKnovel, args.nExemplars), args.model_name)

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    # print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        # print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
    # else:
        # print("Currently using CPU (GPU is highly recommended)")
    
    args.log_dir = osp.join(args.save_dir, 'log')

    writer = SummaryWriter(log_dir=args.log_dir)

    # print('Initializing image data manager')
    args.phase = 'test'
    args.epoch_size = 5000
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    data_loader = {'train': trainloader,
                   'test': testloader
                   }
    word_dim = 0
    if args.dataset == 'cub':
        word_dim = 312
    elif args.dataset == 'sun':
        word_dim = 102
    elif args.dataset == 'flower':
        word_dim = 1024

    if args.backbone == 'conv4':
            module_word =word_encoder1(word_dim,64)
            module_image = conv4_image()
            channel_dim = 64
    elif args.backbone == 'resnet12':
            module_word=word_encoder2(word_dim,640)
            module_image = resnet12()
            channel_dim = 640

    module_trans = Propagation(args,  channel_dim)
    cross_atten = cross_attention(channel_dim,2)
    cross_atten1 = cross_attention1(channel_dim,2)
    # create trainer
    tester = ModelTrainer(args=args,
                          module_word=module_word,
                          module_image=module_image,
                          module_trans = module_trans,
                          data_loader=data_loader,
                          writer=writer,
                          word_mask_rate = word_mask_rate,image_mask_rate = image_mask_rate,cross_atten=cross_atten,cross_atten1=cross_atten1,word_dim = word_dim,channel_dim=channel_dim)

    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    tester.module_word.load_state_dict(checkpoint['module_word_state_dict'])
    tester.module_image.load_state_dict(checkpoint['module_image_state_dict'])
    tester.module_trans.load_state_dict(checkpoint['module_trans_state_dict'])
    tester.cross_atten.load_state_dict(checkpoint['cross_atten_state_dict'])
    tester.cross_atten1.load_state_dict(checkpoint['cross_atten1_state_dict'])
    tester.word_image_rate=checkpoint['word_image_rate_state_dict']
    print("load pre-trained done!")
    # tester.val_acc = checkpoint['val_acc']
    # tester.acc1 = checkpoint['val_acc1']
    # tester.acc2 = checkpoint['val_acc2']
    # tester.t1 = checkpoint['t1']
    # tester.t2 = checkpoint['t2']
    tester.global_step = checkpoint['iteration']

    print(tester.global_step)
    val_acc, h,val_acc1,h1, val_acc2,h2,val_acc3,h3,val_acc4,h4,val_acc_u,h_u,word_u_true,word_u_false,image_u_true,image_u_false = tester.eval(partition='test', word_image_rate = tester.word_image_rate)
    modality = 'semantic'
    if modality =='semantic':
        s_true = pd.Series(word_u_true)
        s_false = pd.Series(word_u_false)
    else:
        s_true = pd.Series(image_u_true)
        s_false = pd.Series(image_u_false)
    matplotlib.pyplot.figure(figsize=(8,5))
    sns.distplot(s_true, hist=False, kde=True,
             vertical=False,label='True',
            axlabel='Uncertainty',kde_kws={'color':'y','shade':True})
    sns.distplot(s_false, hist=False, kde=True,
             vertical=False,label='False',
            axlabel='Uncertainty',kde_kws={'color':'r','shade':True})
    # 用标准正态分布拟合
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlim((0, 1))
    matplotlib.pyplot.grid(linestyle='--')
    if args.dataset =='cub':
        data = 'CUB'
    elif args.dataset =='sun':
        data = 'SUN'
    else:
        data = 'Flower'
    matplotlib.pyplot.title('{} {}-image'.format(data,modality))
    matplotlib.pyplot.savefig('/home/lyh/.local/MAP-Net-main (copy)/{}_5_way_{}_shot_{}_image.pdf'.format(data,args.nExemplars,modality))
    # plt.imshow(sns.distplot)
    matplotlib.pyplot.show()
    # print("a1", 1 / (1 + torch.exp(tester.module_trans.a1.relationtransformer1.layers.fi1)))
    # print("a2", 1 / (1 + torch.exp(tester.module_trans.a1.relationtransformer1.layers.fi2)))
    # print("val acc1: %.2f +- %.2f %% " % (tester.acc1 * 100,h*100))
    # print("val acc2: %.2f  +- %.2f %%" % (tester.acc2 * 100,h*100))
    print("test acc1: %.2f +- %.2f %% " % (val_acc1 * 100,h1*100))
    print("test acc2: %.2f  +- %.2f %%" % (val_acc2 * 100,h2*100))
    print("test acc3: %.2f  +- %.2f %%" % (val_acc3 * 100,h3*100))
    print("test acc4: %.2f  +- %.2f %%" % (val_acc4 * 100,h4*100))
    print("test acc_u: %.2f  +- %.2f %%" % (val_acc_u * 100,h_u*100))
    print("test accuracy: %.2f +- %.2f %%" % (val_acc*100, h*100))
if __name__ == '__main__':
    # b = [0.2,0.3,0.4,0.6,0.7,0.8]
    # for i in b:
    # n = [ 0.1,0.2, 0.3, 0.4,0.5, 0.6, 0.7,0.8,0.9]
    # for m in n:
         i=0.5
         m = 0.5
         print(m)
         main(i,m)

