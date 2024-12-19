from __future__ import print_function
from __future__ import division
import test
import os
import sys
import os.path as osp
import torch
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



def main(word_mask_rate,image_mask_rate):
        parser = argument_parser()
        args = parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        use_gpu = torch.cuda.is_available()
        args.save_dir = os.path.join(args.save_dir, args.dataset, args.backbone, '%dway_%dshot'%(args.nKnovel, args.nExemplars), args.model_name)
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
        # print("==========\nArgs:{}\n==========".format(args))

        args.log_dir = osp.join(args.save_dir, 'log')

        writer = SummaryWriter(log_dir=args.log_dir)

        if use_gpu:
            # print("Currently using GPU {}".format(args.gpu_devices))
            cudnn.benchmark = True
        # else:
            # print("Currently using CPU (GPU is highly recommended)")

        # print('Initializing image data manager')
        dm = DataManager(args, use_gpu)
        trainloader, testloader = dm.return_dataloaders()

        data_loader = {'train': trainloader,
                       'val': testloader
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
            module_image = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5)
            channel_dim = 640
        module_trans = Propagation(args,channel_dim)
        cross_atten = cross_attention(channel_dim,2)
        cross_atten1 = cross_attention1(channel_dim,2)
        # create trainer
        trainer = ModelTrainer(args=args,
                               module_word=module_word,
                               module_image=module_image,
                               module_trans = module_trans,
                               data_loader=data_loader,
                               writer=writer,
                               word_mask_rate=word_mask_rate,
                               image_mask_rate=image_mask_rate,
                               cross_atten = cross_atten,
                               cross_atten1 = cross_atten1,
                               word_dim =word_dim,
                               channel_dim = channel_dim)

        if args.resume:
            checkpoint = torch.load(args.save_dir + '/model_best.pth.tar')
            trainer.module_support.load_state_dict(checkpoint['enc_module_state_dict'])
            trainer.module_query.load_state_dict(checkpoint['enc_module_state_dict'])
            print("load pre-trained enc_nn done!")
            trainer.lp_module.load_state_dict(checkpoint['lp_module_state_dict'])
            print("load pre-trained lp_nn done!")

            trainer.val_acc = checkpoint['val_acc']
            trainer.global_step = checkpoint['iteration']
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])

            print(trainer.global_step)



        trainer.train()

if __name__ == '__main__':
    # b = [0.2,0.4,0.6,0.8,1.2,1.4,1.6,1.8,2.0]
    # ci = 0
    # # # # # for word_mask_rate in b:
    # n = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # for image_mask_rate in n:
    #      print("round",ci)
    #      ci = ci+1
         word_mask_rate = 1
         image_mask_rate = 0.3
         main(word_mask_rate,image_mask_rate)
         test.main(word_mask_rate,image_mask_rate)
         print(word_mask_rate,image_mask_rate)
