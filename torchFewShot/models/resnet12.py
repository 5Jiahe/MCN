import torch.nn as nn
import torch.nn.functional as F
from .dropblock import DropBlock
import torch

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(dropblock_size, stride=1)
            # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        # self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.norm = nn.LayerNorm(640, eps=1e-5)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_emb = nn.MaxPool2d(5)(x)
        # x_emb = kmax_pooling(x.view(x.shape[0],640,25),2,3)
        # if self.keep_avg_pool:
        #     x = self.avgpool(x)
        x_emb = self.norm(x_emb.view(-1,640))
        #x = x.view(-1,512)
        
        return x_emb,x

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return torch.mean(x.gather(dim, index),dim=dim)

def resnet12(avg_pool=True, **kwargs):
    """Constructs a ResNet12 model.
    """
    model = ResNet(BasicBlock, avg_pool=avg_pool, **kwargs)
    return model
# '''
# Reference: https://github.com/cyvius96/few-shot-meta-baseline/blob/master/models/resnet12.py
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
#                     padding=1, bias=False)

# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
#                     bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()

#         self.relu = nn.LeakyReLU(0.1)

#         self.conv1 = conv3x3(in_planes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv3x3(planes, planes)
#         self.bn3 = nn.BatchNorm2d(planes)

#         self.downsample = downsample    # residual branch downsample block
#         self.stride = stride

#         self.maxpool = nn.MaxPool2d(2)

#     def forward(self, inputs):
#         identity = inputs

#         outputs = self.conv1(inputs)
#         outputs = self.bn1(outputs)
#         outputs = self.relu(outputs)

#         outputs = self.conv2(outputs)
#         outputs = self.bn2(outputs)
#         outputs = self.relu(outputs)

#         outputs = self.conv3(outputs)
#         outputs = self.bn3(outputs)

#         if self.downsample is not None:
#             identity = self.downsample(inputs)

#         outputs += identity
#         outputs = self.relu(outputs)

#         outputs = self.maxpool(outputs)

#         return outputs

# class ResNet(nn.Module):

#     def __init__(self, block=BasicBlock, n_channels=[64, 128, 256, 512], zero_init_residual=False):
#         super(ResNet, self).__init__()
#         self.in_planes = 3

#         self.layer1 = self._make_layer(block, n_channels[0])
#         self.layer2 = self._make_layer(block, n_channels[1])
#         self.layer3 = self._make_layer(block, n_channels[2])
#         self.layer4 = self._make_layer(block, n_channels[3])
#         self.reduce = nn.Linear(512,64)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

#         self.out_channels = 512

#         self._init_conv()

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)


#     def _make_layer(self, block, planes, stride=1):
#         downsample = None
#         if stride != 1 or self.in_planes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.in_planes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion)
#             )

#         layers = []
#         layers.append(block(self.in_planes, planes, stride, downsample))
#         self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def _init_conv(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


#     def forward(self, inputs):

#         outputs = self.layer1(inputs)
#         outputs = self.layer2(outputs)
#         outputs = self.layer3(outputs)
#         outputs = self.layer4(outputs)

#         outputs = nn.MaxPool2d(5)(outputs)
#         outputs = outputs.view(-1,640)
#         # outputs = self.reduce(outputs)



#         return outputs

# def resnet12(**kwargs):
#     return ResNet(BasicBlock, n_channels=[64, 160, 320, 640], **kwargs)

# class word_encoder(nn.Module):
#     def __init__(self,
#                  indim,
#                  drop_rate):
#         super(word_encoder, self).__init__()
#
#         self.layer1 = nn.Sequential(nn.Linear(indim, 300),
#                                     # nn.BatchNorm1d(512),
#                                     nn.ReLU())
#         self.layer2 = nn.Sequential(nn.Linear(300, 64))
#         self.dropout = nn.Dropout(drop_rate)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.dropout(out)
#         out = self.layer2(out)
#
#         return out
#
# class fai(nn.Module):
#     def __init__(self,
#                  indim=128,
#                  drop_rate=0.5):
#         super(fai, self).__init__()
#
#         self.layer1 = nn.Sequential(nn.Linear(indim, 300),
#                                     # nn.BatchNorm1d(512),
#                                     nn.ReLU())
#         self.layer2 = nn.Sequential(nn.Linear(300, 1))
#         self.dropout = nn.Dropout(drop_rate)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.dropout(out)
#         out = self.layer2(out)
#
#         return out
#
# class res12_support(nn.Module):
#     def __init__(self):
#         super(res12_support, self).__init__()
#         self.image_encoder = ResNet(BasicBlock, n_channels=[64, 128, 256, 512])
#         self.word_encoder = word_encoder(312,0.5)
#         self.fai = fai()
#         self.out_channels = 64
#         self.layer1 = nn.Linear(512,300)
#         self.drop = nn.Dropout(0.5)
#         self.layer2 = nn.Linear(300,64)
#
#     def forward(self, image,word):
#         image = self.image_encoder(image)
#         image = self.layer1(image)
#         image = self.drop(image)
#         image = self.layer2(image)
#         word = self.word_encoder(word)
#         image_word = torch.cat([image,word],dim=1)
#         fai = self.fai(image_word)
#         return fai*image+(1-fai)*word

