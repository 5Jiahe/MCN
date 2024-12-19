import torch.nn as nn
import torch.nn.functional as F
# Basic ConvNet with Pooling layer
import matplotlib.pyplot as plt
import torch
import numpy as np

# support module
class word_encoder1(nn.Module):
    def __init__(self,
                 word_dim,
                 size):
        super(word_encoder1, self).__init__()
        dim = 300
        self.layer1 = nn.Sequential(nn.Linear(word_dim, dim),
                                    # nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    )
        self.layer2 = nn.Linear(dim, size)
        self.dropout = nn.Dropout(0.5)
        self.fi1 = nn.Parameter((1/8)*torch.ones(1),requires_grad=True)
        self.fi2 = nn.Parameter((1/8)*torch.ones(1),requires_grad=True)
        self.fi3 = nn.Parameter((1/8)*torch.ones(1),requires_grad=True)
        self.fi4 = nn.Parameter((1/50)*torch.ones(1),requires_grad=True)
        self.norm = nn.LayerNorm(size, eps=1e-5) 

        # self.prototype = nn.Parameter(torch.randn(size, size),requires_grad=True)
        # # self.norm = nn.LayerNorm(size, eps=1e-5)
        # self.norm1 = nn.LayerNorm(size, eps=1e-5)
        

    def forward(self, x,y):
        # if y == 1:
        # out = self.dropout1(x)
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        # out = self.norm(out)

        # out = self.norm1(torch.sum(out.unsqueeze(2)*self.norm(self.prototype).unsqueeze(0),dim=1))

        return out,self.fi1,self.fi2,self.fi3,self.fi4 
        
class word_encoder2(nn.Module):
    def __init__(self,
                 word_dim,
                 size):
        super(word_encoder2, self).__init__()

        self.prototype = nn.Parameter(torch.randn(word_dim, size),requires_grad=True)
        self.norm = nn.LayerNorm(size, eps=1e-5)
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.fi1 = nn.Parameter((1/80)*torch.ones(1),requires_grad=True)
        self.fi2 = nn.Parameter((1/80)*torch.ones(1),requires_grad=True)
        self.fi3 = nn.Parameter((1/50)*torch.ones(1),requires_grad=True)
        self.fi4 = nn.Parameter((1/50)*torch.ones(1),requires_grad=True)
    def forward(self, x,y):
        out = self.norm1(torch.sum(x.unsqueeze(2)*self.norm(self.prototype).unsqueeze(0),dim=1))
        return out,self.fi1,self.fi2,self.fi3,self.fi4 

class map(nn.Module):
    def __init__(self):
        super(map, self).__init__()

        self.layer1 = nn.Linear(640*2,640)
        self.layer2 = nn.Linear(640,640)
        self.layer3 = nn.Linear(640,300)
        self.layer4 = nn.Linear(300,1)
        

    def forward(self, x,y):
        out1 = self.layer2(self.layer1(torch.cat([x,y],dim=-1)))
        out2 = self.layer4(self.layer3(x))
        return out1,out2

class cpn(nn.Module):
    def __init__(self):
        super(cpn, self).__init__()
        self.layer1 = nn.Linear(640,1)
    def forward(self, x):
        out = self.layer1(x)
        return out

# class word_encoder(nn.Module):
#     def __init__(self,
#                  word_dim,
#                  mask_rate):
#         super(word_encoder, self).__init__()
#         self.bert = nn.TransformerEncoderLayer(d_model=64,nhead=1,dim_feedforward=64)
#         self.emb = nn.Linear(13, 64, bias=False)
#         self.norm = nn.LayerNorm(64, eps=1e-5)
        
#     def forward(self, x,y):
#         n = x.shape[0]
#         out = x.view(n,24,13)
#         out = self.emb(out)
#         out = self.bert(out)
#         out = torch.mean(out,dim=1)
#         out = self.norm(out)
#         return out,self.fi1,self.fi2

# query_module
def conv_block1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class conv4_image(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block1(x_dim, hid_dim),
            conv_block1(hid_dim, hid_dim),
            conv_block1(hid_dim, hid_dim),
            conv_block1(hid_dim, z_dim),
        )
        self.norm = nn.LayerNorm(64, eps=1e-5)
        self.layer1 = nn.Linear(64,64,bias=None)
    def forward(self, x):
        # pic = x[0,:,:,:]
        # pic = pic.cpu().numpy()
        # pic = np.transpose(pic,(1,2,0))
        # plt.imshow(pic)
        # plt.show()
        # x = self.encoder(x)
        # g = x[0,:,:,:]
        # g = g.view(64,25)
        # a, suoyin= torch.max(g,dim = 1)
        # g = g.view(64,5,5).cpu().numpy()
        # relitu = torch.zeros((64,25))
        # for xi in range(64) :
        #    relitu[xi,suoyin[xi]] = 1
        # relitu = torch.sum(relitu,dim = 0)
        # relitu = relitu.view(5,5)
        # relitu = relitu.cpu().numpy()


        x = self.encoder(x)
        # x= x.view(x.shape[0],64,25)
        x_emb = nn.MaxPool2d(5)(x)
        # x_emb = kmax_pooling(x,2,3)
        # x = torch.transpose(self.norm(torch.transpose(x,1,3)),1,3)
        x_emb = x_emb.view(-1,64)
        # x_emb = self.layer1(x_emb)
        # x_emb = self.norm(x_emb)
        # x_q_for_word = nn.MaxPool2d(5)(x_q)
        # x_q_for_word = x_q_for_word.view(-1,64)
        # x_q_for_word = self.norm(x_q_for_word)
        return x_emb,x

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return torch.mean(x.gather(dim, index),dim=dim)

# class cross_attention(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(128,64)
#         self.layer2 = nn.Linear(64, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         #space

#     def forward(self, support_image_emb,support_word_emb):
#         out = torch.cat([support_image_emb,support_word_emb],dim=2)
#         a = self.layer2(self.dropout(self.relu(self.layer1(out))))
#         return a

class cross_attention(nn.Module):

    def __init__(self,in_channels,resize_factor):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(5,3),
            nn.ReLU(),
            nn.Linear(3,1),
        )
    
    def forward(self, s):
        s = torch.transpose(s,1,2)
        weights = torch.transpose(torch.sigmoid(self.layer(s)),1,2)
        return weights
         
class cross_attention1(nn.Module):

    def __init__(self,in_channels,resize_factor):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(5,3),
            nn.ReLU(),
            nn.Linear(3,1),
        )
    
    def forward(self, s):
        s = torch.transpose(s,1,2)
        weights = torch.transpose(torch.sigmoid(self.layer(s)),1,2)
        return weights
         




# class cross_attention(nn.Module):

#     def __init__(self,in_channels,resize_factor):
#         super().__init__()
#         self.channel = in_channels
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         hid_channels = int(in_channels/resize_factor)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels*2, hid_channels, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(hid_channels, in_channels*2, kernel_size=1, bias=False),
#         )
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
#         )
       
#         self.conv1 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1, bias=False)
#         # self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
    
#     def forward(self, s,q):
#          num_s = s.shape[1]
#          channel = s.shape[2]
#          size = s.shape[3]
#          q = q.repeat(1,num_s,1,1,1)
#          full = torch.cat([s,q],dim=2)
#          full = full.view(-1,2*channel,size,size)
#          weights= self.attention(full)
#          full = self.max_pool(weights*full+full).squeeze(2).squeeze(2)
#          full = full.view(-1,num_s,channel*2)
#          s_t = full[:,:,:channel]
#          q_t = full[:,:,channel:]
#          return s_t,q_t
         

#     def attention(self,data):
#         c_weights = self.channel_attention(data)
#         s_weights = self.space_attention(data)
#         # c_weights0 = c_weights[:,:self.channel ,:,:]
#         # c_weights1 = c_weights[:,self.channel:,:,:]
#         # s_weights0 = s_weights[:,0,:,:].unsqueeze(1)
#         # s_weights1 = s_weights[:,1,:,:].unsqueeze(1)
#         # d = c_weights*s_weights
#         # d = c_weights0*s_weights0
#         weights = torch.sigmoid(self.conv1(c_weights*s_weights))
#         # weights1 = torch.sigmoid(self.conv1(c_weights*s_weights))
#         # weights0
#         # weights1
#         return weights
        
#     def channel_attention(self,data):
#         # avg_pool_weights = self.fc(self.avg_pool(data))
#         max_pool_weights = self.fc(self.max_pool(data))
#         weights = max_pool_weights
#         return weights
        
#     def space_attention(self,data):
        
#         transpose_features = data.view(*data.shape[:2], -1).transpose(1, 2).unsqueeze(3)
#         avg_pooled_features = self.avg_pool(transpose_features).squeeze(3)
#         max_pooled_features = self.max_pool(transpose_features).squeeze(3)
#         pooled_features = torch.cat((avg_pooled_features, max_pooled_features), 2)
#         pooled_features = pooled_features.transpose(1, 2).view(-1, 2, *data.shape[2:])
        
#         # data0 = data[:,:self.channel,:,:] 
#         # transpose_features0 = data0.view(*data0.shape[:2], -1).transpose(1, 2).unsqueeze(3)
#         # avg_pooled_features0 = self.avg_pool(transpose_features0).squeeze(3)
#         # max_pooled_features0 = self.max_pool(transpose_features0).squeeze(3)
#         # pooled_features0 = torch.cat((avg_pooled_features0, max_pooled_features0), 2)
#         # pooled_features0 = pooled_features0.transpose(1, 2).view(-1, 2, *data0.shape[2:])
        
#         # data1 = data[:,self.channel:,:,:]
#         # transpose_features1 = data1.view(*data1.shape[:2], -1).transpose(1, 2).unsqueeze(3)
#         # avg_pooled_features1 = self.avg_pool(transpose_features1).squeeze(3)
#         # max_pooled_features1 = self.max_pool(transpose_features1).squeeze(3)
#         # pooled_features1 = torch.cat((avg_pooled_features1, max_pooled_features1), 2)
#         # pooled_features1 = pooled_features1.transpose(1, 2).view(-1, 2, *data1.shape[2:])
#         # torch.cat([pooled_features0,pooled_features1],dim=1)
#         weights = self.fc1(pooled_features)
        
#         return weights



# class cross_attention(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.norm = nn.LayerNorm(64, eps=1e-5)
#         self.norm1 = nn.LayerNorm(32, eps=1e-5)
#         self.batchnorm1 = nn.LayerNorm(32)
#         self.batchnorm2 = nn.LayerNorm(32)
#         self.batchnorm3 = nn.LayerNorm(32)
#         self.batchnorm4 = nn.LayerNorm(64)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         # attention
#         self.conv2d_1 = nn.Conv2d(64,32,kernel_size=1)
#         self.conv2d_2 = nn.Conv2d(64, 32, kernel_size=1)
#         self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=1)
#
#         self.conv2d_11 = nn.Conv2d(64, 64, kernel_size=1)
#         self.conv2d_22 = nn.Conv2d(64, 64, kernel_size=1)
#         self.conv2d_33 = nn.Conv2d(64, 64, kernel_size=1)
#
#         self.layer1 = nn.Linear(64,64)
#         self.layer2 = nn.Linear(5,5)
#         self.layer3 = nn.Linear(25,25)
#         self.layer4 = nn.Linear(25, 25)
#
#         self.layer5 = nn.Linear(128,64)
#         self.layer6 = nn.Linear(64, 32)
#         self.layer7 = nn.Linear(64, 64)
#         self.layer8 = nn.Linear(64, 64, bias=False)
#         self.conv2_2 =  nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
#         self.conv1_1 =  nn.Conv2d(64,64,kernel_size=1)
#         self.conv1_1_1 = nn.Conv2d(64,64, kernel_size=1)
#         self.conv1_1_1_1 = nn.Conv2d(64,64, kernel_size=1)
#
#         self.sigmoid = nn.Sigmoid()
#         # space
#         # self.conv = nn.Conv2d()
#
#
#         #space
#
#
#     def forward(self, support_image_feature,support_image_emb,support_word_emb,query_image_feature,query_image_emb,num_support):
#         support_image_feature_attention = self.attention(torch.cat([support_word_emb,support_image_emb],dim = 2),support_image_feature)
#         support_image_feature = self.norm(nn.MaxPool2d(5)((support_image_feature+support_image_feature_attention).view(375*num_support,64,5,5)).squeeze(2).squeeze(2).view(375,num_support,64))
#         a = torch.softmax(torch.sum(self.layer7(query_image_emb)*self.layer7(support_word_emb)/ (64 ** (1 / 2)),dim=2),dim=1)
#         # b = torch.softmax(torch.sum(self.layer8(query_image_emb)*self.layer8(support_image_emb,)/ (64 ** (1 / 2)),dim=2),dim=1)
#         query_word_emb = torch.sum(a.unsqueeze(2)*support_word_emb,dim=1).unsqueeze(1)
#         query_image_feature_attention = self.attention(torch.cat([query_word_emb,query_image_emb],dim = 2),query_image_feature)
#         query_image_feature = self.norm(nn.MaxPool2d(5)((query_image_feature+query_image_feature_attention).view(375*1,64,5,5)).squeeze(2).squeeze(2).view(375,1,64))
#         # support_word_emb = self.layer1(support_word_emb)
#         return support_image_feature,query_image_feature,support_word_emb
#
#     # def attention(self,word,image):
#     #     a = word.shape[0]
#     #     b = word.shape[1]
#     #     word_for_attention =word
#     #     image = image.view(a*b,64,5,5)
#     #     image_for_attention = image
#     #     # image_for_attention = torch.transpose(self.batchnorm2(torch.transpose(image_for_attention,1,3)),1,3)
#     #     image_for_attention = image_for_attention.view(a, b, 64, 5, 5)
#     #     # image1 = self.conv2d_2(image)
#     #     # image1 = image1.view(a,b,32,5,5)
#     #     word_for_attention = word_for_attention.unsqueeze(3).unsqueeze(3)
#     #     attention = torch.sum(image_for_attention*word_for_attention,dim = 2)/64
#     #     at = torch.sum(torch.sum(image_for_attention*word_for_attention,dim = 3),dim=3)/25
#     #     attention = self.conv2_2(attention.view(a*b,5,5).unsqueeze(1))
#     #     at = self.conv1_1_1(self.conv1_1(at.view(a*b,64).unsqueeze(2).unsqueeze(2)))
#     #     attention_need = self.sigmoid(self.conv1_1_1_1(at*attention))
#     #     # m = attention.cpu().numpy()
#     #     # attention = attention.cpu().numpy()
#     #     out = attention_need.view(a, b, 64, 5, 5) * image_for_attention
#     #     return out
#
#     def attention(self,word,image):
#         a = word.shape[0]
#         b = word.shape[1]
#         word_for_attention =  self.layer5(word)
#         image = image.view(a*b,64,5,5)
#         image_for_attention = self.conv2d_11(image)
#         image1 = self.conv2d_22(image)
#         image1 = image1.view(a,b,64,5,5)
#         image_for_attention = image_for_attention.view(a,b,64,5,5)
#         word_for_attention = word_for_attention.unsqueeze(3).unsqueeze(3)
#         attention = torch.softmax(torch.sum(image_for_attention*word_for_attention/ (64 ** (1 / 2)),dim = 2).view(a,b,25),dim = 2)
#         # m = attention.cpu().numpy()
#         out = attention.view(a,b,5,5).unsqueeze(2)*image1
#         out = out.view(a*b,64,5,5)
#         out = self.conv2d_33(out)
#         out = out.view(a,b,64,5,5)
#         return out
    # def space_attention(self,):