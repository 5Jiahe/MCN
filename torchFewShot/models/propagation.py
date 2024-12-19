import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

class SelfAttention1(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention1, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.fi = nn.Parameter(torch.empty(1))

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        return out,attention


class SelfAttention2(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention2, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        return out,attention
    
class SelfAttention3(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention3, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        return out,attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(TransformerBlock, self).__init__()
        # self.norm= nn.LayerNorm(64, eps=1e-5)
        # self.knovel = knovel
        self.norm_word = nn.LayerNorm(embed_size, eps=1e-5)
        self.norm_image = nn.LayerNorm(embed_size, eps=1e-5)
        self.norm_fusion = nn.LayerNorm(embed_size, eps=1e-5)
        self.norm1 = nn.LayerNorm(embed_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_size, eps=1e-5)
        self.norm3 = nn.LayerNorm(embed_size, eps=1e-5)
        
        self.drop_word = nn.Dropout(dropout)
        self.drop_image = nn.Dropout(dropout)
        self.drop_fusion = nn.Dropout(dropout)
        
        self.attention1 = SelfAttention1(embed_size, heads)
        self.attention2 = SelfAttention2(embed_size, heads)
        self.attention3 = SelfAttention3(embed_size, heads)
        
        self.linear11 = nn.Linear(embed_size,embed_size)
        self.linear12 = nn.Linear(embed_size,embed_size)
        self.linear21 = nn.Linear(embed_size,embed_size)
        self.linear22 = nn.Linear(embed_size,embed_size)
        self.linear31 = nn.Linear(embed_size,embed_size)
        self.linear32 = nn.Linear(embed_size,embed_size)
        self.fusion = nn.Sequential(
            nn.Linear(embed_size*2,embed_size),
            nn.ReLU(),
            nn.Linear(embed_size,embed_size),
        )
        
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)
        self.dropout32 = nn.Dropout(dropout)
        self.dropout33 = nn.Dropout(dropout)
        
        # self.linear33 = nn.Linear(64,64)
        # self.he = nn.Linear(6,6,bias=False)
        self.num_supports = 5
        self.batch_size = 5
        self.num_queries = 75
        self.a1 = nn.TransformerEncoderLayer(d_model=embed_size,nhead = 1,dim_feedforward=embed_size)
        self.a2 = nn.TransformerEncoderLayer(d_model=embed_size,nhead=1,dim_feedforward=embed_size)

        #task attention
        self.task_atten1 =  nn.Sequential(
            nn.Linear(5,3),
            nn.ReLU(),
            nn.Linear(3,1),
        )
        self.task_atten2 =  nn.Sequential(
            nn.Linear(5,3),
            nn.ReLU(),
            nn.Linear(3,1),
        )
        self.classifier = nn.Linear(64,1024)

    def forward(self, word,image,query,s_word,knovel):
        # need = self.classifier(image)
        # need1 = self.classifier(word)
        # loss = torch.sum(torch.sum((need1-s_word)**2,dim=-1)/102)/(375*5)
        loss =0
        word_full = torch.cat([word,query],dim = 1)
        image_full = torch.cat([image,query], dim=1)

        fusion_full = torch.cat([self.fusion(torch.cat([word,image],dim=-1)),query],dim = 1)

        word_full_test,w1 = self._f_block1(word_full)
        image_full_test,w2 = self._f_block2(image_full)
        fusion_full_test,w2 = self._f_block3(fusion_full)
        
        
        word_pro = word_full_test[:,:knovel,:]
        word_test = word_full_test[:,-1,:].unsqueeze(1).contiguous()
        image_pro = image_full_test[:, :knovel, :]
        image_test = image_full_test[:,-1, :].unsqueeze(1).contiguous()
        fusion_pro = fusion_full_test[:, :knovel, :]
        fusion_test = fusion_full_test[:,-1, :].unsqueeze(1).contiguous()
        
        
        return word_pro, image_pro,fusion_pro, word_test, image_test,fusion_test,loss
        
    def _f_block1(self, x):
        a,w = self.attention1(x,x,x)
        word_full_trans = self.drop_word(a)
        word_full_test = self.norm_word(x+word_full_trans)
        # word_full_test = self.linear12(self.dropout12(self.relu1(self.linear11(word_full_test))))
        word_full_test = self.norm1(word_full_test+self.dropout13(self.linear12(self.dropout12(self.relu1(self.linear11(word_full_test))))))
        return word_full_test,w

    def _f_block2(self, x):
        a,w = self.attention2(x,x,x)
        image_full_trans = self.drop_image(a)
        image_full_test = self.norm_image( x+ image_full_trans)
        # image_full_test = self.linear22(self.dropout22(self.relu2(self.linear21(image_full_test))))
        image_full_test = self.norm2(image_full_test+self.dropout23(self.linear22(self.dropout22(self.relu2(self.linear21(image_full_test))))))
        return image_full_test,w
    
    def _f_block3(self, x):
        a,w = self.attention3(x,x,x)
        image_full_trans = self.drop_fusion(a)
        image_full_test = self.norm_fusion( x+ image_full_trans)
        # image_full_test = self.linear22(self.dropout22(self.relu2(self.linear21(image_full_test))))
        image_full_test = self.norm3(image_full_test+self.dropout33(self.linear32(self.dropout32(self.relu2(self.linear31(image_full_test))))))
        return image_full_test,w
    
    

class Propagation(nn.Module):
    """Label Propagation"""
    def __init__(self, args,word_dim):
        super(Propagation, self).__init__()
        self.propagation = TransformerBlock(word_dim,1,0.1)
    def forward(self, word,image,query,query_word,knovel):
        word_pro, image_pro,fusion_pro, word_test, image_test,fusion_test ,loss= self.propagation(word,image,query,query_word,knovel)
        return word_pro, image_pro,fusion_pro, word_test, image_test,fusion_test,loss







