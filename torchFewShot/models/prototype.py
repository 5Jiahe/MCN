'''
Reference: https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/utils/prototype.py
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchFewShot.models.tmc import TMC
def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)
    prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

    return prototypes

def prototypical_loss(support_word_test, support_image_test,support_fusion_test, query_word_test, query_image_test,query_fusion_test, targets,epoch,image_mask_rate,channel_dim,fi1,fi2,fi3,fi4,task_atten, **kwargs):
    
    sq_distances1 = torch.sum(support_word_test* query_word_test, dim=-1)  
    sq_distances2 = torch.sum(support_image_test * query_image_test, dim=-1)
    sq_distances3 = torch.sum(support_fusion_test * query_fusion_test, dim=-1)
    sq = dict()
    Tanh = nn.Tanh()
    sq[0],sq[1],sq[2] = torch.exp(sq_distances1*fi1),torch.exp(sq_distances2*fi2),torch.exp(sq_distances3*fi3)
    
    targets = targets.squeeze(1)
    tmc = TMC(classes=5)
    dis = dict()
    dis[0],dis[1] ,dis[2]= sq_distances1,sq_distances2,sq_distances3
    loss1,loss2,loss3,loss4,u,a,b,mean_k_e = tmc.mmm(sq,dis,targets,epoch,image_mask_rate,fi3 )
    # loss3 = F.cross_entropy(,targets)
    return loss1,loss2,loss3,loss4,u,mean_k_e
    # return F.cross_entropy(squared_distances1,targets),F.cross_entropy(squared_distances2,targets),0

def get_proto_accuracy_12(support_word_test, support_image_test, support_fusion_test, query_word_test, query_image_test,query_fusion_test, targets,rate,fi1,fi2,fi3,fi4):
    
    sq_distances1 = torch.sum(support_word_test* query_word_test, dim=-1)  
    sq_distances2 = torch.sum(support_image_test * query_image_test, dim=-1)
    sq_distances3 = torch.sum(support_fusion_test * query_fusion_test, dim=-1)
    sq = dict()
    Tanh = nn.Tanh()
    sq[0],sq[1],sq[2] = torch.exp(sq_distances1*fi1),torch.exp(sq_distances2*fi2),torch.exp(sq_distances3*fi3)
    targets = targets.squeeze(1)
    tmc = TMC()
    dis = dict()
    dis[0],dis[1] ,dis[2]= sq_distances1,sq_distances2,sq_distances3
    loss1,loss2,loss3,loss4,u,a,b,b_a = tmc.mmm(sq,dis,targets,30,0,fi3  )
    
    
    _, predictions1 = torch.max(sq_distances1, dim=-1)
    _, predictions2 = torch.max(sq_distances2, dim=-1)
    _, predictions3 = torch.max(sq_distances3, dim=-1)
    _, predictions4 = torch.max(b[0]+b[1], dim=-1)
    return torch.mean(predictions1.eq(targets).float()), torch.mean(predictions2.eq(targets).float()),torch.mean(predictions3.eq(targets).float()), torch.mean(predictions4.eq(targets).float()),predictions1,predictions2


 
def get_proto_accuracy(support_word_test, support_image_test, support_fusion_test, query_word_test, query_image_test,query_fusion_test,
                       targets,epoch,rate,channel_dim,fi1,fi2,fi3,fi4,word_image_rate,task_atten):
    sq_distances1 = torch.sum(support_word_test* query_word_test, dim=-1)  
    sq_distances2 = torch.sum(support_image_test * query_image_test, dim=-1)
    sq_distances3 = torch.sum(support_fusion_test * query_fusion_test, dim=-1)
    sq = dict()
    Tanh = nn.Tanh()
    sq[0],sq[1],sq[2] = torch.exp(sq_distances1*fi1),torch.exp(sq_distances2*fi2),torch.exp(sq_distances3*fi3)
    targets = targets.squeeze(1)
    tmc = TMC(classes=5)
    
    dis = dict()
    dis[0],dis[1] ,dis[2]= sq_distances1,sq_distances2,sq_distances3
    loss1,loss2,loss3,loss4,u,a,b,b_a= tmc.mmm(sq,dis,targets,30,0,fi3 )
    
    
    u_w = u[0].squeeze(1).cpu().numpy()
    u_i = u[1].squeeze(1).cpu().numpy()

    
    # _, predictions1 = torch.max(sq_distances1*(1/u_need[0])*p_sim[:,0].unsqueeze(1)+sq_distances2*(1/u_need[1])*p_sim[:,1].unsqueeze(1)+sq_distances3*(1/u_need[2])*p_sim[:,2].unsqueeze(1), dim=1)
    # _, predictions2 = torch.max(sq_distances1*(1/u_need[0])*p_sim[:,0].unsqueeze(1)+sq_distances2*(1/u_need[1])*p_sim[:,1].unsqueeze(1)+sq_distances3*(1/u_need[2])*p_sim[:,2].unsqueeze(1), dim=1)
    n = dict()
    for i in range(3):
        n[i] = sq[i]/(torch.sum(sq[0],dim=-1)+5).unsqueeze(1)

     
    # _, predictions1 = torch.max(n[0]*p_sim[:,0].unsqueeze(1)+n[1]*p_sim[:,1].unsqueeze(1), dim=1)
    # _, predictions2 = torch.max(n[0]*p_sim[:,0].unsqueeze(1)+n[1]*p_sim[:,1].unsqueeze(1), dim=1)
    # _, predictions1 = torch.max(a, dim=1)
    # _, predictions2 = torch.max(a, dim=1)
    a = 0.3
    _, predictions1 = torch.max(sq_distances1*u[1]*a+sq_distances2*u[0]*(1-a), dim=1)
    _, predictions2 = torch.max(sq_distances1*u[1]*a+sq_distances2*u[0]*(1-a), dim=1)
    # _, predictions1 = torch.max(sq_distances1*(1/u_need[0])+sq_distances2*(1/u_need[1])+sq_distances3*(1/u_need[2]), dim=1)
    # _, predictions2 = torch.max(sq_distances1*(1/u_need[0])+sq_distances2*(1/u_need[1])+sq_distances3*(1/u_need[2]), dim=1)
    
    return torch.mean(predictions1.eq(targets).float()),torch.mean(predictions2.eq(targets).float()),u_w,u_i
