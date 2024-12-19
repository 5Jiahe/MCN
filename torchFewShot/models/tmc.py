import torch
import torch.nn as nn
import torch.nn.functional as F


# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step,image_mask_rate):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    # annealing_coef *
    alp = E * (1 - label) + 1
    B =  annealing_coef*KL(alp, c)*image_mask_rate
# +B.squeeze(1)
    return A.squeeze(1)+B.squeeze(1)
# torch.mean(A.squeeze(1)+B.squeeze(1))

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class TMC(object):

    def __init__(self, classes=5, views=2, lambda_epochs=25):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
    

    def DS_Combin(self, alpha,dim,fi3):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2,dim,fi3):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = dim/S[v]
            
            # if need=1:
               
            # if need=2:
               

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, dim, 1), b[1].view(-1, 1, dim))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag
            
            # k=1-torch.sum(b[0]*b[1],dim=1).unsqueeze(1)-u[0]*u[1]
            # k_e = torch.exp(-k)
           
            # b_a = torch.mul(b[0], b[1])+k*k_e*(b[0]+b[1])/2
            # u_a=u[0]*u[1]+k*k_e*(u[0]+u[1])/2+k*(1-k_e)
           
           
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))
            
            # calculate new S
            S_a = dim / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a,u,b,u_a,b_a
        # B = range(len(alpha)-1)
        # for v in range(len(alpha)-1):
        #     if v==0:
        alpha_a,u,b,u_a,b_a = DS_Combin_two(alpha[0], alpha[1],dim,fi3)
        # alpha_a,u,b,u_a,b_a  = DS_Combin_two(alpha_a, alpha[2],dim,fi3)
            # else:
            #     alpha_a = DS_Combin_two(alpha_a, alpha[v+1],dim)
        
        
        return alpha_a,u,b,u_a,b_a

    def mmm(self, X,dis, y, global_step,image_mask_rate,fi3):
        # word_pro,image_pro = word_pro.cpu().numpy(),image_pro.cpu().numpy()
        evidence = X
        alpha = dict()
        alpha[0] = evidence[0] + 1
        loss1 = ce_loss(y, alpha[0], self.classes, global_step, 30, image_mask_rate)
        
        alpha[1] = evidence[1] + 1
        loss2 = ce_loss(y, alpha[1], self.classes, global_step, 30, image_mask_rate)

        alpha[2] = evidence[2] + 1
        loss3 = ce_loss(y, alpha[2], self.classes, global_step, 30, image_mask_rate)
        
        # p,all =  dict(),dict()
        # for v in range(3):
        #         all[v] = torch.sum(evidence[v], dim=1, keepdim=True)
        #         p[v] = evidence[v]/(all[v].expand(evidence[v].shape))
        # p_cat = torch.cat([p[0].unsqueeze(1),p[1].unsqueeze(1),p[2].unsqueeze(1)],dim=1)
        # p_dis = (torch.sum((p_cat.unsqueeze(2)-p_cat.unsqueeze(1))**2,dim=-1))**0.5
        
        # p_dis = torch.sum((p_cat.unsqueeze(2)-p_cat.unsqueeze(1))**2,dim=-1)
        # p_mask = torch.tensor([[0,1,1],[1,0,1],[1,1,0]]).unsqueeze(0).cuda()
        # p_dis_sum = torch.sum(p_dis*p_mask,dim=-1)
        # p_sim = torch.softmax(-p_dis_sum,dim=-1)
        # mean = torch.mean(p_sim,dim=0)
        # print(mean)

        alpha_a,u,b,u_a,b_a = self.DS_Combin(alpha,self.classes,fi3)
        loss4 = ce_loss(y, alpha[2], self.classes, global_step, 30, image_mask_rate)
        
        
        # loss4 = self.loss_tri(u_a.squeeze(1),dis[0],y)+self.loss_tri(u_a.squeeze(1),dis[1],y)
        
        # kl = nn.KLDivLoss(reduction='batchmean')
        # p1 = torch.cat([b[0],u[0]],dim=1)
        # p2 = torch.cat([b[1],u[1]],dim=1)
        # p = torch.cat([b_a,u_a],dim=1)
        # p1 = evidence[0]/(torch.sum(evidence[0],dim=-1).unsqueeze(1))
        # p2 = evidence[1]/(torch.sum(evidence[1],dim=-1).unsqueeze(1))
        # p = (evidence[0]+evidence[1])/(torch.sum(evidence[0]+evidence[1],dim=-1).unsqueeze(1))
        # loss3 = -(kl(torch.log(p1),p.detach())+kl(torch.log(p2),p.detach()))*0.5
        u_a = u_a.detach()
        return loss1,loss2,loss3,loss4,u,alpha_a,b,b_a
    
    def loss_tri(self,u_a,dis,y):
        a, idx = torch.sort(u_a, descending=True)
        n_top =100
        idx_top = idx[:n_top]
        dis_tri = dis[idx_top,:]
        label_tri = y[idx_top]
        need = torch.range(0,n_top-1).cuda().int()
        dis_pos = dis_tri[need,label_tri]
        mask = 1 - F.one_hot(label_tri, num_classes=5)
        dis_mask = dis_tri*mask + mask*10000
        a_1, idx_1 = torch.sort(dis_mask, descending=True,dim=1)
        idx_neg = idx_1[:,:4]
        dis_neg = torch.cat([dis_tri[need,idx_neg[:,0]].unsqueeze(1),dis_tri[need,idx_neg[:,1]].unsqueeze(1),dis_tri[need,idx_neg[:,2]].unsqueeze(1)
                             ,dis_tri[need,idx_neg[:,3]].unsqueeze(1)],dim=1)
        margin = 3
        com = -dis_pos.unsqueeze(1)+dis_neg+margin
        mask_label = (com > 0)
        loss = torch.sum(mask_label*com)/(4*n_top)
        return loss
    
    def shang(self,p):
        out = torch.sum(-p*torch.log(p),dim=1)
        return out



