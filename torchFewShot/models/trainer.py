import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import shutil
import numpy as np
import os
import scipy as sp
import scipy.stats
from torchFewShot.models.prototype import get_prototypes, prototypical_loss, get_proto_accuracy, get_proto_accuracy_12

# class h(nn.Module):
#     def __init__(self):
#         super(h, self).__init__()
#         self.we = nn.Parameter(torch.randn(1,requires_grad=False))
#     def forward(self, x):
#         self.we = x

class ModelTrainer(object):
    def __init__(self,
                args,
                module_word,
                module_image,
                module_trans,
                data_loader,
                writer,
                word_mask_rate,image_mask_rate,cross_atten,cross_atten1,word_dim,channel_dim):
        self.word_dim = word_dim
        self.channel_dim = channel_dim
        self.image_mask_rate = image_mask_rate
        self.word_mask_rate = word_mask_rate
        self.acc1 = 0
        self.acc2 = 0
        self.t1 = 0
        self.t2 = 0
        self.args = args
        self.module_image = module_image.cuda()
        self.module_word = module_word.cuda()
        self.module_trans = module_trans.cuda()
        self.cross_atten = cross_atten.cuda()
        self.cross_atten1 = cross_atten1.cuda()
        self.word_image_rate = 0
        self.word_true_queue = []
        self.word_false_queue = []
        self.image_true_queue = []
        self.image_false_queue = []
        self.image_queue = [1]
        self.word_queue = [1]
        self.loss_rate = 0.5
        # get data loader
        self.data_loader = data_loader
        # set optimizer
        self.module_params = list(self.module_word.parameters()) + list(self.module_image.parameters())+list(self.module_trans.parameters())+list(self.cross_atten.parameters())+list(self.cross_atten1.parameters())

        if self.args.optim == 'adam':
            self.optimizer = optim.Adam(params=self.module_params,
                                        lr=args.lr,
                                        weight_decay=args.weight_decay
                                        )
        elif self.args.optim == 'sgd':
            self.optimizer = optim.SGD(params=self.module_params,
                                        lr=args.lr,
                                        momentum=0.9,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)


        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

        self.writer = writer

    def train(self):
        val_acc = self.val_acc
        batch_size = self.args.train_batch
        num_supports = self.args.nKnovel * self.args.nExemplars
        num_queries = self.args.train_nTestNovel
        # batch_size = self.args.train_batch
        # num_supports = 60
        # num_queries = 60
        nKnovel = 5
        
        

        # for each iteration
        for epoch in range(self.global_step + 1, self.args.max_epoch + 1):
    
            # set as train mode
            self.module_word.train()
            self.module_image.train()
            self.module_trans.train()
            self.cross_atten.train()
            self.cross_atten1.train()

            self.global_step = epoch
            train_len = len(self.data_loader['train'])

            for idx, (support_data, support_word, support_label, query_data, query_word_real, query_label, label) in enumerate(self.data_loader['train']):
                support_image = support_data.cuda()
                support_word = support_word.cuda()
                query_image = query_data.cuda()
                query_word_real = query_word_real.cuda()
                support_label = support_label.cuda()
                query_label = query_label.cuda()
                query_label_test = query_label.contiguous().view(batch_size * num_queries, -1)

                # feature_extracter
                support_word_emb,fi1,fi2,fi3,fi4 = self.module_word(support_word.view(-1, self.word_dim),0)
                support_word_protype = get_prototypes(support_word_emb.view(batch_size,num_supports,self.channel_dim), support_label, self.args.nKnovel)
                # support_word_protype = support_word_emb.view(batch_size,num_supports,self.channel_dim)
                # need =  get_prototypes(support_word.view(batch_size,num_supports,102), support_label, self.args.nKnovel)
                # need = (need).unsqueeze(1).repeat(1,num_queries,1,1).view(batch_size*num_queries,nKnovel,102)
                need = 0
                support_word_protype = (support_word_protype).unsqueeze(1).repeat(1,num_queries,1,1).view(batch_size*num_queries,nKnovel,self.channel_dim)


                support_image_emb,s_image = self.module_image(support_image.view(-1, *query_image.shape[2:]))
                support_image_protype = get_prototypes(support_image_emb.view(batch_size,num_supports,self.channel_dim), support_label, self.args.nKnovel)
                # support_image_protype =support_image_emb.view(batch_size,num_supports,self.channel_dim)
                
                
                support_image_protype = (support_image_protype).unsqueeze(1).repeat(1,num_queries,1,1).view(batch_size*num_queries,nKnovel,self.channel_dim)
                
                query_image_emb,q_image= self.module_image(query_image.view(-1, *support_image.shape[2:]))
                query_image_emb = query_image_emb.view(batch_size,num_queries,self.channel_dim)
                query_image_emb = (query_image_emb).view(batch_size*num_queries,self.channel_dim).unsqueeze(1)
                
                
                support_word_protype,support_image_protype,support_fusion_protype,query_for_word,query_for_image,fusion_for_image,loss_mse= self.module_trans(support_word_protype ,support_image_protype,query_image_emb,need,nKnovel)
                # support_word_protype,support_image_protype,support_fusion_protype,query_for_word,query_for_image,fusion_for_image= support_word_protype ,support_image_protype,support_image_protype,query_for_word,query_for_word,query_for_word
                
                loss1,loss2,loss3,loss4,u,mean_k_e= prototypical_loss(support_word_protype,
                                               support_image_protype,
                                               support_fusion_protype,
                                               query_for_word,
                                               query_for_image,
                                               fusion_for_image,
                                               query_label_test,
                                               epoch,self.image_mask_rate,self.channel_dim,fi1,fi2,fi3,fi4,0)
                
                acc1, acc2,acc3,acc4,p1,p2 = get_proto_accuracy_12(support_word_protype,
                                               support_image_protype,
                                               support_fusion_protype,
                                               query_for_word,
                                               query_for_image,
                                               fusion_for_image,
                                                query_label_test,self.word_image_rate,fi1,fi2,fi3,fi4)
                # acc1, acc2=torch.zeros(1),torch.zeros(1)
                loss_word = torch.mean(loss1)
                loss_image = torch.mean(loss2)
                loss3 = torch.mean(loss3)
                loss4 = torch.mean(loss4)

                loss =loss_word+loss_image
                # loss =loss_word+loss_image
                # +torch.mean(loss3)+torch.mean(loss4)
                # loss = loss1+loss2s
                # self.loss_rate = word_grad_norm/image_grad_norm

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # self.mo_op(self.module_support_mo.parameters(),self.module_support.parameters())
                # self.mo_op(self.module_query_mo.parameters(),self.module_query.parameters())

                # adjust learning rate
                self.adjust_learning_rate(optimizers=[self.optimizer],lr=self.args.lr,iters=self.global_step)
                
                self.writer.add_scalar('Loss/loss', loss.item(), (epoch-1)*self.args.train_batch*len(self.data_loader['train']) + self.args.train_batch*(idx))
                self.writer.add_scalar('Loss/loss_cls', loss.item(), (epoch-1)*self.args.train_batch*len(self.data_loader['train']) + self.args.train_batch*(idx))

                
                # print(i)
                if (idx) % (train_len // 5) == 0:
                    print('Epoch %d: train/loss1 %.4f, train/accr1 %.4f, train/accr2 %.4f' % (self.global_step, loss.data.cpu(), acc1.data.cpu(), acc2.data.cpu()))
                    # print(loss3,mean_k_e)
            # evaluation
            word_true_queue_tensor = torch.tensor(self.word_true_queue)
            word_true_u_mean = torch.mean(word_true_queue_tensor, dim=0)
            word_false_queue_tensor = torch.tensor(self.word_false_queue)
            word_false_u_mean = torch.mean(word_false_queue_tensor, dim=0)
            word_fenlidu = word_false_u_mean-word_true_u_mean

            image_true_queue_tensor = torch.tensor(self.image_true_queue)
            image_true_u_mean = torch.mean(image_true_queue_tensor, dim=0)
            image_false_queue_tensor = torch.tensor(self.image_false_queue)
            image_false_u_mean = torch.mean(image_false_queue_tensor, dim=0)
            image_fenlidu = image_false_u_mean-image_true_u_mean

            self.word_image_rate =image_fenlidu/word_fenlidu
            
            val_acc, h,val_acc1,h1, val_acc2,h2,val_acc3,h3,val_acc4,h4,val_acc_u,h_u,wu1,wu2,wu3,wu4 = self.eval(partition='val', word_image_rate=self.word_image_rate)
            is_best = 0

            if val_acc >= self.val_acc:
                self.val_acc = val_acc
                is_best = 1

            print('===>  Epoch %d: val/accr1 %.4f,val/accr2 %.4f,val/accr3 %.4f,val/accr4 %.4f,val/accr_u %.4f,val/accr %.4f, val/best_accr %.4f' % (self.global_step,val_acc1,val_acc2, val_acc3, val_acc4,val_acc_u,val_acc, self.val_acc))
            print('rate:',self.loss_rate)
            self.save_checkpoint({
                'iteration': self.global_step,
                'module_word_state_dict': self.module_word.state_dict(),
                'module_image_state_dict': self.module_image.state_dict(),
                'module_trans_state_dict': self.module_trans.state_dict(),
                'cross_atten_state_dict': self.cross_atten.state_dict(),
                'cross_atten1_state_dict': self.cross_atten1.state_dict(),
                'word_image_rate_state_dict': self.word_image_rate,
                # 't1':self.t1,
                # 't2':self.t2,
                # 'val_acc': val_acc,
                # 'val_acc1': val_acc1,
                # 'val_acc2': val_acc2,
                # 'val_acc3': val_acc3,
                'optimizer': self.optimizer.state_dict()
                }, is_best)

    def eval(self, partition='test', log_flag=True, word_image_rate = 0):
        best_acc = 0
        # set edge mask (to distinguish support and query edges)

        batch_size = self.args.test_batch
        num_supports = self.args.nKnovel * self.args.nExemplars
        num_queries = self.args.nTestNovel
        num_samples = num_supports + num_queries
        nKnovel = 5
        acc_all = []
        acc1_all = []
        acc2_all = []
        acc3_all = []
        acc4_all = []
        acc_all_u =[]
        k_all = []
        image_uncertainty_true = []
        image_uncertainty_false = []
        word_uncertainty_true = []
        word_uncertainty_false = []
        # set as eval mode
        self.module_word.eval()
        self.module_image.eval()
        self.module_trans.eval()
        self.cross_atten.eval()
        self.cross_atten1.eval()
        data = np.load('/home/lyh/.local/MAP-Net-main (copy)/home/abc/Datasets/cub/CUB_200_2011/npz/attributes-train.npz')
        fields = data['features'], data['targets']
        fields1 = fields[0]
        train_word = torch.tensor(fields1)
        train_word = train_word.cuda()

        with torch.no_grad():
            for i, (support_data, support_word, support_label, query_data, query_word_real, query_label) in enumerate(self.data_loader[partition]):
                support_image = support_data.cuda()
                support_word = support_word.cuda()
                query_image = query_data.cuda()
                query_word_real = query_word_real.cuda()
                support_label = support_label.cuda()
                query_label = query_label.cuda()
                query_label_test = query_label.contiguous().view(batch_size * num_queries, -1)

                # feature_extracter
                support_word_emb,fi1,fi2,fi3,fi4 = self.module_word(support_word.view(-1, self.word_dim),0)
                support_word_protype = get_prototypes(support_word_emb.view(batch_size,num_supports,self.channel_dim), support_label, self.args.nKnovel)
                # support_word_protype = support_word_emb.view(batch_size,num_supports,self.channel_dim)
                # need =  get_prototypes(support_word.view(batch_size,num_supports,102), support_label, self.args.nKnovel)
                # need = (need).unsqueeze(1).repeat(1,num_queries,1,1).view(batch_size*num_queries,nKnovel,102)
                need = 0
                support_word_protype = (support_word_protype).unsqueeze(1).repeat(1,num_queries,1,1).view(batch_size*num_queries,nKnovel,self.channel_dim)


                support_image_emb,s_image = self.module_image(support_image.view(-1, *query_image.shape[2:]))
                support_image_protype = get_prototypes(support_image_emb.view(batch_size,num_supports,self.channel_dim), support_label, self.args.nKnovel)
                # support_image_protype =support_image_emb.view(batch_size,num_supports,self.channel_dim)
                
                
                support_image_protype = (support_image_protype).unsqueeze(1).repeat(1,num_queries,1,1).view(batch_size*num_queries,nKnovel,self.channel_dim)
                
                query_image_emb,q_image= self.module_image(query_image.view(-1, *support_image.shape[2:]))
                query_image_emb = query_image_emb.view(batch_size,num_queries,self.channel_dim)
                query_image_emb = (query_image_emb).view(batch_size*num_queries,self.channel_dim).unsqueeze(1)
                
                
                support_word_protype,support_image_protype,support_fusion_protype,query_for_word,query_for_image,fusion_for_image,loss_mse= self.module_trans(support_word_protype ,support_image_protype,query_image_emb,need,nKnovel)
                # support_word_protype,support_image_protype,support_fusion_protype,query_for_word,query_for_image,fusion_for_image= support_word_protype ,support_image_protype,support_image_protype,query_for_word,query_for_word,query_for_word  
                acc1,acc2,acc3,acc4,p1,p2 = get_proto_accuracy_12(support_word_protype,
                                               support_image_protype,
                                               support_fusion_protype,
                                               query_for_word,
                                               query_for_image,
                                               fusion_for_image,
                                                   query_label_test,self.image_mask_rate,fi1,fi2,fi3,fi4)
                
                        

                acc,acc_u,a,b = get_proto_accuracy(support_word_protype,
                                               support_image_protype,
                                               support_fusion_protype,
                                               query_for_word,
                                               query_for_image,
                                               fusion_for_image,
                                         query_label_test,i,self.image_mask_rate,self.channel_dim,fi1,fi2,fi3,fi4,word_image_rate,0)

                for i in range(batch_size*num_queries):
                    if p1[i]==query_label_test[i]:
                        word_uncertainty_true.insert(0, a[i])
                    else:
                        word_uncertainty_false.insert(0, a[i])
                        
                for i in range(batch_size*num_queries):
                    if p2[i]==query_label_test[i]:
                        image_uncertainty_true.insert(0, b[i])
                    else:
                        image_uncertainty_false.insert(0, b[i])


                acc1_all.append(acc1.data.cpu())
                acc2_all.append(acc2.data.cpu())
                acc3_all.append(acc3.data.cpu())
                acc4_all.append(acc4.data.cpu())
                acc_all.append(acc.data.cpu())
                acc_all_u.append(acc_u.data.cpu())

        #    word_for_need[i]=train_word[min_indice[i]]
        acc_mean1, h1 = self.mean_confidence_interval(acc1_all)
        acc_mean2, h2 = self.mean_confidence_interval(acc2_all)
        acc_mean3, h3 = self.mean_confidence_interval(acc3_all)
        acc_mean4, h4 = self.mean_confidence_interval(acc4_all)
        acc_mean, h = self.mean_confidence_interval(acc_all)
        acc_mean_u, h_u = self.mean_confidence_interval(acc_all_u)
        
        return acc_mean, h,acc_mean1,h1,acc_mean2,h2,acc_mean3, h3,acc_mean4, h4,acc_mean_u, h_u,np.array(word_uncertainty_true),np.array(word_uncertainty_false),np.array(image_uncertainty_true),np.array(image_uncertainty_false)

    def adjust_learning_rate(self, optimizers, lr, iters):
        # new_lr = lr * (0.5 ** (int(iters / 15)))

        if iters in self.args.schedule:
            lr *= 0.1
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].cuda()

    def save_checkpoint(self, state, is_best):
        torch.save(state, os.path.join(self.args.save_dir, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(self.args.save_dir, 'checkpoint.pth.tar'),
                            os.path.join(self.args.save_dir, 'model_best.pth.tar'))

    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0*np.asarray(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m, h

    def image_mask_train(self, x,mask_rate,batch_size,num_supports):
        mask = torch.ones((batch_size*num_supports, 84, 84)).cuda()
        if mask_rate != 0 :
            for xi in range(batch_size*num_supports):
                a = np.arange(0, 12 * 12)
                np.random.shuffle(a)
                b = a[0:int(mask_rate*144)]
                for yi in range(int(mask_rate*144)):
                    mask_y = int(b[yi] / 12) * 7
                    mask_y_end = mask_y + 7
                    mask_x = (b[yi] % 12) * 7
                    mask_x_end = mask_x + 7
                    mask[xi, mask_x:mask_x_end, mask_y:mask_y_end] = 0
        mask = mask.view(batch_size,num_supports,84,84).unsqueeze(2)
        out = x*mask
        return out

    def image_daluan_train(self, x,batch_size = 5,num_supports = 75):
        mask = torch.ones((batch_size*num_supports,3,84, 84)).cuda()
        x = x.view(batch_size*num_supports,3,84, 84)
        for xi in range(batch_size*num_supports):
            a = np.arange(0, 4 * 4)
            np.random.shuffle(a)
            yuan = np.arange(0, 4 * 4)
            for yi in range(4 * 4):
                mask_y = int(a[yi] / 4) * 21
                mask_y_end = mask_y + 21
                mask_x = (a[yi] % 4) * 21
                mask_x_end = mask_x + 21

                yuan_y = int(yuan[yi] / 4) * 21
                yuan_y_end = yuan_y + 21
                yuan_x = (yuan[yi] % 4) * 21
                yuan_x_end = yuan_x + 21

                mask[xi, :,mask_x:mask_x_end, mask_y:mask_y_end] = x[xi, :,yuan_x:yuan_x_end, yuan_y:yuan_y_end]
        mask = mask.view(batch_size,num_supports,3,84,84)
        return mask

    def mask_test(self,x_train, x):
        x = x.view(25,312)
        dis_matrix = torch.abs(x.unsqueeze(1) - x_train.unsqueeze(0))
        dis_matrix = torch.where(dis_matrix>0.5,dis_matrix,torch.zeros(25,100,312).cuda())
        dis_matrix = torch.where(dis_matrix<0.000001,dis_matrix,torch.ones(25,100,312).cuda())
        dis_matrix = 1-dis_matrix
        dis_matrix = torch.sum(dis_matrix,dim = 1)
        mask = torch.where(dis_matrix<0.00000000000001,dis_matrix,torch.ones(25,312).cuda())
        # a = mask.cpu().numpy()
        x = x*mask
        return x.view(5,5,312)



    def mask_val(self, x):
        mask = torch.ones((25,312)).cuda()
        for xi in range(25):
            a = np.arange(0, 312)
            np.random.shuffle(a)
            b = a[0:30]
            mask[xi,b] = 0


        return x.view(5,5,312)

    def mask(self):
        a1= np.arange(0, 20)
        np.random.shuffle(a1)
        b = 4
        a = np.arange(0,28)
        np.random.shuffle(a)
        b = a[0:b]
        mask = torch.ones(312)
        for m in b:
            if m == 0:
                mask[0:9] = 0
            elif m == 1:
                mask[9:24] = 0
            elif m == 2:
                mask[24:54] = 0
            elif m == 3:
                mask[54:58] = 0
            elif m == 4:
                mask[58:73] = 0
            elif m == 5:
                mask[73:79] = 0
            elif m == 6:
                mask[79:94] = 0
            elif m == 7:
                mask[94:99] = 0
            elif m == 8:
                mask[99:105] = 0
            elif m == 9:
                mask[105:120] = 0
            elif m == 10:
                mask[120:135] = 0
            elif m == 11:
                mask[135:149] = 0
            elif m == 12:
                mask[149:152] = 0
            elif m == 13:
                mask[152:167] = 0
            elif m == 14:
                mask[167:182] = 0
            elif m == 15:
                mask[182:197] = 0
            elif m == 16:
                mask[197:212] = 0
            elif m == 17:
                mask[212:217] = 0
            elif m == 18:
                mask[217:222] = 0
            elif m == 19:
                mask[222:236] = 0
            elif m == 20:
                mask[236:240] = 0
            elif m == 21:
                mask[240:244] = 0
            elif m == 22:
                mask[244:248] = 0
            elif m == 23:
                mask[248:263] = 0
            elif m == 24:
                mask[263:278] = 0
            elif m == 25:
                mask[278:293] = 0
            elif m == 26:
                mask[293:308] = 0
            elif m == 27:
                mask[308:312] = 0
        return mask

    def mask_val1(self,dic,data):
        train_word = dic.cuda()
        query = data
        query = query.unsqueeze(2).repeat(1, 1, 100, 1)
        dic = train_word.unsqueeze(0).repeat(5, 1, 1).unsqueeze(0).repeat(5, 1, 1, 1)
        diff = torch.abs(query - dic)
        sim = torch.zeros((5, 5, 100, 28)).cuda()
        sim[:, :, :, 0] = torch.sum(diff[:, :, :, 0:9], dim=3)/(0-9)
        sim[:, :, :, 1] = torch.sum(diff[:, :, :, 9:24], dim=3)/(9-24)
        sim[:, :, :, 2] = torch.sum(diff[:, :, :, 24:54], dim=3)/(24-54)
        sim[:, :, :, 3] = torch.sum(diff[:, :, :, 54:58], dim=3)/(54-58)
        sim[:, :, :, 4] = torch.sum(diff[:, :, :, 58:73], dim=3)/(58-73)
        sim[:, :, :, 5] = torch.sum(diff[:, :, :, 73:79], dim=3)/(73-79)
        sim[:, :, :, 6] = torch.sum(diff[:, :, :, 79:94], dim=3)/(79-94)
        sim[:, :, :, 7] = torch.sum(diff[:, :, :, 94:99], dim=3)/(94-99)
        sim[:, :, :, 8] = torch.sum(diff[:, :, :, 99:105], dim=3)/(99-105)
        sim[:, :, :, 9] = torch.sum(diff[:, :, :, 105:120], dim=3)/(105-120)
        sim[:, :, :, 10] = torch.sum(diff[:, :, :, 120:135], dim=3)/(120-135)
        sim[:, :, :, 11] = torch.sum(diff[:, :, :, 135:149], dim=3)/(135-149)
        sim[:, :, :, 12] = torch.sum(diff[:, :, :, 149:152], dim=3)/(149-152)
        sim[:, :, :, 13] = torch.sum(diff[:, :, :, 152:167], dim=3)/(152-167)
        sim[:, :, :, 14] = torch.sum(diff[:, :, :, 167:182], dim=3)/(167-182)
        sim[:, :, :, 15] = torch.sum(diff[:, :, :, 182:197], dim=3)/(182-197)
        sim[:, :, :, 16] = torch.sum(diff[:, :, :, 197:212], dim=3)/(197-212)
        sim[:, :, :, 17] = torch.sum(diff[:, :, :, 212:217], dim=3)/(212-217)
        sim[:, :, :, 18] = torch.sum(diff[:, :, :, 217:222], dim=3)/(217-222)
        sim[:, :, :, 19] = torch.sum(diff[:, :, :, 222:236], dim=3)/(222-236)
        sim[:, :, :, 20] = torch.sum(diff[:, :, :, 236:240], dim=3)/(236-240)
        sim[:, :, :, 21] = torch.sum(diff[:, :, :, 240:244], dim=3)/(240-244)
        sim[:, :, :, 22] = torch.sum(diff[:, :, :, 244:248], dim=3)/(244-248)
        sim[:, :, :, 23] = torch.sum(diff[:, :, :, 248:263], dim=3)/(248-263)
        sim[:, :, :, 24] = torch.sum(diff[:, :, :, 263:278], dim=3)/(263-278)
        sim[:, :, :, 25] = torch.sum(diff[:, :, :, 278:293], dim=3)/(278-293)
        sim[:, :, :, 26] = torch.sum(diff[:, :, :, 293:308], dim=3)/(293-308)
        sim[:, :, :, 27] = torch.sum(diff[:, :, :, 308:312], dim=3)/(308-312)
        sim = -sim
        find_zeros = torch.where(sim < 0.02, sim, torch.zeros((5, 5, 100, 28)).cuda())
        find_zeros = torch.sum(find_zeros, dim=2)
        mask = torch.ones(5,5,312).cuda()
        for xi in range(5):
            for yi in range(5):
                if find_zeros[xi, yi, 0] == 0:
                    mask[xi, yi,0:9] = 0
                if find_zeros[xi, yi, 1] == 0:
                    mask[xi, yi,9:24] = 0
                if find_zeros[xi, yi, 2] == 0:
                    mask[xi, yi,24:54] = 0
                if find_zeros[xi, yi, 3] == 0:
                    mask[xi, yi,54:58] = 0
                if find_zeros[xi, yi, 4] == 0:
                    mask[xi, yi,58:73] = 0
                if find_zeros[xi, yi, 5] == 0:
                    mask[xi, yi,73:79] = 0
                if find_zeros[xi, yi, 6] == 0:
                    mask[xi, yi,79:94] = 0
                if find_zeros[xi, yi, 7] == 0:
                    mask[xi, yi,94:99] = 0
                if find_zeros[xi, yi, 8] == 0:
                    mask[xi, yi,99:105] = 0
                if find_zeros[xi, yi, 9] == 0:
                    mask[xi, yi,105:120] = 0
                if find_zeros[xi, yi, 10] == 0:
                    mask[xi, yi,120:135] = 0
                if find_zeros[xi, yi, 11] == 0:
                    mask[xi, yi,135:149] = 0
                if find_zeros[xi, yi, 12] == 0:
                    mask[xi, yi,149:152] = 0
                if find_zeros[xi, yi, 13] == 0:
                    mask[xi, yi,152:167] = 0
                if find_zeros[xi, yi, 14] == 0:
                    mask[xi, yi,167:182] = 0
                if find_zeros[xi, yi, 15] == 0:
                    mask[xi, yi,182:197] = 0
                if find_zeros[xi, yi, 16] == 0:
                    mask[xi, yi,197:212] = 0
                if find_zeros[xi, yi, 17] == 0:
                    mask[xi, yi,212:217] = 0
                if find_zeros[xi, yi, 18] == 0:
                    mask[xi, yi,217:222] = 0
                if find_zeros[xi, yi, 19] == 0:
                    mask[xi, yi,222:236] = 0
                if find_zeros[xi, yi, 20] == 0:
                    mask[xi, yi,236:240] = 0
                if find_zeros[xi, yi, 21] == 0:
                    mask[xi, yi,240:244] = 0
                if find_zeros[xi, yi, 22] == 0:
                    mask[xi, yi,244:248] = 0
                if find_zeros[xi, yi, 23] == 0:
                    mask[xi, yi,248:263] = 0
                if find_zeros[xi, yi, 24] == 0:
                    mask[xi, yi,263:278] = 0
                if find_zeros[xi, yi, 25] == 0:
                    mask[xi, yi,278:293] = 0
                if find_zeros[xi, yi, 26] == 0:
                    mask[xi, yi,293:308] = 0
                if find_zeros[xi, yi, 27] == 0:
                    mask[xi, yi,308:312] = 0
        # find_zeros = find_zeros.cpu().numpy()
        # a  = mask.cpu().numpy()
        return mask