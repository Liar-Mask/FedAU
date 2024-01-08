import os
from utils.args import parser_args
from utils.datasets import *
import copy
import random
import datetime
from tqdm import tqdm
import numpy as np
import math
from scipy import spatial
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import models as models

# from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from dataset import CIFAR10, CIFAR100
# import wandb

class IPRFederatedLearning(Experiment):
    """
    Perform federated learning
    """
    def __init__(self, args):
        super().__init__(args) # define many self attributes from args
        self.watch_train_client_id=0
        self.watch_val_client_id=1

        self.criterion = torch.nn.CrossEntropyLoss()
        self.in_channels = 3
        self.optim=args.optim
        self.num_bit = args.num_bit
        self.num_trigger = args.num_trigger
        self.proportion=args.proportion
        if args.ul_mode=='ul_class' or args.ul_mode=='retrain_class':
            self.proportion=1.0
        self.dp = args.dp
        self.sigma = args.sigma
        self.cosine_attack =args.cosine_attack  
        self.sigma_sgd = args.sigma_sgd
        self.grad_norm=args.grad_norm
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_root = args.data_root

        self.ul_mode=args.ul_mode
        self.ul_class_id=args.ul_class_id
        self.ul_clients=list(np.random.choice([i for i in range(args.num_users)], args.num_ul_users, replace=False))
 
        print('==> Preparing data...')
        self.train_set, self.test_set, self.ul_test_set, self.dict_users, self.train_idxs, self.val_idxs, self.private_samples_idxs, self.final_train_idxs = get_data(dataset=self.dataset,
                                                        data_root = self.data_root,
                                                        proportion=self.proportion,
                                                        iid = self.iid,
                                                        num_users = self.num_users,
                                                        UL_clients=self.ul_clients,
                                                        data_aug=self.args.data_augment,
                                                        noniid_beta=self.args.beta,
                                                        samples_per_user=args.samples_per_user,
                                                        ul_mode=self.ul_mode,
                                                        ul_class_id=self.ul_class_id
                                                        )
        # self.train_idxs # dict id 2 array
        # self.val_idxs_for_mia=[]
        # for i in range(self.num_users):
        #     if i == self.watch_train_client_id:
        #         continue
        #     self.val_idxs_for_mia.extend(self.train_idxs[i])
        # random.shuffle (self.val_idxs_for_mia )
        # self.val_idxs_for_mia=self.val_idxs_for_mia[0:len(self.train_idxs[self.watch_train_client_id])]

        if self.args.dataset == 'cifar10':
            # self.dataset = CIFAR10
            self.num_classes = 10
            # self.dataset_size = 60000
        elif self.args.dataset == 'cifar100':
            # self.dataset = CIFAR100
            self.num_classes = 100
            # self.dataset_size = 60000
            
        elif self.args.dataset == 'dermnet':
            # self.dataset = CIFAR100
            self.num_classes = 23
            # self.dataset_size = 19500
        elif self.args.dataset == 'oct':
            # self.dataset = CIFAR100
            self.num_classes = 4
            # self.dataset_size = 19500
     
        self.MIA_trainset_dir=[]
        self.MIA_valset_dir=[]
        self.MIA_trainset_dir_cos=[]
        self.MIA_valset_dir_cos=[]
        self.train_idxs_cos=[]
        self.testset_idx=(50000+np.arange(10000)).astype(int) # 最后10000样本的作为test set
        self.testset_idx_cos=(50000+np.arange(1000)).astype(int)

        print('==> Preparing model...')

        self.logs = {'train_acc': [], 'train_sign_acc':[], 'train_loss': [],
                     'val_acc': [], 'val_loss': [],
                     'test_acc': [], 'test_loss': [],
                     'keys':[],

                     'best_test_acc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     }

        self.construct_model()
        
        self.w_t = copy.deepcopy(self.model.state_dict())

        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma,self.num_classes,'none')
        self.trainer_ul=TrainerPrivate(self.model_ul, self.device, self.dp, self.sigma,self.num_classes,self.ul_mode)
        self.tester = TesterPrivate(self.model, self.device)

        self.makedirs_or_load()
    
              
    def construct_model(self):

        # model = models.__dict__[self.args.model_name](num_classes=self.num_classes*2)
        model = models.__dict__[self.args.model_name](num_classes=self.num_classes)
        # if not ModuleValidator.is_valid(model):
        #     model = ModuleValidator.fix(model)
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        model_ul=models.__dict__[self.args.model_name+'_ul'](num_classes=self.num_classes)
        self.model_ul=model_ul.to(self.device)
        
        # torch.backends.cudnn.benchmark = True
        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))


    def train(self):
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(DatasetSplit(self.train_set,self.train_set.final_train_list), batch_size=self.batch_size *2, shuffle=False, num_workers=4)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size *2, shuffle=False, num_workers=4)
        test_ldr = DataLoader(self.test_set, batch_size=self.batch_size , shuffle=False, num_workers=0)
        ul_ldr = DataLoader(self.ul_test_set, batch_size=self.batch_size *2, shuffle=False, num_workers=4)
        
        # 不区分iid 和 non iid (还是要区分的，两者的dict_users不同)

        # torch.backends.cudnn.benchmark = True

        local_train_ldrs = []
        if args.iid:
            for i in range(self.num_users):
                local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]), batch_size = self.batch_size,
                                                shuffle=True, num_workers=2)
                local_train_ldrs.append(local_train_ldr)

        else:  #copy原版的
            for i in range(self.num_users):
                local_train_ldr = DataLoader(self.dict_users[i], batch_size = self.batch_size,
                                                shuffle=True, num_workers=2)
                local_train_ldrs.append(local_train_ldr) 


        total_time=0
        time_mark=str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime()))
        file_name = "_".join(
                ['FedUL', f's{args.seed}',str(args.num_users), str(args.batch_size),str(args.lr), str(args.lr_up), str(args.iid), time_mark])
        log_dir=os.getcwd() + '/'+args.log_folder_name+'/'+ args.model_name +'/'+ args.dataset

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        #fn=log_dir+'/'+file_name+'.txt'
        fn=log_dir+'/'+file_name+'.log'

        print("training log saved in:",fn)

        lr_0=self.lr
        ul_state_dicts={}
        print('UL_clients:',self.ul_clients)
        for i in self.ul_clients:
            # print(i)
            ul_state_dicts[i]=copy.deepcopy(self.model_ul.state_dict())

        for epoch in range(self.epochs):

            global_state_dict=copy.deepcopy(self.model.state_dict())

            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)

            local_ws, local_losses,= [], []

            start = time.time()
            '''
            开始训练, 对于每轮每个client先判断是否为ul_client, 再判断ul_mode是否为retrain, 以此选择训练方式
            '''
            for idx in tqdm(idxs_users, desc='Epoch:%d, lr:%f' % (self.epochs, self.lr)):
 
                if (idx in self.ul_clients) ==False:
                    # print(idx,"True1000000")
                    self.model.load_state_dict(global_state_dict) # 还原 global model
                    
                    local_w, local_loss= self.trainer._local_update(local_train_ldrs[idx], self.local_ep, self.lr, self.optim) 
                    local_ws.append(copy.deepcopy(local_w))
                    local_losses.append(local_loss)
                else:
                    # print(idx,"False1000000")
                    if self.ul_mode != 'retrain_samples' and self.ul_mode != 'retrain_class':
                        # print("ul-idx:",idx)
                        self.model_ul.load_state_dict(ul_state_dicts[idx])
                        # ul_model除W2外替换为global model的参数
                        self.model_ul.load_state_dict(global_state_dict,strict=False)
                        # 参数替换完毕，开始训练
                        local_w_ul, local_loss, classify_loss, normalize_loss= self.trainer_ul._local_update_ul(local_train_ldrs[idx], self.local_ep, self.lr, self.optim,self.ul_class_id) 
                        
                        # 本次ul_model结果保存（用于下轮更新W2）
                        ul_state_dicts[idx]=copy.deepcopy(local_w_ul)
                        # 提取W1 (全局模型加载W1，保存到待avg列表中)
                        self.model.load_state_dict(local_w_ul,strict=False)

                        # class_loss,class_acc=self.trainer.test(ul_ldr)
                        # print('**** local class loss: {:.4f}  local class acc: {:.4f}****'.format(class_loss,class_acc))
                        
                        local_ws.append(copy.deepcopy(self.model.state_dict()))
                    else:   # retrain scheme
                        self.model.load_state_dict(global_state_dict)
                        # retrain训练，除sample数量减少外（训练过程对ul sample剔除），过程与正常客户端相同
                        local_w, local_loss= self.trainer._local_update(local_train_ldrs[idx], self.local_ep, self.lr, self.optim,self.ul_mode ) 
                        local_ws.append(copy.deepcopy(local_w))
                        local_losses.append(local_loss)

                
                # test_loss, test_acc=self.trainer.test(val_ldr)  

                ## 计算model grads，处理量化，稀疏化和差分隐私
                # model_grads={}
                # for name, local_param in self.model.named_parameters():
                #     if local_param.requires_grad == True:
                #         model_grads[name]= local_w[name] - global_state_dict[name]
                # local_ws.append(copy.deepcopy(model_grads))# 应该是计算local delta w
                # local_ws.append(copy.deepcopy(local_w))
                # local_losses.append(local_loss)

            # if self.optim=="sgd":
            #     self.lr=0.0001+lr_0 * (1 + math.cos(math.pi * epoch/ self.args.epochs)) / 2 
            # else:
            #     pass

            if self.optim=="sgd":
                if self.args.lr_up=='common':
                    self.lr = self.lr * 0.99
                elif self.args.lr_up =='milestone':
                    if args.epochs==500:
                        milestones=[275,400]
                    elif args.epochs==300:
                        milestones=[150,225]
                    
                    if epoch in milestones:
                        self.lr *= 0.1
                else:
                    self.lr=lr_0 * (1 + math.cos(math.pi * epoch/ self.args.epochs)) / 2 
            else:
                pass

            client_weights = []
            for i in range(self.num_users):
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[i]))/len(self.train_set)
                client_weights.append(client_weight)
            
            self.fed_avg(local_ws, client_weights, 1)
            self.model.load_state_dict(self.w_t)# 经过avg之后的model作为下一轮的global model
            

            end = time.time()
            interval_time = end - start
            total_time+=interval_time
            '''
            测试global model和ul_model效果
            '''
            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
                loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)
                loss_class_mean, acc_class_mean = self.trainer.test(ul_ldr)  #测试ul之前, global model对该类别样本的识别效果
                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean

                """
                需要对self.model_ul测试: 
                测试 (W1+W2)/2 后的 ul_acc、val_acc
                """
                if (self.ul_mode != 'retrain_samples') and (self.ul_mode != 'retrain_class'):
                    if self.ul_mode=='ul_samples' or self.ul_mode=='ul_samples_backdoor':
                        ul_state_dict=copy.deepcopy(self.w_t)
                        
                        # print("W1:",self.model.state_dict()['classifier.weight'])
                        with torch.no_grad():

                            alpha=0.99
                            weight_ul=((1-alpha)*self.model_ul.state_dict()['classifier.weight']+alpha*self.model_ul.state_dict()['classifier_ul.weight'])
                            ul_state_dict['classifier.weight']=copy.deepcopy(weight_ul)

                            bias_ul=((1-alpha)*self.model_ul.state_dict()['classifier.bias']+alpha*self.model_ul.state_dict()['classifier_ul.bias'])
                            ul_state_dict['classifier.bias']=copy.deepcopy(bias_ul)

                            self.model.load_state_dict(ul_state_dict)
                    else:
                                                # ul_model除W2外替换为global model的参数
                        # W2=[]
                        # model_dict=self.model_ul.state_dict()
                        # for layer in  model_dict.keys():
                        #     if layer in global_state_dict.keys()==False:
                        #             for idx in self.ul_clients:  

                        #                 W2.append(ul_state_dicts[idx][layer]*1/int(len(self.ul_clients)))
                        combined_state_dict=copy.deepcopy(self.w_t)

                        with torch.no_grad():
                            weight_ul=combined_state_dict['classifier.weight']
                            bias_ul=combined_state_dict['classifier.bias']
                            for idx in self.ul_clients:
                                weight_ul-= 1/int(len(self.ul_clients)) * ul_state_dicts[idx]['classifier_ul.weight']
                                bias_ul-= 1/int(len(self.ul_clients)) * ul_state_dicts[idx]['classifier_ul.bias']
                            
                            combined_state_dict['classifier.weight']=copy.deepcopy(weight_ul)
                            combined_state_dict['classifier.bias']=copy.deepcopy(bias_ul)

                            self.model.load_state_dict(combined_state_dict)
                       


                # print("(W1+W2)/2:",self.model.state_dict()['classifier.weight'])
                """
                测试 基于Ul module替换W1之后的效果, 包括:
                1. 验证集acc
                2. ul_test_set的acc 
                    (ul_samples时, ul_test_set=指定的Unlearn样本集合;
                     ul_class时, ul_test_set=origin test_set中 target=ul_class_id的样本集合)
                
                测试完毕后重新加载回 global model以备下一轮训练
                """
                loss_val_ul__mean, acc_ul_val_mean = self.trainer.test(val_ldr)
                
                loss_ul_mean, acc_ul_mean = self.trainer.ul_test(ul_ldr)

                self.model.load_state_dict(self.w_t) #重新加载回global model

                self.logs['train_acc'].append(acc_train_mean)
                self.logs['train_loss'].append(loss_train_mean)
                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))


                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())

                print('Epoch {}/{}  --time {:.1f}'.format(
                    epoch, self.epochs,
                    interval_time
                )
                )

                print(
                    "Train Loss {:.4f}  -- Val Loss {:.4f} --Unlearned Val Loss {:.4f}"
                    .format(loss_train_mean, loss_val_mean, loss_val_ul__mean))
                print("Train acc {:.4f} -- Val acc {:.4f} --Class acc {:.4f} --Best acc {:.4f}".format(acc_train_mean,
                                                                                    acc_val_mean,
                                                                                    acc_class_mean,
                                                                                    self.logs['best_test_acc']
                                                                                                        )
                    )
                print("Unlearned Val acc {:.4f} -- Unlearn effect {:.4f}".format(acc_ul_val_mean,acc_ul_mean)
                    )
                # s = 'epoch:{}, lr:{}, val_acc:{:.4f}, val_loss:{:.4f}, tarin_acc:{:.4f}, train_loss:{:.4f},time:{:.4f}, total_time:{:.4f}'.format(epoch,self.lr,acc_val_mean,loss_val_mean,acc_train_mean,loss_train_mean,interval_time,total_time)
                
                # with open(fn, 'a', encoding = 'utf-8') as f:   
                #     f.write(s)
                #     f.write('\n')
                with open(fn,"a") as f:
                    json.dump({"epoch":epoch,"lr":round(self.lr,4),"train_acc":round(acc_train_mean,4  ),"test_acc":round(acc_val_mean,4),"UL val acc":round(acc_ul_val_mean,4),"UL effect":round(acc_ul_mean,4),"time":round(total_time,2)},f)
                    f.write('\n')
            today = datetime.date.today()
            if (epoch+1) % 100==0:
                self.model_ul.load_state_dict(self.w_t,strict=False) #更新model_ul，用于保存
                save_dir =os.getcwd() + f'/{self.args.log_folder_name}/' + self.args.model_name +'/' + self.args.dataset
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pkl_name = "_".join(
                            ['FedUL_model', f's{self.args.seed}',f'e{epoch}', str(args.num_users), str(args.batch_size),str(args.lr), str(args.iid), f'{today.year}_{today.month}_{today.day}',time_mark])
                pkl_name=save_dir+'/'+ pkl_name
                print("pkl_name:",pkl_name)

                save_dict={'model_state_dict':copy.deepcopy(self.model_ul.state_dict()),
                           "private_samples_idxs":self.private_samples_idxs,
                           "final_train_idxs":self.final_train_idxs
                           }
                torch.save(save_dict, pkl_name+".pkl")
        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f}  '.format(self.logs['best_test_loss'], 
                                                                                       self.logs['best_test_acc']
                                                                                       ))

        return self.logs, interval_time, self.logs['best_test_acc'], acc_test_mean


    def _fed_avg_ldh(self,global_model, local_ws, client_weights, lr_outer ): # conduct fedavg with local delta w
        w_avg=copy.deepcopy(global_model)
        client_weights=1.0/len(local_ws)
        for k in w_avg.keys():
            for i in range(0, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights*lr_outer
            self.w_t[k] = w_avg[k]
        return w_avg
    
    def fed_avg(self, local_ws, client_weights, lr_outer):

        w_avg = copy.deepcopy(local_ws[0])
        client_weight=1.0/len(local_ws)
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weight

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weight *lr_outer

            self.w_t[k] = w_avg[k]


def main(args):
    logs = {'net_info': None,
            'arguments': {
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_outer': args.lr_outer,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,      
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,                
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'num_users': args.num_users
            }
            }
    # args.save_dir=""
    # save_dir = args.save_dir
    save_dir=args.log_folder_name
    fl = IPRFederatedLearning(args)

    logg, time, best_test_acc, test_acc = fl.train()                                         
                                             
    logs['net_info'] = logg  #logg=self.logs,    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())
    logs['test_acc'] = test_acc
    logs['bp_local'] = True if args.bp_interval == 0 else False
    #print(logg['keys'])
    
    pkl_path=os.getcwd()+'/'+save_dir + '/'+args.model_name +'/' + args.dataset
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    torch.save(logs,
               pkl_path + '/Dp_{}_{}_iid_{}_num_sign_{}_w_type_{}_loss_{}_B_{}_alpha_{}_num_back_{}_type_{}_T_{}_epoch_{}_E_{}_u_{}_{:.1f}_{:.4f}_{:.4f}.pkl'.format(
                   args.dp, args.sigma, args.iid, args.num_sign, args.weight_type, args.loss_type, args.num_bit, args.loss_alpha, args.num_back, args.backdoor_indis, args.num_trigger, args.epochs, args.local_ep, args.num_users, args.frac, time, test_acc
               ))
    return

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parser_args()
    print(args)

    setup_seed(args.seed)

    main(args)
    # wandb.finish()