import time
import random
import os
import copy
import dill
import json
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.backends import cudnn

import models as models
from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from utils.args import parser_args
from dataset import CIFAR10, CIFAR100
from baselines.class_pruning_base import generate, acculumate_feature, calculate_cp, \
    get_threshold_by_sparsity, TFIDFPruner, load_model_pytorch
from baselines.federaser_base import eraser_unlearning
from baselines.fedrecovery_base import fedrecovery_operation


class Unlearn_after_Learn(Experiment):
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
        self.proportion=args.proportion
        if args.ul_mode=='ul_class' or args.ul_mode=='retrain_class':
            self.proportion=1.0
        self.dp = args.dp
        self.sigma = args.sigma

        if self.args.dataset == 'cifar10':
            self.num_classes = 10
            self.in_channels=3
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
            self.in_channels=3
        elif self.args.dataset == 'mnist':
            self.num_classes = 10
            self.in_channels=1

        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_root = args.data_root
        self.data_ldr_path='log_test_class/amnesiac_ul_class/lenet/mnist/FedUL_dataloader_s0_10_32_0.01_1_2024_1_16.pkl'

        self.pretrained_model_path='log_test_class/amnesiac_ul_class/lenet/mnist/Final_s0_iid1_epoch_100_E_2_batch32_lr0.01_c_10_1.0_27.0878_0.9938.pkl'
        # model_state_dict= pkl_dict['net_info']['best_model']
        self.ul_mode=args.ul_mode
        self.ul_class_id=args.ul_class_id
        self.ul_clients=list(np.random.choice([i for i in range(args.num_users)], args.num_ul_users, replace=False))
 
        

    def construct_pretrained_model(self,pkl_path,model_name,in_channels):
        model = models.__dict__[model_name](num_classes=10,in_channels=in_channels)
        # if not ModuleValidator.is_valid(model):
        #     model = ModuleValidator.fix(model)
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        # model_ul=models.__dict__[self.args.model_name+'_ul'](num_classes=self.num_classes)
        # self.model_ul=model_ul.to(self.device)
        
        pkl_state_dict=torch.load(pkl_path)
        self.model.load_state_dict(pkl_state_dict['model_state_dict'])

        # for name in self.model.named_modules():
        #     print(name)

    
    def class_pruning(self,sparsity,unlearn_class):

        # preparing dataloader
        if args.dataset == 'cifar10':
            in_channels=3
        elif args.dataset=='mnist':
            in_channels=1

        self.args.model_name='lenet'
        self.args.dataset='mnist'
        # self.args.dataset
        self.construct_pretrained_model(self.pretrained_model_path,'lenet',1)

        with open(self.data_ldr_path,'rb') as f:
            dataloader_save = dill.load(f)
        """ dataloader_save_dict={'train_ldr':train_ldr,
                    "val_ldr":val_ldr,
                    "ul_ldr":ul_ldr,
                    "local_train_ldrs":local_train_ldrs,
                    "ul_clients":self.ul_clients
                    }
        """
        train_ldr=dataloader_save['train_ldr']
        val_ldr=dataloader_save['val_ldr']
        ul_ldr=dataloader_save['ul_ldr']

        # pre-processing
        stop_batch=1
        coe=0
        num_classes=10
        # unlearn_class= 9
        # sparsity = 0.05
        dataset=self.args.dataset
        feature_iit, classes = acculumate_feature(self.model, train_ldr, stop_batch)
        tf_idf_map = calculate_cp(feature_iit, classes, dataset, coe, unlearn_class=unlearn_class)

        # threshold_list=[]
        # for sparsity in range(5,100,5):
        #     sparsity=sparsity / 100
        threshold = get_threshold_by_sparsity(tf_idf_map, sparsity)
            # threshold_list.append(threshold)
            # threshold=torch.tensor(0.05)
            # print('threshold', threshold)
        # print('threshold_list',threshold_list)

        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma,self.num_classes,'none')

        # test before class-unlearning

        loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
        loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)
        loss_class_mean, acc_before_mean = self.trainer.test(ul_ldr)
        loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean
        loss_val_ul__mean, acc_ul_val_mean = 0, 0
        
        loss_ul_mean, acc_ul_mean = 0, 0

        print("Train acc {:.4f} -- Val acc {:.4f} --Class acc {:.4f}".format(acc_train_mean,
                                                                                    acc_val_mean,
                                                                                    acc_before_mean
                                                                            )
            )
        # for threshold,sparsity in zip(threshold_list,range(5,100,5)):
        #     # pruning...
        #     sparsity=sparsity / 100
        cp_config={ "threshold": threshold, "map": tf_idf_map }
        config_list = [{
            'sparsity': sparsity,  
            'op_types': ['Conv2d']
            }]
        pruner = TFIDFPruner(self.model, config_list, cp_config=cp_config)
        pruner.compress()
        pruned_save_path = 'log_test_class/'+self.args.model_name +'/'+self.args.dataset
        if not os.path.exists(pruned_save_path):
            os.makedirs(pruned_save_path)
        pruned_model_path = pruned_save_path +'/' +'seed_'+str(self.args.seed)+ \
                                            str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))+ \
                                            '_model.pth'

        pruned_mask_path = pruned_save_path +'/'+ ('seed_'+str(self.args.seed)+ \
                                                time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+ \
                                                '_mask.pth')
        pruner.export_model(pruned_model_path, pruned_mask_path, input_shape=(3,32,32),device=self.device)


        model = models.__dict__['lenet'](num_classes=self.num_classes,in_channels=1)
        # if not ModuleValidator.is_valid(model):
        #     model = ModuleValidator.fix(model)
        #model = torch.nn.DataParallel(model)
        self.pruned_net = model.to(self.device)
        load_model_pytorch(self.pruned_net, pruned_model_path,self.args.model_name)
        

        self.trainer_pruning = TrainerPrivate(self.pruned_net, self.device, self.dp, self.sigma,self.num_classes,'none')

        loss_val_ul__mean, acc_ul_val_mean = self.trainer_pruning.test(val_ldr)
        
        loss_ul_mean, acc_ul_mean = self.trainer_pruning.ul_test(ul_ldr)

        print("Pruning Sparsity {} Unlearned Val acc {:.4f} -- Unlearn effect {:.4f}".format(sparsity, acc_ul_val_mean,acc_ul_mean)
                    )

        """"Fine-tuning..."""

        list_allclasses = list(range(num_classes))
        unlearn_listclass = [unlearn_class]
        list_allclasses.remove(unlearn_class) # rest classes
        ft_dataloader=generate(dataset, list_allclasses)
        
        ft_epoch=10
        ft_state_dict,ft_loss=self.trainer_pruning._local_update(ft_dataloader, ft_epoch, self.lr, self.optim) 
        
        # test fot finetuning

        loss_val_ft__mean, acc_ft_val_mean = self.trainer_pruning.test(val_ldr)
        
        loss_ft_mean, acc_ft_ul_mean = self.trainer_pruning.ul_test(ul_ldr)

        print("--------------  Finetuned Val acc {:.4f} -- Finetnued Unlearn effect {:.4f}".format(sparsity, acc_ft_val_mean,acc_ft_ul_mean)
                    )

        """save logs"""
        file_name = "_".join(
                            ['Class_pruning',str(args.model_name), str(args.dataset), str(unlearn_class), f's{args.seed}',str(args.num_users), str(args.batch_size)])
        log_dir=os.getcwd() + '/'+'baselines'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        #fn=log_dir+'/'+file_name+'.txt'
        fn=log_dir+'/'+file_name+'.log'

        print("training log saved in:",fn)

        with open(fn,"a") as f:
            json.dump({"Sparsity":sparsity,"Val acc":round(acc_val_mean,4),
                       "Class acc":round(acc_before_mean, 4  ),
                       "UL val acc":round(acc_ul_val_mean,4),"UL effect":round(acc_ul_mean,4),
                       "Finetuned UL val acc":round(acc_ft_val_mean,4),"Finetuned UL effect":round(acc_ft_ul_mean,4)
                       },f)
            f.write('\n')


        return copy.deepcopy(self.pruned_net.state_dict()), acc_val_mean,acc_ul_val_mean, acc_ul_mean


    def amnesiac_unlearning(self):#,update_path, model_path, data_ldr_path

        model_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_class/amnesiac_ul_class/alexnet/cifar10/FedUL_model_s101_10_32_0.01_1_2024_1_24.pkl'
        update_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_class/amnesiac_ul_class/alexnet/cifar10/FedUL_updates_list_s101_e199_10_32_0.01_1_2024_1_24_2024_01_24_204747.pkl'
        data_ldr_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_class/amnesiac_ul_class/alexnet/cifar10/FedUL_dataloader_s101_10_32_0.01_1_2024_1_24.pkl'
        if args.dataset == 'cifar10':
            in_channels=3
        elif args.dataset=='mnist':
            in_channels=1
        self.construct_pretrained_model(model_path,'alexnet',3)

        with open(data_ldr_path,'rb') as f:
            dataloader_save = dill.load(f)
        """ dataloader_save_dict={'train_ldr':train_ldr,
                    "val_ldr":val_ldr,
                    "ul_ldr":ul_ldr,
                    "local_train_ldrs":local_train_ldrs,
                    "ul_clients":self.ul_clients
                    }
        """
        train_ldr=dataloader_save['train_ldr']
        val_ldr=dataloader_save['val_ldr']
        ul_ldr=dataloader_save['ul_ldr']

        update_list=torch.load(update_path)
        update_sum=update_list[0]

        ul_mode='amnesiac_ul_class'
        epochs=200

        if 'samples' in ul_mode and 'client' not in ul_mode:
            # if self.epochs= 200
            start_epoch= self.epochs-10  #int(self.epochs * (190/200))
            scale=1.0
        elif 'class' in ul_mode or  'client'  in ul_mode:
            start_epoch=int(0.5 * epochs)
            # scale=1/self.num_users
            scale=1/10
        pretrained_state_dict=copy.deepcopy(self.model.state_dict())
        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma,self.num_classes,'none')
        loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)
        loss_class_mean, acc_before_mean = self.trainer.test(ul_ldr)  #测试ul之前, global model对该类别样本的识别效果
        print("-- Val acc {:.4f} --UL val acc {:.4f} ".format(acc_val_mean,acc_before_mean))

        
        for epoch in range(start_epoch,epochs):
            if epoch==120:
                break
            am_start=time.time()
            for param_name in update_sum:
                update_sum[param_name]+=update_list[epoch + 1- start_epoch ][param_name] # 
            amnesiac_state_dict=copy.deepcopy(pretrained_state_dict)
            with torch.no_grad():
                for param_name in update_sum:
                    amnesiac_state_dict[param_name]-=update_sum[param_name] *0.02
                self.model.load_state_dict(amnesiac_state_dict)   
            end_time=time.time()
            print('----Epoch {}  cost time: {}'.format(epoch,am_start - end_time))

            """testing"""
            
            loss_val_ul__mean, acc_ul_val_mean = self.trainer.test(val_ldr)
            
            loss_ul_mean, acc_ul_mean = self.trainer.ul_test(ul_ldr)

            save_path='/CIS32/zgx/Unlearning/FedUnlearning/baselines/models/amnesiac_model_sub.pkl'
            torch.save(copy.deepcopy(self.model.state_dict()),save_path)

            self.model.load_state_dict(pretrained_state_dict) #重新加载回global model

            print("Epoch: {} -- Unlearned Val acc {:.4f} -- Unlearn effect {:.4f}".format(epoch,acc_ul_val_mean,acc_ul_mean)
                    )


    def federaser_unlearning(self):
        
        ## 0.05
        # old_model_dict_path='log_test_client/federaser_ul_samples_client/0.05/alexnet/cifar10/FedUL_model_state_lists_s1_10_32_0.01_1_2024_1_17.pkl'
        # model_path='log_test_client/federaser_ul_samples_client/0.05/alexnet/cifar10/FedUL_model_s1_e199_10_32_0.01_1_2024_1_17_2024_01_17_014256.pkl'
        # data_ldr_path='log_test_client/federaser_ul_samples_client/0.05/alexnet/cifar10/FedUL_dataloader_s1_10_32_0.01_1_2024_1_17.pkl'
        ## 0.02
        old_model_dict_path='log_test_client/federaser_ul_samples_client/0.1/alexnet/cifar10/FedUL_model_state_lists_s1_10_32_0.01_1_2024_1_17.pkl'
        model_path='log_test_client/federaser_ul_samples_client/0.1/alexnet/cifar10/FedUL_model_s1_e99_10_32_0.01_1_2024_1_17_2024_01_17_014335.pkl'
        data_ldr_path='log_test_client/federaser_ul_samples_client/0.1/alexnet/cifar10/FedUL_dataloader_s1_10_32_0.01_1_2024_1_17.pkl'
        
        # self.construct_pretrained_model(model_path,'alexnet',3)
        model = models.__dict__['alexnet'](num_classes=10,in_channels=3)
        # if not ModuleValidator.is_valid(model):
        #     model = ModuleValidator.fix(model)
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        # model_ul=models.__dict__[self.args.model_name+'_ul'](num_classes=self.num_classes)
        # self.model_ul=model_ul.to(self.device)
        
        pkl_state_dict=torch.load(model_path)
        self.model.load_state_dict(pkl_state_dict['model_state_dict'])
        
        with open(data_ldr_path,'rb') as f:
            dataloader_save = dill.load(f)
        """ dataloader_save_dict={'train_ldr':train_ldr,
                    "val_ldr":val_ldr,
                    "ul_ldr":ul_ldr,
                    "local_train_ldrs":local_train_ldrs,
                    "ul_clients":self.ul_clients
                    }
        """
        local_train_ldrs=dataloader_save['local_train_ldrs']
        train_ldr=dataloader_save['train_ldr']
        val_ldr=dataloader_save['val_ldr']
        ul_ldr=dataloader_save['ul_ldr']

        old_model_dicts=torch.load(old_model_dict_path)
        old_global_model_list=old_model_dicts['old_global_model_list']
        old_local_model_list=old_model_dicts['old_local_model_list']
        
        eraser_epoch=100
        print('----------------------FedEraser unlearning start-------------------')
        # Preparing the inital model and variable
        unlearn_global_model_list=[]
        new_global_model=copy.deepcopy(self.model)
        new_global_model.load_state_dict(old_global_model_list[0])
        eraser_trainer = TrainerPrivate(new_global_model, self.device, self.dp, self.sigma,self.num_classes,'none')
        eraser_lr=0.01
        ers_total_time=0
        idxs_users = np.random.choice(range(self.num_users), self.num_users, replace=False)

        file_name = "_".join(
                ['FedEraser', str(args.ul_mode), f's{args.seed}',str(args.num_users), str(args.batch_size),str(args.lr), str(args.lr_up), str(args.iid)])
        log_dir='log_test_client/federaser_ul_samples_client/0.1/alexnet/cifar10'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        #fn=log_dir+'/'+file_name+'.txt'
        fn=log_dir+'/'+file_name+'.log'

        for epoch in range(eraser_epoch):
            ers_start=time.time()
            new_local_models=[]
            for idx in range(self.num_users):
                if (idx in self.ul_clients) ==False:
                    local_w, local_loss= eraser_trainer._local_update(local_train_ldrs[idx], self.local_ep, eraser_lr, self.optim) 
                    new_local_models.append(copy.deepcopy(local_w))   
            if eraser_epoch<101:      
                eraser_lr*=0.98
            else:
                eraser_lr*=0.99
            # fedavg...
            new_global_state_dict=copy.deepcopy(new_global_model.state_dict())
            for layer in new_global_state_dict.keys():
                new_global_state_dict[layer] *= 0 
                for local_model_dict in new_local_models:
                    new_global_state_dict[layer]+=local_model_dict[layer] / len(new_local_models)
            # federaser unlearning operation
            unlearn_state_dict=eraser_unlearning(old_local_model_list[epoch],new_local_models, old_global_model_list[epoch+1], new_global_state_dict)
            new_global_model.load_state_dict(unlearn_state_dict)
            unlearn_global_model_list.append(copy.deepcopy(unlearn_state_dict))

            ers_end = time.time()
            ers_interval_time = ers_end - ers_start
            ers_total_time+=ers_interval_time

            # testing
            loss_eraser_train_mean, acc_eraser_train_mean = eraser_trainer.test(train_ldr)
            loss_val_eraser__mean, acc_eraser_val_mean = eraser_trainer.test(val_ldr)
            loss_eraser_mean, acc_eraser_mean = eraser_trainer.ul_test(ul_ldr)
            
            print('Epoch {}/{}  --time {:.1f}'.format(
                epoch, eraser_epoch,
                ers_interval_time
            ))
            
            print('Erasered val loss: {:.4f} --- Erasered test acc: {:.4f} ---Eraser effect: {:.4f} '.format(loss_val_eraser__mean, 
                                                                                    acc_eraser_val_mean,
                                                                                    acc_eraser_mean
                                                                                    ))    
                  
            with open(fn,"a") as f:
                json.dump({"Eraser epoch":epoch,"lr":round(eraser_lr,4),"train_acc":round(acc_eraser_train_mean,4  ),"test_acc":round(acc_eraser_val_mean,4),"UL effect":round(acc_eraser_mean,4),"time":round(ers_total_time,2)},f)
                f.write('\n')
        
        eraser_pkl_name = "_".join(
                    ['FedUL_model', f's{self.args.seed}',f'e{epoch}', str(args.num_users), str(args.batch_size),str(args.lr), str(args.iid)])
        eraser_pkl_name=log_dir+'/'+ eraser_pkl_name
        print("eraser_pkl_name:",eraser_pkl_name)

        # save_dict={'eraser_model_state_dict':copy.deepcopy(new_global_model.state_dict()),
        #             'model_ul_state_dict':copy.deepcopy(self.model_ul.state_dict()),
        #             "private_samples_idxs":self.private_samples_idxs,
        #             "final_train_idxs":self.final_train_idxs,
        #             "ul_clients":self.ul_clients,
        #             "dict_users":self.dict_users
        #             }
        torch.save(new_global_model.state_dict(), eraser_pkl_name+".pkl")

    def fedrecovery(self):
        ## 0.02
        old_model_dict_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/amnesiac_ul_samples_client/0.05/alexnet/cifar10/FedUL_model_state_lists_s101_10_32_0.01_1_2024_1_24.pkl'
        # lenet_state_pkl_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/federaser_ul_samples_client/0.02/lenet/mnist/FedUL_model_state_lists_s19_10_32_0.01_1_2024_1_23.pkl'
        model_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/amnesiac_ul_samples_client/0.05/alexnet/cifar10/FedUL_model_s101_10_32_0.01_1_2024_1_24.pkl'
        data_ldr_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/amnesiac_ul_samples_client/0.05/alexnet/cifar10/FedUL_dataloader_s101_10_32_0.01_1_2024_1_24.pkl'
        

        # old_model_dict_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/federaser_ul_samples_client/0.02/lenet/mnist/FedUL_model_state_lists_s19_10_32_0.01_1_2024_1_23.pkl'
        # lenet_state_pkl_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/federaser_ul_samples_client/0.02/lenet/mnist/FedUL_model_state_lists_s19_10_32_0.01_1_2024_1_23.pkl'
        # model_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/federaser_ul_samples_client/0.02/lenet/mnist/FedUL_model_s1_e99_10_32_0.01_1_2024_1_16_2024_01_16_164905.pkl'
        # data_ldr_path='/CIS32/zgx/Unlearning/FedUnlearning/log_test_client/federaser_ul_samples_client/0.02/lenet/mnist/FedUL_dataloader_s19_10_32_0.01_1_2024_1_23.pkl'
        
        # self.construct_pretrained_model(model_path,'alexnet',3)
        model = models.__dict__['alexnet'](num_classes=10,in_channels=3)
        # model_le = models.__dict__['alexnet'](num_classes=10,in_channels=3)
        # if not ModuleValidator.is_valid(model):
        #     model = ModuleValidator.fix(model)
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        # self.model_le=model_le.to(self.device)
        # model_ul=models.__dict__[self.args.model_name+'_ul'](num_classes=self.num_classes)
        # self.model_ul=model_ul.to(self.device)
        
        pkl_state_dict=torch.load(model_path)
        print(pkl_state_dict.keys())
        self.model.load_state_dict(pkl_state_dict['model_state_dict'])

        
        with open(data_ldr_path,'rb') as f:
            dataloader_save = dill.load(f)
        """ dataloader_save_dict={'train_ldr':train_ldr,
                    "val_ldr":val_ldr,
                    "ul_ldr":ul_ldr,
                    "local_train_ldrs":local_train_ldrs,
                    "ul_clients":self.ul_clients
                    }
        """
        local_train_ldrs=dataloader_save['local_train_ldrs']
        train_ldr=dataloader_save['train_ldr']
        val_ldr=dataloader_save['val_ldr']
        ul_ldr=dataloader_save['ul_ldr']
        ul_client=dataloader_save['ul_clients'][0]

        old_model_dicts=torch.load(old_model_dict_path)
        old_global_model_list=old_model_dicts['old_global_model_list']
        old_local_model_list=old_model_dicts['old_local_model_list']
        # for name ,_ in old_global_model_list[0].items():
        #     print(old_global_model_list[150][name])
        #     print(old_global_model_list[100][name])
        #     break
        # assert 0
        
        # eraser_lr=0.01
        ers_total_time=0
        # idxs_users = np.random.choice(range(self.num_users), self.num_users, replace=False)

        file_name = "_".join(
                ['FedRecovery', str(args.ul_mode), f's{args.seed}',str(args.num_users), str(args.batch_size),str(args.lr), str(args.lr_up), str(args.iid)])
        log_dir='log_test_client/fedrecovery/0.05/alexnet/cifar10'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        #fn=log_dir+'/'+file_name+'.txt'
        fn=log_dir+'/'+file_name+'.log'

        # ers_start=time.time()
        # federaser unlearning operation
        for std in [0.020,0.022,0.025]: #,0.028,0.030,0.032,0.034,0.036,0.038,0.040]:
            recovery_state_dict=fedrecovery_operation(old_local_model_list, old_global_model_list, ul_client,std)
            self.model.load_state_dict(recovery_state_dict)

            # ers_end = time.time()
            # ers_interval_time = ers_end - ers_start
            # ers_total_time+=ers_interval_time
            recovery_trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma,self.num_classes,'none')
            # testing
            loss_eraser_train_mean, acc_eraser_train_mean = recovery_trainer.test(train_ldr)
            loss_val_eraser__mean, acc_eraser_val_mean = recovery_trainer.test(val_ldr)
            loss_eraser_mean, acc_eraser_mean = recovery_trainer.ul_test(ul_ldr)
            
            # print('--Time {:.1f}'.format(
            #     ers_interval_time
            # ))
            
            print('Sigma: {}  --- Erasered val loss: {:.4f} --- Erasered test acc: {:.4f} ---Eraser effect: {:.4f} '.format(std,loss_val_eraser__mean, 
                                                                                    acc_eraser_val_mean,
                                                                                    acc_eraser_mean
                                                                                    ))    
                    
            with open(fn,"a") as f:
                json.dump({"Sigmat":std,"rain_acc":round(acc_eraser_train_mean,4  ),"test_acc":round(acc_eraser_val_mean,4),"UL effect":round(acc_eraser_mean,4),"time":round(ers_total_time,2)},f)
                f.write('\n')
        
        eraser_pkl_name = "_".join(
                    ['FedRCV_model', f's{self.args.seed}', str(args.num_users), str(args.batch_size),str(args.lr), str(args.iid)])
        eraser_pkl_name=log_dir+'/'+ eraser_pkl_name
        print("eraser_pkl_name:",eraser_pkl_name)

    def test_time(self):

        old_model_dict_path='log_test_client/federaser_ul_samples_client/0.1/alexnet/cifar10/FedUL_model_state_lists_s1_10_32_0.01_1_2024_1_17.pkl'
        model_path='log_test_client/federaser_ul_samples_client/0.1/alexnet/cifar10/FedUL_model_s1_e99_10_32_0.01_1_2024_1_17_2024_01_17_014335.pkl'
        data_ldr_path='log_test_client/federaser_ul_samples_client/0.1/alexnet/cifar10/FedUL_dataloader_s1_10_32_0.01_1_2024_1_17.pkl'
        
        # self.construct_pretrained_model(model_path,'alexnet',3)
        model = models.__dict__['alexnet'](num_classes=10,in_channels=3)
        self.model = model.to(self.device)
        model_ul=models.__dict__['alexnet'+'_ul'](num_classes=10,in_channels=3)
        self.model_ul=model_ul.to(self.device)
        
        pkl_state_dict=torch.load(model_path)
        self.model.load_state_dict(pkl_state_dict['model_state_dict'])
        self.model_ul.load_state_dict(pkl_state_dict['model_ul_state_dict'])
    
        ers_start=time.time()
        alpha=0.9
        ul_state_dict=copy.deepcopy(self.model.state_dict())
        weight_ul=((1-alpha)*self.model_ul.state_dict()['classifier.weight']+alpha*self.model_ul.state_dict()['classifier_ul.weight'])
        ul_state_dict['classifier.weight']=copy.deepcopy(weight_ul)

        bias_ul=((1-alpha)*self.model_ul.state_dict()['classifier.bias']+alpha*self.model_ul.state_dict()['classifier_ul.bias'])
        ul_state_dict['classifier.bias']=copy.deepcopy(bias_ul)

        self.model.load_state_dict(ul_state_dict)

        ers_end = time.time()
        ers_interval_time = ers_end - ers_start

        print('---------cost time:',ers_interval_time)

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
    unlearn_after_learn = Unlearn_after_Learn(args)
    ul_mode='amnesiac_ul'
    if ul_mode=='class_pruning':
        sparsity=args.class_prune_sparsity
        unlearn_class=args.class_prune_target
        for sparsity in range(5,100,5):
            sparsity=sparsity / 100
            state_dict,  acc_val_mean,acc_ul_val_mean, acc_ul_mean = unlearn_after_learn.class_pruning(sparsity,unlearn_class)                                         
        

        logs['net_info'] = state_dict  #logg=self.logs,    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())
        logs['test_acc'] = acc_val_mean
        logs['test_acc_after_ul'] = acc_ul_val_mean
        logs['ul_effect'] = acc_ul_mean
    elif ul_mode=='amnesiac_ul':
        unlearn_after_learn.amnesiac_unlearning()
    elif ul_mode =='federaser_ul':
        unlearn_after_learn.federaser_unlearning()
    elif ul_mode=='test_time':
        unlearn_after_learn.test_time()
    elif ul_mode == 'fedrecovery':
        unlearn_after_learn.fedrecovery()

    
    #print(logg['keys'])
    
    # pkl_path=os.getcwd()+'/'+save_dir + '/'+args.model_name +'/' + args.dataset
    # if not os.path.exists(pkl_path):
    #     os.makedirs(pkl_path)
    # torch.save(logs,
    #            pkl_path + '/Final_s{}_iid{}_epoch_{}_E_{}_batch{}_lr{}_c_{}_{:.1f}_{:.4f}_{:.4f}.pkl'.format(
    #                args.seed,args.iid, args.epochs, args.local_ep,args.batch_size, args.lr, args.num_users, args.frac, time, acc_ul_val_mean
    #            ))
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














