import torch
import copy
import time
import tqdm
import numpy


def fedrecovery_operation(old_local_model_list,old_global_model_list,ul_client,std):
    device=0
    grad_list=[]
    weigh_list=[]
    grad_residual={}
    # compute \delta F_i， weight p_i
    with torch.no_grad():
        for i in range(len(old_global_model_list)-1):
            grad_dict={}
            for name ,param in old_global_model_list[i].items():
                grad_dict[name]=old_global_model_list[i+1][name]-old_global_model_list[i][name]
            grad_list.append(torch.cat([xx.reshape((-1, 1)) for _, xx in grad_dict.items()], dim=0).squeeze().to(device)) # \delta F_i
        # print(grad_list[0:2][0:10])
        for i in range(1,int(len(grad_list))):
            j=0
            sum_norm=torch.tensor(0.0).to(device)
            while j < i:
                sum_norm+=torch.sum(torch.pow(grad_list[j],2))
                j+=1
            weigh_list.append(torch.sum(torch.pow(grad_list[i], 2)) / sum_norm)
        # print('len:',len(weigh_list))
        # print(weigh_list[-1])

        # 计算 grad_residuals和修正后的params
        #    grad_residual[i] = 1/(n-1) *(global_model[i] - global_model[i-1] -1/n \delta W_ul)
        #                     = 1/(n-1) *(global_model[i] - global_model[i-1] - 
        #                       (local_model[i][ul_client] - global_model[i-1]) / n) )
        grad_residual_list=[]
        corrected_param=old_global_model_list[-1]
        for i in range(len(weigh_list)):
            for name, param in corrected_param.items():
                grad_residual[name]= 1/ (len(old_local_model_list[i+1])-1) * ((old_global_model_list[i+1][name]-old_global_model_list[i][name] )
                                    - (old_local_model_list[i+1][ul_client][name]-old_global_model_list[i][name])/len(old_local_model_list[i+1]) )
                corrected_param[name]-=5* weigh_list[i] * grad_residual[name] 
        # print('corrected_param:',corrected_param[name])
        # print('grad_residual:',grad_residual[name])
    # 添加高斯噪声
    
    for name, param in corrected_param.items():
        noise = torch.empty_like(corrected_param[name]).normal_(0,std) 
        corrected_param[name]+=noise
    return corrected_param







