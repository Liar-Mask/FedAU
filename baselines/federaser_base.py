import torch
import copy
import time
import tqdm
import numpy
from experiments.trainer_private import TrainerPrivate, TesterPrivate

# def federaser(initial_model, device, num_classes):
#     """
#     Parameters:
#     initial_model: old_global_model_list[0]

#     """
#     eraser_epoch=50
#     print('------------------FedEraser unlearning start-----------------')

#     unlearn_global_model_list=[]
#     new_global_model=initial_model[0]
#     eraser_trainer = TrainerPrivate(new_global_model, device, False, 0 , num_classes,'none')
#     idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)
#     eraser_lr=0.01
#     ers_total_time=0

#     for epoch in range(eraser_epoch):
#         ers_start=time.time()
#         new_local_models=[]
#         for idx in tqdm(idxs_users, desc='Epoch:%d, lr:%f' % (self.epochs, self.lr)):
#             if (idx in self.ul_clients) ==False:
#                 local_w, local_loss= eraser_trainer._local_update(local_train_ldrs[idx], self.local_ep, eraser_lr, self.optim) 
#                 new_local_models.append(copy.deepcopy(local_w))         
#         eraser_lr*=0.95

#         # fedavg...
#         new_global_state_dict=copy.deepcopy(new_global_model.state_dict())
#         for layer in new_global_state_dict.keys():
#             new_global_state_dict[layer] *= 0 
#             for local_model_dict in new_local_models:
#                 new_global_state_dict[layer]+=local_model_dict[layer] / len(new_local_models)
#         # federaser unlearning operation
#         unlearn_state_dict=eraser_unlearning(old_local_model_list[epoch],new_local_models, old_global_model_list[epoch+1], new_global_state_dict)
#         new_global_model.load_state_dict(unlearn_state_dict)
#         unlearn_global_model_list.append(copy.deepcopy(unlearn_state_dict))


def eraser_unlearning(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    """

    Parameters
    ----------
    old_client_models : list of DNN models
        When there is no choice to forget (if_forget=False), use the normal continuous learning training to get each user's local model.The old_client_models do not contain models of users that are forgotten.
        Models that require forgotten users are not discarded in the Forget function
    ref_client_models : list of DNN models
        When choosing to forget (if_forget=True), train with the same Settings as before, except that the local epoch needs to be reduced, other parameters are set in the same way.
        Using the above training Settings, the new global model is taken as the starting point and the reference model is trained.The function of the reference model is to identify the direction of model parameter iteration starting from the new global model
        
    global_model_before_forget : The old global model
        DESCRIPTION.
    global_model_after_forget : The New global model
        DESCRIPTION.

    Returns
    -------
    return_global_model : After one iteration, the new global model under the forgetting setting

    """
    old_param_update = dict()#Model Params： oldCM - oldGM_t
    new_param_update = dict()#Model Params： newCM - newGM_t
    
    new_global_model_state = global_model_after_forget#newGM_t
    
    return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    
    assert len(old_client_models) == len(new_client_models)
    
    for layer in global_model_before_forget.keys():
        old_param_update[layer] = 0*global_model_before_forget[layer]
        new_param_update[layer] = 0*global_model_before_forget[layer]
        
        return_model_state[layer] = 0*global_model_before_forget[layer]
        
        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii][layer]
            new_param_update[layer] += new_client_models[ii][layer]
        old_param_update[layer] /= (ii+1)#Model Params： oldCM
        new_param_update[layer] /= (ii+1)#Model Params： newCM
        
        old_param_update[layer] = old_param_update[layer] - global_model_before_forget[layer]#参数： oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget[layer]#参数： newCM - newGM_t
        
        step_length = torch.norm(old_param_update[layer])#||oldCM - oldGM_t||
        step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])#(newCM - newGM_t)/||newCM - newGM_t||
        
        return_model_state[layer] = new_global_model_state[layer] + step_length*step_direction
    
    
    return return_model_state


