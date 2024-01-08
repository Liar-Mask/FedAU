#!/bin/bash


# unlearn backdoor samples 

model=alexnet
dataset=cifar10
seed=19  # seed=29-->aplha=0.9; 19-->alpha=0.99
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=10

ul_mode='ul_samples_backdoor'

proportion=0.002
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

proportion=0.005
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &


proportion=0.005
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

proportion=0.01
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &


proportion=0.02
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

# UL3+UL1 0.01
# UL2 0.005
# UL4 0.002
# UL5 0.02
model=alexnet
dataset=cifar10
seed=0
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=10

ul_mode='retrain_class'  #'random'

CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/multi_client/$ul_mode \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user 

ul_mode='retrain'
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_forth/prpt$proportion/retrain --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &






#UL2   UL集中指定client
model=alexnet
dataset=cifar10
seed=9
num_users=10
samples_per_user=5000
lr_up=common
lr=0.05
epochs=300
proportion=0.00

CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --log_folder_name log_test_sec/prpt$proportion --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user



# Correct Resnet iid
 CUDA_VISIBLE_DEVICES=0 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine' \
 --save_dir '../tmp_resnet_iid_lr01' --iid \
 --log_folder_name '/training_log_correct_iid/' &

  CUDA_VISIBLE_DEVICES=1 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp_resnet_iid_adam' --iid \
 --log_folder_name '/training_log_correct_iid/' &

 CUDA_VISIBLE_DEVICES=2 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
 --save_dir '../tmp_alexnet_iid_lr01' --iid \
 --log_folder_name '/training_log_correct_iid/' &

  CUDA_VISIBLE_DEVICES=3 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
 --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
 --save_dir '../tmp_alexnet_iid_adam' --iid \
 --log_folder_name '/training_log_correct_iid/' &
 


#  CUDA_VISIBLE_DEVICES=0 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine' \
#  --save_dir '../tmp_resnet_iid_lr01' --iid \
#  --log_folder_name '/training_log_correct_iid/' &

#   CUDA_VISIBLE_DEVICES=1 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name ResNet18\
#  --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_resnet_iid_adam' --iid \
#  --log_folder_name '/training_log_correct_iid/' &

#  CUDA_VISIBLE_DEVICES=2 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.1 --batch_size 100 --optim 'sgd' --lr_up 'cosine'\
#  --save_dir '../tmp_alexnet_iid_lr01' --iid \
#  --log_folder_name '/training_log_correct_iid/' &

#   CUDA_VISIBLE_DEVICES=3 python main_zgx.py --num_back 0 --num_trigger 0 --num_sign 10 --num_bit 0 --num_users 10 --dataset cifar100 --model_name alexnet\
#  --epochs 300 --local_ep 1 --lr 0.001 --batch_size 100 --optim 'adam' --lr_up 'cosine'\
#  --save_dir '../tmp_alexnet_iid_adam' --iid \
#  --log_folder_name '/training_log_correct_iid/' &
 

