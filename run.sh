#!/bin/bash

# ul_sample of lenet MNIST

model=lenet
dataset=mnist
seed=0  # 84
seed=1  # 84 *4
seed=2  # 84 *10
seed=3 # multi fc
num_users=10
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1
ul_mode='ul_samples_backdoor'

proportion=0.01
alpha=0.9
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

proportion=0.02
alpha=0.9
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

proportion=0.005
alpha=0.9
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

# retrain client


model=alexnet
dataset=cifar10
seed=1
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1
ul_mode='retrain_samples_client'

proportion=0.02
# gamma=0.5
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user  &

proportion=0.05
# gamma=0.5
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user   &

proportion=0.1
# gamma=0.5
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user   &



# ul_whole_client


model=alexnet
dataset=cifar10
seed=1  # 完全W2，acc 20%
seed=2  # W2'=0.5 W1+0.5 W2, gamma=0.5  
seed=3  # W2'=0.75 W1+0.25 W2, gamma=0.75
seed=4  # pro=0.02
seed=2
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1
ul_mode='ul_samples_whole_client'

proportion=0.1
gamma=0.5
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user  --ul_client_gamma $gamma



# federaser scheme

model=alexnet
dataset=cifar10
seed=1  
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1
ul_mode='federaser_ul_samples'

proportion=0.005
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user


## amnesiac_ul_class

model=alexnet
dataset=cifar10
seed=1  # seed=29-->aplha=0.9; 19-->alpha=0.99
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=10



ul_mode='amnesiac_ul_class'
seed=1  #start=150 /10^2
seed=2  #start=180 /10^2
seed=3  #start=100 /10^2
proportion=1.0

ul_class_id=0
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=1
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=2
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=3
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=4
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=5
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=6
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=7
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

ul_class_id=8
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

for ul_class_id in {0..8}
do 
GPU_id= expr $ul_class_id % 4
echo $GPU_id
CUDA_VISIBLE_DEVICES=$GPU_id python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &
done


# 批量命令



# unlearn backdoor samples 
## amnesiac_ul_samples
model=alexnet
dataset=cifar10
seed=1  # seed=29-->aplha=0.9; 19-->alpha=0.99
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1

# ul_mode='ul_samples_backdoor'
ul_mode='amnesiac_ul_samples'



# seed=1 -->update *1/n * 1/10
seed=1
# seed=2 --> update *1/n  start epoch=190
seed=2
# seed=3 --> update *1/n  start epoch=190, minus update_list[0-9]
seed=3
seed=4  #--> ul_client_num==1
proportion=0.002
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

proportion=0.01
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &


proportion=0.02
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

proportion=0.005
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &


proportion=0.02
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &


# retrain class
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


# retrain_samples MNIST




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
 

