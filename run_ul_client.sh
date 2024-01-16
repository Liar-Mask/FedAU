

# ul_whole_client

## ul_samples_whole_client
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

proportion=0.05
gamma=0.5
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user  --ul_client_gamma $gamma


## amnesiac ul client
model=alexnet
dataset=cifar10
seed=1  # 完全W2，acc 20%
seed=2  # W2'=0.5 W1+0.5 W2, gamma=0.5  
seed=3  # W2'=0.75 W1+0.25 W2, gamma=0.75
seed=4  # pro=0.02
seed=0
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1
ul_mode='amnesiac_ul_samples_client'

proportion=0.02
# 单卡
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

proportion=0.1
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &

## federaser ul client

model=alexnet
dataset=cifar10
seed=0
num_users=10
samples_per_user=5000
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1
ul_mode='federaser_ul_samples_client'

proportion=0.02
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users 1 --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_client/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user &