

# ul-class with all clients
model=lenet
dataset=mnist
seed=0 # seed=29-->aplha=0.9; 19-->alpha=0.99
num_users=10
lr_up=common
lr=0.01
epochs=100
local_ep=2
num_ul_users=10

ul_mode='ul_class'
proportion=1.0  # 好像没作用，如果有，应取1.0
ul_beta=1.0
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/${ul_mode} \
 --lr $lr --lr_up $lr_up &

## retrain class

model=lenet
dataset=mnist
seed=0 # seed=29-->aplha=0.9; 19-->alpha=0.99
num_users=10
lr_up=common
lr=0.01
epochs=100
local_ep=2
num_ul_users=10

ul_mode='retrain_class'
proportion=1.0  # 好像没作用，如果有，应取1.0
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/${ul_mode} \
 --lr $lr --lr_up $lr_up &


## amnesiac_ul_class

model=lenet
dataset=mnist
seed=0 # seed=29-->aplha=0.9; 19-->alpha=0.99
num_users=10
lr_up=common
lr=0.01
epochs=100
local_ep=2
num_ul_users=10

ul_mode='amnesiac_ul_class'
seed=0  #start=500 /10^2
proportion=1.0

CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/$ul_class_id/${ul_mode} \
 --lr $lr --lr_up $lr_up &