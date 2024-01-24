
# Non-iid
## 1. ul samples
### 1.1 ul backdoor samples
model=alexnet
dataset=cifar10
seed=0 
num_users=10
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1
ul_mode='ul_samples_backdoor'
iid=0
beta=10

proportion=0.01
alpha=0.9
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_noniid${beta}/ul_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha --iid 0 --beta $beta &


### 1.2 ul 


