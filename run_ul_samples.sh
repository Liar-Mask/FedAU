


# ul_sample of lenet MNIST

model=lenet
dataset=mnist
seed=0  # 84
seed=1  # 84 *4
seed=2  # 84 *10
seed=4 # multi fc
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

# Retrain samples
model=lenet
dataset=mnist

seed=0
num_users=10
lr_up=common
lr=0.01
epochs=100
local_ep=2
num_ul_users=1
ul_mode='retrain_samples'

proportion=0.01
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up 

proportion=0.005
# 单卡训练
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up  &

proportion=0.02
# 单卡训练
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up  &

# amnesiac

model=lenet
dataset=mnist

seed=1
num_users=10
lr_up=common
lr=0.01
epochs=100
local_ep=2
num_ul_users=1
ul_mode='amnesiac_ul_samples'

proportion=0.01
# 单卡训练
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up &

proportion=0.02

CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up &

proportion=0.005

CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up &

