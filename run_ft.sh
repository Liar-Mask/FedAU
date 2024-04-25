model=alexnet
dataset=cifar10

num_users=10
lr_up=common
lr=0.005
epochs=100
local_ep=2
num_ul_users=1

ul_mode='ul_class'
proportion=1.0  # 好像没作用，如果有，应取1.0
ul_beta=1.0
CUDA_VISIBLE_DEVICES=1 python unlearn_pretrain.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_finetune/${ul_mode} \
 --lr $lr --lr_up $lr_up 