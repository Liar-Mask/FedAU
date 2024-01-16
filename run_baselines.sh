
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


ul_mode='none'

seed=4  # the pkl file is related to seed 3
proportion=1.0
ul_class=9
CUDA_VISIBLE_DEVICES=3 python unlearn_after_learn.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/${ul_mode} \
 --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user --class_prune_target $ul_class &
# for sparsity in $(seq 0 0.05 1)
# do
# for ul_class in {0..9}
# do
# CUDA_VISIBLE_DEVICES= ($sparsity * 100 )$ 4 python unlearn_after_learn.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
#  --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_class/${ul_mode} \
#  --lr $lr --lr_up $lr_up --samples_per_user $samples_per_user --class_prune_sparsity $sparsity --class_prune_target $ul_class &
# wait
# done