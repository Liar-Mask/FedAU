


# ul_sample of lenet MNIST

model=alexnet
dataset=cifar10

model=resnet18
dataset=cifar100
seed=19
seed=3 #alpha=0.8
seed=4 # resnet18  bacth_size=128
seed=5 # resnet18 bs=128, alpha=0.8
seed=7 # resnet18  bacth_size=128, learn based W1
seed=8 # resnet18  bacth_size=128, learn based W1 , correct labels
num_users=10
lr_up=common
lr=0.01
epochs=200
local_ep=2
num_ul_users=1

ul_mode='ul_samples_backdoor'
# ul_mode='retrain_samples'

batch_size=128
proportion=0.01
alpha=0.9
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs --batch_size $batch_size\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_backdoor/${ul_mode}/${proportion} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

proportion=0.05
num_ul_users=5
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}_multi/${proportion}_${num_ul_users} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha 0.9 &



proportion=0.08
num_ul_users=8-seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
CUDA_VISIBLE_DEVICES=2 python main_zgx.py -
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}_multi/${proportion}_${num_ul_users} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha 0.9 &

proportion=0.10
num_ul_users=10
CUDA_VISIBLE_DEVICES=1 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}_multi/${proportion}_${num_ul_users} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha 0.9 &

proportion=0.01
alpha=0.7
CUDA_VISIBLE_DEVICES=4 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

proportion=0.01
alpha=0.6
CUDA_VISIBLE_DEVICES=5 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

proportion=0.01
alpha=0.99
CUDA_VISIBLE_DEVICES=2 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

proportion=0.01
alpha=0.5
CUDA_VISIBLE_DEVICES=3 python main_zgx.py --seed $seed --num_users $num_users --dataset $dataset --model_name $model --epochs $epochs\
 --proportion $proportion --num_ul_users $num_ul_users --ul_mode $ul_mode --local_ep $local_ep --log_folder_name log_test_samples/${ul_mode}/${proportion}_${alpha} \
 --lr $lr --lr_up $lr_up --ul_samples_alpha $alpha &

# Retrain samples
model=lenet
dataset=mnist
epochs=100

seed=0
num_users=10
lr_up=common
lr=0.01

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

