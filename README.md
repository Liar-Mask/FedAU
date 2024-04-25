# FedAU-Repository

### This is the official pytorch implementation of the paper:

- **Unlearning during Learning: An Efficient Federated Machine Unlearning Method**
- **Accepted by IJCAI 2024**


## Description

In recent years, Federated Learning (FL) has garnered significant attention as a distributed machine learning paradigm. To facilitate the implementation of the right to be forgotten, the concept of federated machine unlearning (FMU) has also emerged. However, current FMU approaches often involve additional time-consuming steps and may not offer comprehensive unlearning capabilities, which renders them less practical in real FL scenarios.

In this paper, we introduce FedAU, an innovative and efficient FMU framework aimed at overcoming these limitations. Specifically, FedAU incorporates a lightweight auxiliary unlearning module into the learning process and employs a straightforward linear operation to facilitate unlearning. This approach eliminates the requirement for extra time-consuming steps, rendering it well-suited for FL.Furthermore, FedAU exhibits remarkable versatility. It not only enables multiple clients to carry out unlearning tasks concurrently but also supports unlearning at various levels of granularity, including individual data samples, specific classes, and even at the client level.We conducted extensive experiments on MNIST, CIFAR10, and CIFAR100 datasets to evaluate the performance of FedAU. The results demonstrate that FedAU effectively achieves the desired unlearning effect while maintaining model accuracy.

<img src="https://raw.githubusercontent.com/Liar-Mask/Photos/main/img/image-20240425121110101.png" alt="FedAU Scheme" style="zoom: 67%;" />

## Getting started

### Preparation

Before executing the project code, please prepare the Python environment according to the `requirement.txt` file. We set up the environment with `python 3.8` and `torch 1.8.1`. 


### How to run

We conduct experiments on three datasets: MNIST,CIFAR10 and CIFAR100. We adopt LeNet  for conducting experiments on MNIST and adopt AlexNet on CIFAR10 and ResNet18 on CIFAR100. The last layer of the model is treated as the auxiliary unlearning module.

There three unlearning modes supported by FedAU:

**1. Unlearn Samples**

```python
python main_zgx.py --num_users 10 --dataset cifar10 --model_name alexnet --epochs 200 --batch_size 128 \
 --proportion 0.01 --num_ul_users 1 --ul_mode 'ul_samples_backdoor' --local_ep 2 --log_folder_name ul_samples/
```

**2. Unlearn Class**

```python
python main_zgx.py --num_users 10 --dataset cifar10 --model_name alexnet --epochs 200 --batch_size 128 \
 --num_ul_users 1 --ul_mode 'ul_class' --local_ep 2 --log_folder_name ul_class/
```

**3. Unlean Client**

```python
python main_zgx.py --num_users 10 --dataset cifar10 --model_name alexnet --epochs 200 --batch_size 128 \
 --num_ul_users 1 --ul_mode 'ul_class' --local_ep 2 --log_folder_name ul_client/
```

