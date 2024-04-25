# FedAU-Repository

### This is the official pytorch implementation of the paper:

- **Unlearning during Learning: An Efficient Federated Machine Unlearning Method**
- **Accepted by IJCAI 2024**


## Description

In recent years, Federated Learning (FL) has garnered significant attention as a distributed machine learning paradigm. To facilitate the implementation of the right to be forgotten, the concept of federated machine unlearning (FMU) has also emerged. However, current FMU approaches often involve additional time-consuming steps and may not offer comprehensive unlearning capabilities, which renders them less practical in real FL scenarios.

In this paper, we introduce FedAU, an innovative and efficient FMU framework aimed at overcoming these limitations. Specifically, FedAU incorporates a lightweight auxiliary unlearning module into the learning process and employs a straightforward linear operation to facilitate unlearning. This approach eliminates the requirement for extra time-consuming steps, rendering it well-suited for FL.Furthermore, FedAU exhibits remarkable versatility. It not only enables multiple clients to carry out unlearning tasks concurrently but also supports unlearning at various levels of granularity, including individual data samples, specific classes, and even at the client level.We conducted extensive experiments on MNIST, CIFAR10, and CIFAR100 datasets to evaluate the performance of FedAU. The results demonstrate that FedAU effectively achieves the desired unlearning effect while maintaining model accuracy.

<<<<<<< HEAD
<img src="https://raw.githubusercontent.com/Liar-Mask/Photos/main/img/image-20240425121110101.png" alt="FedAU Scheme" style="zoom: 67%;" />
=======
<img src="https://raw.githubusercontent.com/Liar-Mask/Photos/main/img/image-20240425121110101.png" alt="FedAU Scheme" style="zoom: 67%;" />
>>>>>>> 52a919885f503b4f726c019560553e41c88b4da4
