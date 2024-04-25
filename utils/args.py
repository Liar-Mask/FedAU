import argparse

def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= federated learning parameters ========================
    parser.add_argument('--seed', type=int, default=0,
                        help="exp mark")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--ul_mode', choices=['none',
                                              'ul_samples', 'ul_samples_backdoor', 'retrain_samples',
                                              'ul_class','retrain_class',
                                              'amnesiac_ul_samples','amnesiac_ul_class','amnesiac_ul_samples_client',
                                              'federaser_ul_samples','federaser_ul_samples_client',
                                              'ul_samples_whole_client','retrain_samples_client'
                                              ],
                         default='ul_class', type=str,
                         help='which unlearning scheme we use') 
    parser.add_argument('--num_ul_users', type=int, default=1,
                        help="number of unlearning users")
    parser.add_argument('--ul_class_id', type=int, default=9,
                        help="id of unlearned class")
    parser.add_argument('--samples_per_user', type=int, default=5000,
                        help="number of users: K")
    parser.add_argument('--persample_bs', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--defense', type=str, default="none",
                        help="defense scheme")
    parser.add_argument('--d_scale', type=float, default=0.0,
                        help="number of users: K")
    parser.add_argument('--save_dir', type=str, default='../FedUL_test/',
                        help='saving path')
    parser.add_argument('--log_folder_name', type=str, default='/training_log_correct_iid/',
                        help='saving path')
    parser.add_argument('--proportion', type=float, default=0.00,
                        help="the proportion of UL samples")
    parser.add_argument('--frac', type=float, default=1,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr_outer', type=float, default=1,
                        help="learning rate")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate for inner update")
    parser.add_argument('--lr_up', type=str, default='milestone',
                        help='optimizer: [common, milestone, cosine]')
    parser.add_argument('--schedule_milestone', type=list, default=[225,325],
                         help="schedule lr")
    parser.add_argument('--gamma', type=float, default=0.99,
                         help="exponential weight decay")
    parser.add_argument('--iid',type=int, default=1,
                        help='dataset is split iid or not')
    parser.add_argument('--fine_tune_mode',type=int, default=0,
                        help='the model is fine-tuned or not')
    parser.add_argument('--beta', type=float, default=1,
                        help='Non-iid Dirichlet param')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='communication round')
    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                         default='uniform', type=str,
                         help='which kind of client sampling we use') 
    parser.add_argument('--data_augment', action='store_true', default=True,
                        help='data_augment')
    parser.add_argument('--lira_attack', action='store_true', default=True,
                        help='lira_attack')
    parser.add_argument('--cosine_attack', action='store_true', default=True,
                        help='cosine_attack')
    parser.add_argument('--class_prune_sparsity', type=float, default=0.05,
                        help='class_prune_sparsity')
    parser.add_argument('--class_prune_target', type=int, default=9,
                        help='class_prune_target')
    parser.add_argument('--ul_client_gamma', type=float, default=0.5,
                        help='ul_client_gamma')
    parser.add_argument('--ul_samples_alpha', type=float, default=0.9,
                        help='ul_samples_alpha')
    
    
    # ============================ Model arguments ===================================
    parser.add_argument('--model_name', type=str, default='alexnet', choices=['lenet','alexnet', 'resnet','resnet18','resnet34'],  #, 'resnet20','ResNet18'
                        help='model architecture name')
    
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    
    parser.add_argument('--data_root', default='/CIS32/zgx/Fed2/Data',
                        help='dataset directory')
    parser.add_argument('--pretrain_model_root', default='/CIS32/zgx/Unlearning/FedUnlearning/log_test_pretrain',
                        help='the saved pre-trained model directory')

    # =========================== Other parameters ===================================
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--bp_interval', default=30, type=int, help='interval for starting bp the local part')
    parser.add_argument('--log_interval', default=1, type=int,
                        help='interval for evaluating loss and accuracy')

    parser.add_argument("--sigma_sgd",
        type=float,
        default=0.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
    "--grad_norm",
    type=float,
    default=1e4,
    help="Clip per-sample gradients to this norm",
    )

  
    # =========================== IPR parameters ===================================
    
    parser.add_argument('--norm-type', default='bn', choices=['bn', 'gn', 'in', 'none'],
                        help='norm type (default: bn)')
    parser.add_argument('--key-type', choices=['random', 'image', 'shuffle'], default='shuffle',
                        help='passport key type (default: shuffle)')
    # signature argument

    parser.add_argument('--num_sign', type=int, default=0,
                        help="number of signature users: K")

    parser.add_argument('--weight_type', default='gamma', choices=['gamma', 'kernel'],
                        help='weight-type (default: gamma)')
    
    parser.add_argument('--num_bit', type=int, default=0,
                        help="number of signature bits")

    parser.add_argument('--loss_type', default='sign', choices=['sign', 'CE'],
                        help='loss type (default: sign)')

    parser.add_argument('--loss_alpha', type=float, default= 0.2,
                        help='sign loss scale factor to trainable (default: 0.2)')

    # backdoor argument 
    parser.add_argument('--backdoor_indis', action='store_false', default=True,
                        help='backdoor in distribution')
    parser.add_argument('--num_back', type=int, default=0,
                        help="number of backdoor users: K")
    parser.add_argument('--num_trigger', type=int, default=0,
                        help="number of signature bits")

    # paths
    parser.add_argument('--passport-config', default='passport_configs/alexnet_passport.json',
                        help='should be same json file as arch')

    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')

    # =========================== DP ===================================
    parser.add_argument('--dp', action='store_true', default=False,
                        help='whether dp')

    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')



    # =========================== Robustness ===================================
    parser.add_argument('--pruning', action='store_true')
    parser.add_argument('--percent', default=5, type=float)

    # parser.add_argument('--im_balance', action='store_true', default=False,
    #                     help='whether im_balance')
    
    args = parser.parse_args()

    return args
