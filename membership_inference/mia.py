import membership_inference
from membership_inference import attack, train, model
import torch
import torch.nn as nn

def mia_prob(unlearn_model,retain_train_dl, retain_valid_dl, forget_train_dl, forget_valid_dl, valid_dl, device):
    # Prepare attack dataset
    # Retain dataset to train attack model
    # Forget dataset used to evaluate the trained attack model
    shadow_trainX, shadow_trainY = train.prepare_attack_data(model=unlearn_model, iterator=retain_train_dl, device=device,
                                                             top_k=False)
    shadow_validX, shadow_validY = train.prepare_attack_data(model=unlearn_model, iterator=valid_dl, device=device,
                                                             top_k=False, test_dataset=True)

    shadowX = shadow_trainX + shadow_validX
    shadowY = shadow_trainY + shadow_validY
    attack_dataset = (shadowX, shadowY)

    forget_trainX, forget_trainY = train.prepare_attack_data(model=unlearn_model, iterator=forget_train_dl, device=device,
                                                             top_k=False)

    forgetX = forget_trainX
    forgetY = forget_trainY

    ###################################
    # Attack Model Training
    ##################################
    # The input dimension to MLP attack model
    input_size = shadowX[0].size(1)
    print('Input Feature dim for Attack Model : [{}]'.format(input_size))

    n_hidden = 128 #128 original
    out_classes = 2  # attack model
    attack_model = membership_inference.model.AttackMLP(input_size, n_hidden, out_classes).to(device)

    # Loss and optimizer
    attack_loss = nn.CrossEntropyLoss()
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001, weight_decay=1e-7)
    attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer, gamma=0.96)

    attack_valacc = train.train_attack_model(model=attack_model, dataset=attack_dataset,
                                             criterion=attack_loss, optimizer=attack_optimizer,
                                             lr_scheduler=attack_lr_scheduler,
                                             device=device, epochs=1, b_size=32, num_workers=1, verbose=False,
                                             earlystopping=False)

    attack_testacc = attack.attack_inference(model=attack_model, test_X=forgetX, test_Y=forgetY, device=device)

    return attack_valacc, attack_testacc