import torch
import torchvision
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader_resnet import BalletDancer,GoalKeeper
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import wandb
from models import resnet,siamese
import loss
import configuration
from wandb_log import training_log


def wandb_initiliazer(arguments):
    with wandb.init(project="TDCV_2", config=arguments):
        config = wandb.config

        model, train_loader,val_loader,loss_fn, optimizer = nn_model(config)

        train(model, train_loader, val_loader, loss_fn, optimizer, config)
    return model


    
def nn_model(config):
    data_transforms = transforms.Compose([transforms.ToTensor()])

    
    train_set = BalletDancer(data_transforms) 
    validation_set = GoalKeeper(data_transforms)
    
    #Loading train and validation set
    train_set_loader = DataLoader(train_set,batch_size=config.batch_size,shuffle=False,num_workers=configuration.training_configuration.number_workers)
    validation_set_loader = DataLoader(validation_set,batch_size=config.batch_size,shuffle=False,num_workers=configuration.training_configuration.number_workers)
    
    #Build the model
    intermediate_net = resnet.generate_model(config.resnet_depth)
    siamese_net = siamese.Siamese(intermediate_net)
    
    if configuration.training_configuration.device.type == 'cuda':
        siamese_net.cuda()

    loss_function = loss.ContrastiveLoss(config.loss_margin)

    
    optimizer = torch.optim.Adam(siamese_net.parameters(),lr=config.lr)
    
    return siamese_net,train_set_loader,validation_set_loader,loss_function,optimizer

def validation_phase(NN_model,val_set_loader,loss_function):
    print("Validating...")
    NN_model.eval()

    mini_batches = 0
    loss_val = 0
    acc_metric_1 = 0
    acc_metric_2 = 0
    acc_metric_3 = 0
    acc_metric_4 = 0

   
    for batch_id, (patch1, patch2, label,x,y,patch1_target,patch2_target) in enumerate(val_set_loader, 1):
        if(configuration.training_configuration.device.type == 'cuda'):
            patch1, patch2, label,patch1_target,patch2_target = patch1.cuda(), patch2.cuda(), label.cuda(),patch1_target.cuda(),patch2_target.cuda()
        else:
            patch1, patch2, label,patch1_target,patch2_target = patch1,patch2,label,patch1_target,patch2_target

        output = NN_model(patch1.float(),patch2.float())
        distances = (output[1] - output[0]).pow(2).sum(1)
        loss = loss_function(distances,label)

        acc_pred_1 = distances < 0.1
        acc_A = 1.0 * (acc_pred_1 == label)
        acc_metric_1 += acc_A.mean()

        acc_pred_2 = distances < 0.4
        acc_B = 1.0 * (acc_pred_2 == label)
        acc_metric_2 += acc_B.mean()

        acc_pred_3 = distances < 0.7
        acc_C = 1.0 * (acc_pred_3 == label)
        acc_metric_3 += acc_C.mean()

        acc_pred_4 = distances < 1.0
        acc_D = 1.0 * (acc_pred_4 == label)
        acc_metric_4 += acc_D.mean()

        mini_batches += 1
        loss_val += float(loss)

    accuracy = [acc_metric_1/mini_batches, acc_metric_2/mini_batches, acc_metric_3/mini_batches, acc_metric_4/mini_batches]
    loss_val = loss_val/configuration.Validation_Data_Config.validation_data_size
    return accuracy, loss_val
    
def train(NN_model,train_set_loader,val_set_loader,loss_function,optimizer,config):
    wandb.watch(NN_model,loss_function,log='all',log_freq=50)
    
    num_seen_samples = 0
    mini_batches = 0
    loss_value = 0
    acc_metric_1 = 0
    acc_metric_2 = 0
    acc_metric_3 = 0
    acc_metric_4 = 0

    for epoch in range(config.epochs):
        for batch_id, (patch1, patch2, label,patch1_target,patch2_target) in enumerate(train_set_loader, 1):
            NN_model.train()
            if(configuration.training_configuration.device.type == 'cuda'):
                patch1, patch2, label, patch1_target,patch2_target = patch1.cuda(), patch2.cuda(), label.cuda(), patch1_target.cuda(), patch2_target.cuda()
            else:
                patch1, patch2, label, patch1_target,patch2_target = patch1,patch2,label, patch1_target,patch2_target
        
            
            output = NN_model(patch1.float(),patch2.float())
            distances = (output[1] - output[0]).pow(2).sum(1)
            loss = loss_function(distances,label)

            acc_pred_1 = distances < 0.1
            acc_A = 1.0 * (acc_pred_1 == label)
            acc_metric_1 += acc_A.mean()

            acc_pred_2 = distances < 0.4
            acc_B = 1.0 * (acc_pred_2 == label)
            acc_metric_2 += acc_B.mean()

            acc_pred_3 = distances < 0.7
            acc_C = 1.0 * (acc_pred_3 == label)
            acc_metric_3 += acc_C.mean()

            acc_pred_4 = distances < 1.0
            acc_D = 1.0 * (acc_pred_4 == label)
            acc_metric_4 += acc_D.mean()

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_seen_samples += len(patch1)
            mini_batches += 1
            loss_value += float(loss)
            

            if (mini_batches % 200) == 0:
                print("Training loss after %d batches"%(int(num_seen_samples/configuration.training_configuration.train_batch_size)))

            #Plotting in wandb
            if (mini_batches % configuration.training_configuration.plot_frequency == 0):
                val_accuracy, val_loss = validation_phase(NN_model,val_set_loader,loss_function)
                accuracy = [acc_metric_1/configuration.training_configuration.plot_frequency, acc_metric_2/configuration.training_configuration.plot_frequency, acc_metric_3/configuration.training_configuration.plot_frequency, acc_metric_4/configuration.training_configuration.plot_frequency]
                training_log(loss_value/configuration.training_configuration.plot_frequency, accuracy, mini_batches)
                training_log(val_loss,val_accuracy,mini_batches,False)
                
                loss_value = 0
                acc_metric_1 = 0
                acc_metric_2 = 0
                acc_metric_3 = 0
                acc_metric_4 = 0
            
            
            print('Epoch-{0} lr: {1:f}'.format(epoch, optimizer.param_groups[0]['lr']))
            
            