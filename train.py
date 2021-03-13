import torch
import torchvision
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader import BalletDancer,GoalKeeper
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import wandb
import resnet
import siamese
import loss
import config

def wandb_initiliazer(arguments):
    with wandb.init(project="TDCV_2", config=arguments):
        config1 = wandb.config

        model, train_loader,val_loader,loss_fn, optimizer = nn_model(config1)

        train(model, train_loader, val_loader, loss_fn, optimizer, config)
    return model


def nn_model(config1):
    data_transforms = transforms.Compose([transforms.ToTensor()]) 

    
    train_set = BalletDancer(data_transforms) 
    validation_set = GoalKeeper(data_transforms)
    
    #Loading train and validation set
    train_set_loader = DataLoader(train_set,batch_size=config.Config.train_batch_size,shuffle=False,num_workers=config.Config.number_workers)
    validation_set_loader = DataLoader(validation_set,batch_size=config.Config.train_batch_size,shuffle=false,num_workers=config.Config.number_workers)
    
    #Build the model
    intermediate_net = resnet.generate_model(config.Config.resnet_depth)
    siamese_net = siamese.Siamese(intermediate_net)

    loss_function = loss.ContrastiveLoss(config.Config.contrastive_margin)

    if config.Config.device.type == 'cuda':
        siamese_net.cuda()

    
    optimizer = torch.optim.Adam(siamese_net.parameters(),lr=config.Config.learning_rate)
    return siamese_net,train_set_loader,validation_set_loader,loss_function,optimizer

def validation_phase(NN_model,val_set_loader,loss_function):
    NN_model.eval()

    mini_batches = 0
    loss_val = 0

    for batch_id, (patch1, patch2, label) in enumerate(val_set_loader, 1):
        if(config.Config.device.type == 'cuda'):
            patch1, patch2, label = patch1.cuda(), patch2.cuda(), label.cuda()
        else:
            patch1, patch2, label = patch1,patch2,label

        output1,output2 = NN_model(patch1.float(),patch2.float())
        distances = (output2 - output1).pow(2).sum(1)
        loss = loss_function(distances,label)

        num_seen_samples += len(patch1)
        mini_batches += 1
        loss_val += float(loss)

    return loss_val/mini_batches
    
def train(NN_model,train_set_loader,val_set_loader,loss_function,optimizer,config):
    wandb.watch(NN_model,loss_function,log='all',log_freq=50)
    
    num_seen_samples = 0
    mini_batches = 0
    loss_value = 0

    for epoch in range(0,config.Config.train_number_epochs):
        for batch_id, (patch1, patch2, label) in enumerate(train_set_loader, 1):
            NN_model.train()
            if(config.Config.device.type == 'cuda'):
                patch1, patch2, label = patch1.cuda(), patch2.cuda(), label.cuda()
            else:
                patch1, patch2, label = patch1,patch2,label
        
            
            output1,output2 = NN_model(patch1.float(),patch2.float())
            distances = (output2 - output1).pow(2).sum(1)
            loss = loss_function(distances,label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_seen_samples += len(patch1)
            mini_batches += 1
            loss_value += float(loss)

            if (mini_batches % 200) == 0:
                print("Training loss after %d batches"%(int(num_seen_samples/config.Config.train_batch_size)))

            #Plotting in wand b
            if (mini_batches % config.Config.plot_frequency == 0):
                val_loss = validation_phase(NN_model,val_set_loader,loss_function)

                training_log(loss_value,mini_batches)
                training_log(val_loss,mini_batches,False)
                loss_val = 0
            
def training_log(loss,iteration,NN_train = True):
    if NN_train:
        wandb.log({"Traning loss:",float(loss)},step=iteration)
        print("Training loss after " + str(iteration) + "iterations:" + str(float(loss)))
    else:
        wandb.log({"Validation loss:",float(loss)},step=iteration)
        print("Validation loss after " + str(iteration) + "iterations:" + str(float(loss)))

