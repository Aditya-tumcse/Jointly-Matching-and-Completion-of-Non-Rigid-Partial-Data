import torch
import torchvision
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader_hg import BalletDancer,GoalKeeper
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import wandb
import configuration
from wandb_log import training_log
from models.hourglassnet import HourGlassNet

def wandb_initiliazer(arguments):
    with wandb.init(project="TDCV_2_hg", config=arguments):
        config = wandb.config

        model, train_loader,val_loader,loss_fn,loss_fn1, optimizer = nn_model(config)

        train(model, train_loader, val_loader, loss_fn,loss_fn1, optimizer, config)
    return model


def nn_model(config):
    data_transforms = transforms.Compose([transforms.ToTensor()])


    train_set = BalletDancer(data_transforms)
    validation_set = GoalKeeper(data_transforms)

    #Loading train and validation set
    train_set_loader = DataLoader(train_set,batch_size=config.batch_size,shuffle=False,num_workers=configuration.training_configuration.number_workers)
    validation_set_loader = DataLoader(validation_set,batch_size=config.batch_size,shuffle=False,num_workers=configuration.training_configuration.number_workers)

    #Build the model
    net = HourGlassNet()

    if configuration.training_configuration.device.type == 'cuda':
        net.cuda()

   
    loss_function = torch.nn.MSELoss()
    loss_function1 = torch.nn.MSELoss(reduce=None)

    optimizer = torch.optim.Adam(net.parameters(),lr=config.lr)
    
    return net,train_set_loader,validation_set_loader,loss_function,loss_function1,optimizer

def validation_phase(NN_model,val_set_loader,loss_function,loss_function1,epoch):
    print("Validating...")
    NN_model.eval()
    
    loss_val = 0

    for batch_id, (patch1,x,patch1_target) in enumerate(val_set_loader, 1):
        if(configuration.training_configuration.device.type == 'cuda'):
            patch1,patch1_target = patch1.cuda(),patch1_target.cuda()
        else:
            patch1,patch1_target = patch1,patch1_target

        output = NN_model(patch1.float())
        loss = loss_function(output.float(),patch1_target.float())

        mini_batches += 1
        loss_val += float(loss)


        loss_l = []
        if(epoch == configuration.training_configuration.train_number_epochs - 1):
            loss1 = torch.zeros(2)
            i = 0
            for i in range(2):
                loss1[i] = loss_function1(output[i].float(),patch1_target[i].float())
                output_np = output[i].float()

                loss_l.append((loss1[i],output_np,x[0][i],x[1][i],x[2][i]))


        if(epoch == configuration.training_configuration.train_number_epochs - 1):
            y = max(loss_l)
            op = y[1].cpu()
            output = op.detach().numpy()

            print("outputfile_" + str(y[2]) + "_" + str(y[3]) + "_" + str(y[4]))
            np.savez_compressed("/rhome/hyu/tdcv_code/a_code/TDCV-Project-2/output1/outputfile_" + str(y[2]) + "_" + str(y[3]) + "_" + str(y[4]) + ".npz",output=output)

    loss_val = loss_val/mini_batches
    return loss_val

def train(NN_model,train_set_loader,val_set_loader,loss_function,loss_function1,optimizer, config):
    wandb.watch(NN_model,loss_function,log='all',log_freq=50)

    num_seen_samples = 0
    mini_batches = 0
    loss_value = 0

    for epoch in range(config.epochs):
        for batch_id, (patch1,patch1_target) in enumerate(train_set_loader, 1):
            NN_model.train()
            if(configuration.training_configuration.device.type == 'cuda'):
                patch1, patch1_target = patch1.cuda(), patch1_target.cuda()
            else:
                patch1, patch1_target = patch1, patch1_target


            output = NN_model(patch1.float())
            loss = loss_function(output.float(),patch1_target.float())

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
                val_loss = validation_phase(NN_model,val_set_loader,loss_function,loss_function1,epoch)
                training_log(loss_value/configuration.training_configuration.plot_frequency, mini_batches)
                training_log(val_loss,mini_batches,False)

                PATH = "model.pt"
                torch.save({'epoch':epoch,'model_state_dict':NN_model.state_dict(),'optimimizer_state_dict':optimizer.state_dict(),'loss':loss_value},PATH)

                loss_value = 0

            print('Epoch-{0} lr: {1:f}'.format(epoch, optimizer.param_groups[0]['lr']))

