import torch
import torchvision
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader import BalletDancer
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import gflags
import resnet
import siamese
import loss
import config

if __name__ == 'main':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda",True,"use cuda")
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy") #Size of validation set
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers") #number of CPU cores
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.") 
    gflags.DEFINE_integer("save_every", 100, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 100, "test model after each test_every iter.")
    gflags.DEFINE_string("model_path", "Provide path to the Siamese model", "path to store model") #Path to the neural network model
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train") #Number of GPU cores
    
    Flags(sys.argv)

    #data_transforms = transforms.Compose([transforms.ToTensor()]) #data augmentation.Here converting the input data into a tensor

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")

    
    train_set = BalletDancer() 
    #validation_set = GoalKeeper(config.Config.validation_set_dir,transform=data_transforms,times=Flags.times,way=Flags.way)
    
    #Loading train and validation set
    train_set_loader = DataLoader(train_set,batch_size=config.Config.train_batch_size,shuffle=False,num_workers=Flags.workers)
    #validation_set_loader = DataLoader(validation_set,batch_size=config.Config.train_batch_size,shuffle=false,num_workers=Flags.workers)

    intermediate_net = resnet.generate_model(config.Config.resnet_depth)
    siamese_net = siamese.Siamese(intermediate_net)

    loss_function = loss.ContrastiveLoss(config.Config.contrastive_margin)

    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        siamese_net = torch.nn.DataParallel(siamese_net)

    if Flags.cuda:
        siamese_net.cuda()

    siamese_net.train()
    optimizer = torch.optim.Adam(siamese_net.parameters(),lr=config.Config.learning_rate)
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    iteration_number = 0
    counter = []

    for epoch in range(0,config.Config.train_number_epochs):
        for batch_id, (img1, img2, label) in enumerate(train_set_loader, 1):
            if(Flags.cuda):
                img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
            else:
                img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        
            optimizer.zero_grad()
            output1,output2 = siamese_net(img1,img2)
            loss = loss_function(output1,output2,label)
            loss_val += loss
            loss.backward()
            optimizer.step()
            
            if(batch_id % Flags.show_every == 0):
                print("epoch number: {}\t Current loss: {}\n".format(epoch,loss.item()))
                iteration_number += 10
                counter.append(iteration_number)
                train_loss.append(loss.item())
    plt.plot(counter,train_loss)
    plt.show()
