import numpy as np
import os
import random
import torch
import configuration
from torch.utils.data import Dataset

class TrainSet(Dataset):
    def __init__(self,transform=None):
        super(TrainSet,self).__init__()
        self.transform = transform
        self.idx = 0
        self.train_set = self.load_patch()

    def load_patch(self):
        rootdir = configuration.training_configuration.training_set_dir
        path, dirs, files = next(os.walk(rootdir))
        data_set = {}
        for i in range(len(dirs)):
            data_set[i] = []

        for i in range(len(dirs)):
            for j in range(configuration.Training_Data_Config.number_keypoints):
                data_set[i].append(j)
            self.idx += 1
        
        print("Data loading complete")
        return data_set


    def __len__(self):
        return configuration.Training_Data_Config.training_data_size

    def __getitem__(self):
        rootdir = configuration.training_configuration.training_set_dir
        p_num = random.randrange(1746411,1747343,5)
        patch = []
        for i in range(configuration.Training_Data_Config.number_keypoints):
            patch.append(np.load(rootdir + str(p_num) + "_gt_patches/tdf_patches_" + str(i) + ".npz")['patch'])

        if self.transform:
            patch = torch.tensor([patch])

        return patch
