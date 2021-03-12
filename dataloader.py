import torch
import os
import numpy as np
import random
import numpy as np
import config
from torch.utils.data import Dataset,DataLoader


class BalletDancer():
    def __init__(self,transform=None):
        super(BalletDancer,self).__init__()
        np.random.seed(0)
        self.idx = 0
        self.data = self.load_patch()
        self.transform = transform

    def load_patch(self):
        rootdir = config.Config.training_set_dir
        path, dirs, files = next(os.walk(rootdir))
        dirs = sorted(dirs)
        data_set = self.data_initializer()
        
        visible_key_point = np.load(rootdir + "/" + "kp_visibility.npz",allow_pickle=True)['visibility']
        k = 0
        for i in range(int(min(dirs)),int(max(dirs)),config.Training_Data_Config.stride):
            cam_num = os.listdir(rootdir + str(i))
            for cam in range(len(cam_num)):
                visible_kp = visible_key_point[i][cam] + str(i))
            for cam in range(len(cam_num)):
                visible_kp = visible_key_point[k][cam]
                for j in range(len(visible_kp)):
                    key_point = visible_kp[j]
                    data_set[key_point].append((i,cam))
                    self.idx += 1
            k += 1
        return data_set
        
    def data_initializer(self):
        data_set = {}
        for n in range(config.Training_Data_Config.number_keypoints):
            data_set[n] = []
        return data_set
        
    def _get_patch_path_(self,frame_num,cam_num,kp_num):
        rootdir = config.Config.training_set_dir
        return(rootdir + str(frame_num) + "/" + str(cam_num) + "/" + "tsdf_path_" + str(kp_num))

    def _get_item_(self,index):
        #get patch from same classs
        if(index % 2 == 1):
            label = 1
            kp = random.randint(0,self.num_kps - 1)
            frame1,cam1 = random.choice(self.data[kp])
            patch1 = np.load(self._get_patch_path_(frame1,cam1,kp))['patch']
            frame2,cam2 = random.choice(self.data[kp])
            patch2 = np.load(self._get_patch_path_(frame2,cam2,kp))['patch']
        #get patch from different class
        else:
            label = 0
            kp_1 = random.randint(0,self.num_kps - 1)
            kp_2 = random.randint(0,self.num_kps - 1)
            while kp_1 == kp_2:
                kp_2 = random.randint(0,self.num_kps - 1)
            frame1,cam1 = random.choice(self.data[kp_1])
            patch1 = np.load(self._get_patch_path_(frame1,cam1,kp_1))['patch']
            frame2,cam2 = random.choice(self.data[kp_2])
            patch2 = np.load(self._get_patch_path_(frame1,cam1,kp_2))['patch'

        if self.transform:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)
        
        return(patch1,patch2,torch.from_numpy(np.array([label],dtype=np.float32)))