import torch
import os
import numpy as np
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader


class BalletDancer():
    def __init__(self,transform=None):
        super(BalletDancer,self).__init__()
        np.random.seed(0)
        self.data,self.num_kps = self.load_patch()
        self._get_patch_path_()
        self.transform = transform

    def load_patch(self):
        rootdir = "/home/aditya/PycharmProjects/OpenCV-python/Project_2/patch_files/"
        path, dirs, files = next(os.walk(rootdir))
        dirs = sorted(dirs)
        data_set = []
        idx = 0
        visible_key_point = np.load("/home/aditya/PycharmProjects/OpenCV-python/Project_2/TDCV-Project-2/kp_visibility.npz",allow_pickle=True)['visibility']
        for i in range(int(min(dirs)),int(max(dirs))):
            cam_num = os.listdir(rootdir + str(i))
            for cam in range(len(cam_num)):
                visible_kp = visible_key_point[i][cam]
                data_set[visible_kp] = []
                for j in range(len(visible_kp)):
                    key_point = visible_kp[j]
                    data_set[key_point].append((i,cam))
                    idx += 1
            i += 1
        return(data_set,idx)
        
        
    def _get_patch_path_(self,frame_num,cam_num,kp_num):
        rootdir = "/home/aditya/PycharmProjects/OpenCV-python/Project_2/patch_files/"
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

    
    

    


            
            
        


#a = np.load("/home/aditya/PycharmProjects/OpenCV-python/Project_2/patch_files/tsdf_patch_0.npz")["patch"]
#a = BalletDancer.loadToMem("/home/aditya/PycharmProjects/OpenCV-python/Project_2/patch_files/1746411/")

if __name__ == '__main__':
    BalletDancer()    