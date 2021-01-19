import os
import numpy as np
import matplotlib.pyplot as plt
import camera as cam
import model_data as md

def camera_orientation():
    rootdir = "/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/"
    path, dirs, files = next(os.walk(rootdir))
    files = sorted(files)

    for i in range(1,2):
        
        cam_pos = cam.random_camera_position(rootdir + files[i])
        cam_pos = cam_pos[0]
        vec_x = np.array([1,0,0])
        vec_y = np.array([0,1,0])
        vec_z = np.array([0,0,1])

        
        cam_pos_x = cam_pos + vec_x
        cam_pos_x = np.append(cam_pos_x,1)
        cam_pos_y = cam_pos + vec_y
        cam_pos_y = np.append(cam_pos_y,1)
        cam_pos_z = cam_pos + vec_z
        cam_pos_z = np.append(cam_pos_z,1)

        params = []
        params.append(cam_pos)
        params.append(cam_pos_x)
        params.append(cam_pos_y)
        params.append(cam_pos_z)
        
        md.write_ply_ascii(files[i],params,4)
        
        
