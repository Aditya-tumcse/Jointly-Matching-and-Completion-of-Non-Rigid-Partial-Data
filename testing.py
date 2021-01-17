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
        
        cam_pos,cam_extrinsic_mat = cam.camera_extrinsics(rootdir + files[i])
        cam_extrinsic_mat = cam_extrinsic_mat[0]
        cam_pos = cam_pos[0]
        vec_x = np.array([1,0,0])
        vec_y = np.array([0,1,0])
        vec_z = np.array([0,0,1])

        
        cam_pos_x = vec_x
        cam_pos_x = np.append(cam_pos_x,1)
        cam_pos_y = vec_y
        cam_pos_y = np.append(cam_pos_y,1)
        cam_pos_z = vec_z
        cam_pos_z = np.append(cam_pos_z,1)

        vec_xx = np.dot(np.linalg.inv(cam_extrinsic_mat),cam_pos_x)
        vec_yy = np.dot(np.linalg.inv(cam_extrinsic_mat),cam_pos_y)
        vec_zz = np.dot(np.linalg.inv(cam_extrinsic_mat),cam_pos_z)

        """
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(cam_pos[0],cam_pos[1],cam_pos[2],'r*')
        ax.scatter(cam_pos_x[0],cam_pos_x[1],cam_pos_x[1],'ko')
        ax.scatter(cam_pos_y[0],cam_pos_y[1],cam_pos_y[2],'bo')
        ax.scatter(cam_pos_z[0],cam_pos_z[1],cam_pos_z[2],'ro')
        print(cam_pos,cam_pos_x)
        plt.show()
        """
        params = []
        params.append(cam_pos)
        params.append(vec_xx)
        params.append(vec_yy)
        params.append(vec_zz)

        #params = np.asarray(params)
        
        md.write_ply_ascii(files[i],params,4)
        
        

camera_orientation()
