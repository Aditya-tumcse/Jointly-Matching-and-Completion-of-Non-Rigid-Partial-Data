import os
import numpy as np
import model_data as md
import random
import math
import view


def random_camera_position(filename):
    """The function generates random camera positions around the object.
    :param filename: Complete path to the .off file
    """
    number_points = 50
    radius = 4
    
    radius = radius**(1.0/3.0)
    x_data = np.zeros([number_points,1])
    y_data = np.zeros([number_points,1])
    z_data = np.zeros([number_points,1])
    camera_positions = np.zeros([number_points,3])
    for i in range(number_points):
        theta = random.uniform(0,math.pi)
        phi = random.uniform(-math.pi,math.pi)

        x_data[i] = radius * np.sin(theta) * np.cos(phi)
        y_data[i] = radius * np.sin(theta) * np.sin(phi)
        z_data[i] = radius * np.cos(theta)
        camera_positions[i] = [x_data[i],y_data[i],z_data[i]]
    #md.write_ply_ascii(filename,camera_positions,number_points)

    return camera_positions  

def camera_extrinsics(filename):
    """The function gives the camera extrinsics. This is an implementation of opengl lookat function
    :param filename:Complete path to the .off file
    """
    up_vec = np.array([0,1,0])
    cam_position = random_camera_position(filename)
    
    centroid = md.centroid_model(filename)
    camera_extrinsics_matrix = np.zeros([50,4,4])

    for i in range(np.shape(cam_position)[0]):
        camera_extrinsics_matrix[i]= view.m3dLookAt(cam_position[i],centroid,up_vec)
    
    return camera_extrinsics_matrix
    
    
 
        


camera_extrinsics("/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/1746412.off")
#random_camera_position("/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/1746412.off")
