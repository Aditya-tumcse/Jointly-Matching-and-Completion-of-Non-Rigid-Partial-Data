import os
import numpy as np
import pandas as pd
from pyntcloud.io import read_off,write_ply
from plyfile import PlyData
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math
import pyrender
import model_data as md
import camera as cam
import trimesh
import cv2
import fusion


def triangle_area(v1,v2,v3):
    """Computes the area of the triangle
    :param v1:Contains coordinates of point v1
    :param v2:Contains coordinates of point v2
    :param v3:Contains coordinates of point v3
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1,v3 - v1),axis=1)

def off_to_ply_conversion():
    """Function that converts a .off file into .ply file"""

    ballet_vicon = read_off("/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/1746411.off")
    n = 7000
        
    ballet_vicon_points_xyz = ballet_vicon["points"][["x", "y", "z"]].values #Gives the x,y,z coordinates of a point
    ballet_vicon_points_rgb = ballet_vicon["points"][["red", "green", "blue"]].values

    v1_xyz = ballet_vicon_points_xyz[ballet_vicon["mesh"]["v1"]] #Coordinates of Point 1 of the triangular face
    v2_xyz = ballet_vicon_points_xyz[ballet_vicon["mesh"]["v2"]] #Coordinates of Point 2 of the triangular face
    v3_xyz = ballet_vicon_points_xyz[ballet_vicon["mesh"]["v3"]] #Coordinates of Point 3 of the triangular face
        
    v1_rgb = ballet_vicon_points_rgb[ballet_vicon["mesh"]["v1"]]
    v2_rgb = ballet_vicon_points_rgb[ballet_vicon["mesh"]["v2"]]
    v3_rgb = ballet_vicon_points_rgb[ballet_vicon["mesh"]["v3"]]

    areas = triangle_area(v1_xyz, v2_xyz, v3_xyz)
    probabilities = areas / areas.sum()
    weighted_random_choices = np.random.choice(range(len(areas)), size=n, p=probabilities) #Gives the vertex number
    #weighted_random_choices are the face numbers.
        
        
    v1_xyz = v1_xyz[weighted_random_choices] # Gives the coordinates of the vertex number weighted_random_choices 
    v2_xyz = v2_xyz[weighted_random_choices]
    v3_xyz = v3_xyz[weighted_random_choices]
        
    v1_rgb = v1_rgb[weighted_random_choices]
    v2_rgb = v2_rgb[weighted_random_choices]
    v3_rgb = v3_rgb[weighted_random_choices]
   
    kp = Barycentric_coordinates(v1_xyz,v2_xyz,v3_xyz,n)  
    write_ply("/home/aditya/PycharmProjects/OpenCV-python/Project_2/ply_files_keypoints/1746411.ply",points=kp)
    return weighted_random_choices

def matching_keypoints():
    """Returns the matching keypoints of other frames"""

    face_num = off_to_ply_conversion() #Gives the face numbers of keypoints associated with first frame
    rootdir = "/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/"
    path, dirs, files = next(os.walk(rootdir))
    files = sorted(files)
    
    for i in range(1,len(files)):
        ba_vicon = read_off(rootdir + files[i])
    
        ba_vicon_points_xyz = ba_vicon["points"][["x","y","z"]].values
    
        vertex1_xyz = np.zeros([np.shape(face_num)[0],3])
        vertex2_xyz = np.zeros([np.shape(face_num)[0],3])
        vertex3_xyz = np.zeros([np.shape(face_num)[0],3])
        for j in range(np.shape(face_num)[0]):
            vertex1_xyz[j] = ba_vicon_points_xyz[ba_vicon["mesh"]["v1"][face_num[j]]]
            vertex2_xyz[j] = ba_vicon_points_xyz[ba_vicon["mesh"]["v2"][face_num[j]]]
            vertex3_xyz[j] = ba_vicon_points_xyz[ba_vicon["mesh"]["v3"][face_num[j]]]

        bary_coords = Barycentric_coordinates(vertex1_xyz,vertex2_xyz,vertex3_xyz,np.shape(face_num)[0])
        if files[i].endswith('.off'):
            files[i] = files[i][:-4]
        write_ply("/home/aditya/PycharmProjects/OpenCV-python/Project_2/ply_files_keypoints/" + files[i] + ".ply",points=bary_coords) 
        

def Barycentric_coordinates(v1,v2,v3,N):
    """Computes the barycentric coordinates
    :param v1: Coordinates of the vertex v1
    :param v2: Coordinates of the vertex v2
    :param v3: Coordinates of the vertex v3
    :param N: Number of keypoints givven randomly
    """

    u = np.random.rand(N, 1)
    v = np.random.rand(N, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]

    w = 1 - (u + v)

    result = pd.DataFrame()
    result_xyz = (v1 * u) + (v2 * v) + (v3 * w)  # Coordinates of the barycentric points
    result_xyz = result_xyz.astype(np.float32)
      

    result["x"] = result_xyz[:, 0] #Barycentric x-coordinates
    result["y"] = result_xyz[:, 1] #Barycentric y-coordinates
    result["z"] = result_xyz[:, 2] #Barycentric z-coordinates

    return result       


def depth_map(fx,fy,cx,cy):
    """The function generates depth maps for each model. Each model will have number_positions depth maps.
    :param fx: Focal length in X-Axis
    :param fy: Focal length in Y-Axis
    :param cx: Centre of the image in x-coordinates
    :param cy: Centre of the image in y-coordinates
    """

    rootdir = "/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/"
    path, dirs, files = next(os.walk(rootdir))
    files = sorted(files)

    #for i in range(len(files)):
    for i in range(1):    
        if files[i].endswith('.off'):
            files[i] = files[i][:-4]
            

        #parent_dir_depth_map = "/home/aditya/PycharmProjects/OpenCV-python/Project_2/Depth_maps"
        parent_dir_depth_map = "/home/aditya/PycharmProjects/OpenCV-python/Project_2"
        directory = files[i]
        path = os.path.join(parent_dir_depth_map,directory)
        os.mkdir(path) #Creates directories in the parent directory Depth_maps
        print("\nCreating directory ",files[i])

        ballet_vicon_trimesh = trimesh.load(rootdir + files[i] + ".off")
        mesh = pyrender.Mesh.from_trimesh(ballet_vicon_trimesh)
    
        extrinsic_matrix = cam.camera_extrinsics(rootdir + files[i] + ".off")
        camera = pyrender.IntrinsicsCamera(fx,fy,cx,cy)
        
        for j in range(np.shape(extrinsic_matrix)[0]):
            scene = pyrender.Scene()
            scene.add(mesh)
            scene.add(camera,pose=extrinsic_matrix[j])
        
            r = pyrender.OffscreenRenderer(1080,1080)
            color,depth = r.render(scene)
            print(depth)
            fig = plt.figure()
            #plt.plot()
            plt.axis('off')
            plt.imshow(depth,cmap=plt.cm.gray_r)
            plt.imshow(color)
            fig.savefig(parent_dir_depth_map + "/" + files[i] + "/figure_" + str(j))
            fig.savefig(parent_dir_depth_map + "/" + files[i] + "/cfig_" + str(j))
            plt.close()

            np.savez_compressed("/home/aditya/PycharmProjects/OpenCV-python/Project_2/TDCV-Project-2/trial1.npz",depth=depth)
       
def tsdf_voxel_volume(fx,fy,cx,cy):
    print("Estimating voxel volume bounds....")
    cam_intrinsics =  cam.camera_intrinsics(fx,fy,cx,cy)
    vol_bnds = np.zeros((3,2))
    depth_image = np.load("/home/aditya/PycharmProjects/OpenCV-python/Project_2/TDCV-Project-2/trial1.npz")['depth']
    camera_pose = np.identity(4)

    view_frust_pts = fusion.get_view_frustum(depth_image,cam_intrinsics,camera_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0],np.amin(view_frust_pts,axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1],np.amax(view_frust_pts,axis = 1))
    

    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)
    
    
    tsdf_vol.integrate(depth_image,depth_image, cam_intrinsics, camera_pose, obs_weight=1.0)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)
    """
    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)
    """

#face_numbers_of_keypoints("/home/aditya/PycharmProjects/OpenCV-python/Project_2/1746411_keypoints.ply","/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/1746411.off")
#matching_keypoints()
#centroid_model("/home/aditya/Documents/Sem_3/TDCV/project_2/tracking/ballet_vicon/mesh/1746411.off")
#random_camera_position(50,4)
#depth_map(3000,3000,540.0,540.0)
tsdf_voxel_volume(3000,3000,540.0,540.0)
