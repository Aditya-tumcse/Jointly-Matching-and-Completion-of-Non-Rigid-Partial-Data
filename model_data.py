import numpy as np
from plyfile import PlyData
from pyntcloud.io import read_off

def read_ply_file(filename):

    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        
       
    return vertices

def write_ply_ascii(file_name,params_to_write,num_points):
    """The function writes the points into .ply file in ascii format.
    :param file_name: Name of the file
    :param params_to_write: The coordinates of the 3D points.
    :num_points: Number of camera positions to be generatedd.
    """
    
    rootdir = "/home/aditya/PycharmProjects/OpenCV-python/Project_2/ascii_ply_files/"
    if file_name.endswith('.off'):
        file_name = file_name[:-4]
   
    fid = open(rootdir + file_name + ".ply","w")
    fid.write("ply\n")
    fid.write("format ascii 1.0\n")
    fid.write("element vertex %d\n"%num_points)
    fid.write("property float x\n")
    fid.write("property float y\n")
    fid.write("property float z\n")
    fid.write("end_header")
    for i in range(num_points):
        fid.write("\n")
        for j in range(3):
            fid.write(str(params_to_write[i][j]))
    
    fid.close()

def centroid_model(file_path):
    """The function generates the centroid of the 3D model.
    :param file_path: Path to the input file(Here path to .off file)
    """
    
    model_points = read_off(file_path)
    model_xyz = model_points["points"][["x","y","z"]].values
    model_x = model_xyz[:,0]
    model_y = model_xyz[:,1]
    model_z = model_xyz[:,2]
    
    model_centroid_x = np.sum(model_x)/np.shape(model_x)[0]
    model_centroid_y = np.sum(model_y)/np.shape(model_y)[0]
    model_centroid_z = np.sum(model_z)/np.shape(model_z)[0]

    model_centroid = [model_centroid_x,model_centroid_y,model_centroid_z]
    return model_centroid