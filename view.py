import math
import numpy as np

def normalize(v):
    l = np.linalg.norm(v)
    return [v[0]/l, v[1]/l, v[2]/l]

def m3dLookAt(eye, target, up):
    mz = normalize(eye - target)
    mx = normalize(np.cross(up,mz))
    my = normalize(np.cross(mz,mx))
    tx =  eye[0]
    ty =  eye[1]
    tz = eye[2]   
    
    return np.array([[mx[0], my[0], mz[0], tx],[mx[1], my[1], mz[1], ty],[mx[2], my[2], mz[2], tz],[0, 0, 0, 1]])