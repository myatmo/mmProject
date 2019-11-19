from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)dp
roperty float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


# load images
path = 'colmap_output/batch2/trial2/images/'
#path = 'img/'
img1 = cv2.imread(path+'left.png')
img2 = cv2.imread(path+'right.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

c1 = np.array([1920, 1080, 3205.38, 960, 540, 0.0999797])
c2 = np.array([1920, 1080, 1948.91, 960, 540])
w = c2[0]
h = c2[1]
f = 2304
cx = c2[3]
cy = c2[4]
K = np.array([[f, 0, cx/2],
              [0, f, cy/2],
              [0, 0, 1]])

# rotation matrix
q1 = np.array([1, 1, 0, 0, 0, -4.54417, 0.249296, 2.07084, 2])
q2 = np.array([2, 0.766459, -0.0366471, -0.641222, -0.00562564, 2.82786, -0.205505, 4.11837, 1])
#R1 = np.array([q1[2], q1[3], q1[4]])
#R2 = np.array([q2[2], q2[3], q2[4]])
R1 = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
R2 = np.array([[0.1776052,  0.0556215, -0.9825287],
               [0.0383742,  0.9972507,  0.0633916],
               [0.9833534, -0.0489624,  0.1749824]])
T1 = np.array([q1[5], q1[6], q1[7]])
T2 = np.array([q2[5], q2[6], q2[7]])
C1 = -R1.T.dot(T1)
C2 = -R2.T.dot(T2)
print(C1)
print(C2)
#ret, mtx, diTst, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


'''
cv2.imshow('camera 1', gray1)
cv2.imshow('camera 2', gray2)
cv2.waitKey(0)
'''

win_size = 3
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp
#Create Block matching object. 
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=10)

'''
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
                               numDisparities = num_disp,
                               blockSize = 5,
                               uniquenessRatio = 5,
                               speckleWindowSize = 5,
                               speckleRange = 5,
                               disp12MaxDiff = 1,
                               P1 = 8*3*win_size**2,#8*3*win_size**2,
                               P2 =32*3*win_size**2) #32*3*win_size**2)
'''
#Compute disparity map
print ("\nComputing the disparity  map...")
disp = stereo.compute(img1, img2)
print(disp)
#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
#plt.imshow(disp,'gray')
#plt.show()

#Generate  point cloud. 
print ("\nGenerating point cloud...")
'''
Q = np.float32([[1,0,0,-w/2.0],
    [0,-1,0,h/2.0],
    [0,0,0,-focal_length],
    [0,0,1,0]])
'''
Q = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,f*0.05,0], 
    [0,0,0,1]])
#print(Q)

#Reproject points into 3D
points = cv2.reprojectImageTo3D(disp, Q)
print(points)
#Get color points
colors = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#Get rid of points with value 0 (i.e no depth)
mask = disp > disp.min()
#Mask colors and points. 
#out_points = points[mask]
out_colors = colors[mask]
print(out_points)
out_fn = 'out.ply'
write_ply(out_fn, out_points, out_colors)
print('%s saved' % out_fn)
'''
cv2.imshow('left', img1)
cv2.imshow('disparity', (disp-min_disp)/num_disp)
cv2.waitKey()
'''
print('Done')




