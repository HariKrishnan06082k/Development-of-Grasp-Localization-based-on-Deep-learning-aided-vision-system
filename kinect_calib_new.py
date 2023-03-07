# CODED BY: P JEMUEL STANLEY

# In[ ]:

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import math


# In[3]:


# Defining the dimensions of checkerboard
CHECKERBOARD = (7,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

c_s = 19.78
# Defining the world coordinates for 3D points
objp = np.zeros((1,CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp*c_s
prev_img_shape = None
corners_ = []

# In[4]:


# Extracting path of individual image stored in a given directory
images = glob.glob("frames\*.png")
c = 1
for fname in images:
    img = cv2.imread(fname)
    img=np.fliplr(img) # mirror image
    plt.imshow(img,'gray')
    plt.title("Image file No. "+str(c))
    plt.show()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    corners_.append(corners)
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = img.astype(np.uint8)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        plt.title("Image file No. "+str(c))
        plt.imshow(img)
        plt.show()
    c+=1
    
    #cv2.imshow('img',img)
    #cv2.waitKey(0)

#cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# https://jasonchu1313.github.io/2017/10/01/kinect-calibration/
#mtx = [[458.4554786,0,343.645038],[0,458.1992727,229.805975],[0,0,1]]
#Xing Zhou - A Study of Microsoft Kinect Calibration
mtx = [[580.606,0,314.758],[0,580.885,252.187],[0,0,1]]
#Calib for 1 frame
#mtx = [[623.38,0,279.476],[0,622.517,239.4477],[0,0,1]]

mtx=np.asarray(mtx)
ret_1, rvecs, tvecs = cv2.solvePnP(objpoints[0],corners_[0],mtx,None)
#grasp_end_point,_ = cv2.projectPoints(np.array([(0.0,0.0,100.0)]),rvec_1,tvec_1,mtx,dist)

#From rodrigues formula of conversion from axis-angle to Rotation matrix
# Note: norm(rvecs[i]) = theta  (in radians)
theta = np.linalg.norm(rvecs)*180/math.pi
#theta = theta_d*math.pi/180
#I = [[1,0,0],[0,1,0],[0,0,1]]
#r_skew = [[0,-rvecs[0][2][0],rvecs[0][1][0]],[rvecs[0][2][0],0,-rvecs[0][0][0]],[-rvecs[0][1][0],rvecs[0][0][0],0]]
#R = np.dot(math.cos(theta),I) + np.dot((1-math.cos(theta)),rvecs[0]*np.transpose(rvecs[0])) + np.dot(math.sin(theta),r_skew)
#R = I + np.dot((1-math.cos(theta)),rvecs[0]*np.transpose(rvecs[0])) + np.dot(math.sin(theta),r_skew)

print("Camera matrix : \n")
print(mtx)
print("Rotation vector: \n")
print(rvecs)
print("Translation vector: \n")
print(tvecs)
R_matrix,_ = cv2.Rodrigues(rvecs)

Tcw = np.append(np.append(R_matrix,tvecs,axis=1),[[0,0,0,1]],axis=0)
print("Transform cam w.r.t world: \n",Tcw)
# In[ ]:
# Saving the transformation metrices
txw = 450
tyw = 0
tzw = 0
 
Twb = [[0,-1,0,txw],[-1,0,0,tyw],[0,0,-1,tzw],[0,0,0,1]]

Tbc = np.dot(Tcw,Twb)
print("Transform cam w.r.t robot base: \n",Tbc)
# In[ ]:
np.save("Cam_ext_par_IR",Tcw)
np.save("Cam_int_par_IR",mtx)ss
asd = np.load('IR_intrinsic.npy')




