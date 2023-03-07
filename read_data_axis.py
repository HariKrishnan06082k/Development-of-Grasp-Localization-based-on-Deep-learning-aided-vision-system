#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:24:28 2022

@author: srmist
"""

import numpy as np
import matplotlib.pyplot as plt
import socket
import freenect
import cv2
import time
import math

def get_depth():
    img = freenect.sync_get_depth()[0]
    return [img,img.shape[0],img.shape[1]]

def segmask(depth, vis, path,name):
    #ROI
    x = depth.copy()
    m,n = x.shape
    for i in range(0,m):
        for j in range(0,n):
            if (j<70 or i>320) or (j>500 or i<140):
                x[i][j]=0


# Thresholding bit values based on bit values:
    for i in range(0,480):
        for j in range(0,640):
            if x[i][j]<=682 and x[i][j]!=0:
                x[i][j]=200
    plt.imshow(x,'gray')
    plt.title('Filled depth')
    plt.show()

#Creates Seg Mask
    seg_mask=np.zeros([m,n])
    for i in range(0,m):
        for j in range(0,n):
            if x[i][j]==200:
                seg_mask[i][j]=2047
            else:
                seg_mask[i][j]=0
    if vis==True:
        plt.imshow(depth,'gray')
        plt.title('Original depth image: ')
        plt.show()
        plt.imshow(x,'gray')
        plt.title('Thresholded depth image: ')
        plt.show()
        plt.imshow(seg_mask,'gray')
        plt.title('Segmentation mask : ')
        plt.show()
    plt.imshow(seg_mask,'gray')
    plt.title('Segmentation mask : ')
    plt.show()
    cv2.imwrite(path+"seg_"+name+".png",seg_mask)
    return seg_mask
    
#Gives grasp representation in camera coordinates
def get_grasp(seg,depth,vis):
    #Contour detection
    seg = seg.astype(np.uint8)
    edges = cv2.Canny(seg, 3, 3)
    contours,_= cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
    #print("Size of contours: ",np.size(contours))
    cnt = contours[np.size(contours)-1]
    M = cv2.moments(cnt)
    #print(M)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #print(cx,cy)
    coeff = [-4.92547027*10**1, 1.295892*10**-1,-1.49627597*10**-4,6.57479679*10**-8,7422.015152146081]
    n = int(depth[cx][cy])
    dist = coeff[0]*n**1 + coeff[1]*n**2 + coeff[2]*n**3 + coeff[3]*n**4 + coeff[4]
    #print("Gray values: ",n)
    #print("Distance: ",dist)
    rect = cv2.minAreaRect(cnt)
    #print("Rectangle with least area: ",rect)
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    #print(box)
    box = np.int0(box)
    cv2.drawContours(depth,[box],0,(0,0,255),2)
    
    cam_intr = ([580.606, 0, 314.758],[0 ,580.885, 252.187],[0, 0, 1])
    #{"_cy": 252.187, "_cx": 314.758, "_fy": 580.885, "_height": 480, "_fx": 580.606, "_width": 640, "_skew": 0.0, "_K": 0, "_frame": "primesense_overhead"}
    centroid = np.dot(dist,([cx],[cy],[1]))
    pos = np.dot(np.linalg.inv(cam_intr),centroid)
    
    rot = [[math.cos(rect[2]*math.pi/180), -math.sin(rect[2]*math.pi/180), 0], [math.sin(rect[2]*math.pi/180), math.cos(rect[2]*math.pi/180), 0],[0,0,1]]
    #print(rot)
    qw = math.sqrt(1+rot[0][0]+rot[1][1]+rot[2][2])/2
    qx = (rot[2][1]-rot[1][2])/(4*qw)
    qy = (rot[0][2]-rot[2][0])/(4*qw)
    qz = (rot[1][0]-rot[0][1])/(4*qw)
    quat = [qx,qy,qz,qw]
    
    #Check whether graspable or not
    w = min(rect[1][0],rect[1][1])
    #print([rect[1][0],rect[1][1]])
    
    #convert to camera coordinate
    w_l = [[w*dist*math.cos((90-rect[2])*math.pi/180)],[w*dist*math.sin((90-rect[2])*math.pi/180)],[1]]
    #print(w_l)
    w_cam = np.linalg.norm(np.dot(np.linalg.inv(cam_intr),w_l))
    print("Calculated gripper width in camera coordinates:",w_cam)
    if w_cam<=50:
        g_able = True
        grasp_rep = [pos,quat,rect[2]]
        print("Object graspable")
    else:
        g_able = False
        grasp_rep = [[0,0,0],[0,0,0,0],0]
        print("Object not graspable")
    #print(quat)
    
    if vis==True:
        plt.imshow(depth,'gray')
        plt.title('Grasp representation : ')
        plt.show()
    return grasp_rep, g_able

# ### Grasp representation
def save_norm_filt_npy(depth,path,name):
    coeff = [-4.92547027*10**1, 1.295892*10**-1,-1.49627597*10**-4,6.57479679*10**-8,7422.015152146081]
   # Thresholding bit values based on bit values:
    for i in range(0,480):
        for j in range(0,640):
            if depth[i][j]>=750 or depth[i][j]==0:
                depth[i][j]=750
        
    plt.imshow(depth,'gray')
    plt.show()

    x = depth.astype(np.float32)

    for i in range(0,480):
        for j in range(0,640):
            n = x[i][j]
            dist = coeff[0]*n**1 + coeff[1]*n**2 + coeff[2]*n**3 + coeff[3]*n**4 + coeff[4]
            x[i][j] = dist/1000
    x = x.reshape((480,640,1))
    np.save(path+str(name),x)
    return x

def drawGrasp(img,x,y,b,ang,save,name,p):
    x_img = img.copy()
    ang = ang*math.pi/180
    w1 = 12.5
    #ang = math.pi * (1/4)
    x1,y1 = (x+(b/2)*math.cos(ang),y+(b/2)*math.sin(ang))
    x2,y2 = (x+(b/2)*math.cos(math.pi + ang),y+(b/2)*math.sin(math.pi + ang))
    cv2.circle(x_img, (x,y), 4, (255,0,0), -1)
    #cv2.line(x_img, (int(x2),int(y2)),(int(x1),int(y1)),(255,0,0),3)
    plt.plot([x2,x1],[y2,y1],'--',color='r', linewidth=2)
    x_21,y_21 = (x2 + w1*math.cos(-((math.pi/2)-ang)), y2 + w1*math.sin(-((math.pi/2)-ang)))
    x_22,y_22 = (x2 + w1*math.cos(math.pi-((math.pi/2)-ang)), y2 + w1*math.sin(math.pi-((math.pi/2)-ang)))
    
    x_11,y_11 = (x1 + w1*math.cos(-((math.pi/2)-ang)), y1 + w1*math.sin(-((math.pi/2)-ang)))
    x_12,y_12 = (x1 + w1*math.cos(math.pi-((math.pi/2)-ang)), y1 + w1*math.sin(math.pi-((math.pi/2)-ang)))
    
    cv2.line(x_img, (int(x2),int(y2)),(int(x_21),int(y_21)),(255,0,0),4)
    cv2.line(x_img, (int(x2),int(y2)),(int(x_22),int(y_22)),(255,0,0),4)
    
    cv2.line(x_img, (int(x1),int(y1)),(int(x_11),int(y_11)),(255,0,0),4)
    cv2.line(x_img, (int(x1),int(y1)),(int(x_12),int(y_12)),(255,0,0),4)
    plt.imshow(x_img, 'gray')
    if save==True:
        path = p+"INF_"+name+".png"
        cv2.imwrite(path,cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB))
    plt.show()
    

#x = np.load('D:/Tasks/Shape base/28-Feb/depthframebottle2.npy')
def to_inference(path,name):
    x = get_depth()[0]
    #path = '/home/srmist/Desktop/gqcnn-1.3.0/data/examples/single_object/primesense/orientation/'
    #name = "mallet_3"
    
    seg_img=segmask(x,False,path,name)
    #Save filtered depth for GQCNN 
    
    x = save_norm_filt_npy(x,path,name+'.npy')
    cv2.imwrite(path+"Real_Depth"+name+".png",x)
    return x,seg_img
    #grasp,able = get_grasp(seg_img,x,True)
    #print(grasp)
  
def disp_data_received(axis,angle,translation):
    port = 60064
    axis = [round(x,3) for x in axis]
    axis = [str(x) for x in axis]
    angle = [round(x,3) for x in angle]
    angle = [str(x) for x in angle]
    translation = [round(x*1000,3) for x in translation]
    translation = [str(x) for x in translation]
    string = list()
    config = ['-1','-1','-1','0']
    for item in translation:
        string.append(item)
    for item in axis:
        string.append(item)
    for item in angle:
        string.append(item)
    for item in config:
        string.append(item)
    send_data = ''
    for item in string:
        send_data = send_data + item + ','
    print("POSE_: ",send_data)   
    #return send_data.encode()
    while True:
        s = socket.socket()
        s.connect(('10.1.12.112',port))
        r_c = s.recv(1024).decode()
        if r_c == 'CAPTURE':
            print("_______Grasp-Execution_______")
            time.sleep(0.1)
            print("SENDING Grasp pose to SERVER....")
            s_c = s.send(send_data.encode())
            time.sleep(0.1)
            print("Sent successfully")
            s.close()
            break
        if r_c == 'STOP':
            break
    # print("Socket closed")
# if __name__=="__main__":
#   string = disp_data_received()
#   print("string pose",string)      
