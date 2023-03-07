#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:24:09 2022

@author: srmist
"""

import read_data
import numpy as np
import time
import socket
# from Grasp_rep_CV import get_depth, segmask, get_grasp, save_norm_filt_npy, to_inference
# from policy_ import  inference
# from policy import read_data

port = 60065

# In[]
# path = '/home/srmist/Desktop/gqcnn-1.3.0/data/examples/single_object/primesense/orientation/'
# name = "mallet_4"
# depth, seg = to_inference(path,name)
# # Pass depth and seg for DL inference
# op = inference(depth, seg, 'primesense.intr')

# In[]
data = "0.08254886,-0.12035486,0.76652517,q1,q2,q3,q4,1,-1,0,4,"
#quaternion = [0.08405,-0.03014,-0.99597,-0.00810]
quaternion = [0.004260,-0.72723,-0.68423,0.03384]
translation = [351,0.07,10.33]
#Camera to robot frame transformation:
#Trc = [[0.0474,-0.98208,0.03619],[-0.9988,-0.04746,-0.00039],[0.00213,-0.03644,-0.9933]]
#trans = np.dot(Trc,translation)
string_to_send = read_data.disp_data_received(quaternion,translation)
print(string_to_send.decode())   

# In[]
count = 1
while True:
    s = socket.socket()
    s.connect(('169.254.98.109',port))
    r_c = s.recv(1024).decode()
    if r_c == 'CAPTURE':
        print("_______Grasp NO. "+str(count)+"_______")
        time.sleep(0.1)
        print("SENDING Grasp pose to SERVER....")
        s_c = s.send(string_to_send)
        time.sleep(0.1)
        print("Sent successfully")
        s.close()
        count+=1
    if count>3:
        break
    if r_c == 'STOP':
        break
       
    
s.close()
print("Socket closed")


