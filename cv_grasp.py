#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:06:52 2022

@author: srmist
"""

from read_data import get_depth,segmask,get_grasp,drawGrasp
import argparse
import socket
import time

def send_inference(send_data):
    port = 60064
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


#if __name__=="__main__":
[depth,h,w] = get_depth()
seg = segmask(depth, True, '/home/srmist/Desktop/gqcnn-1.3.0/examples/live_data/','seg')
gp_rep_str,grasp_able = get_grasp(seg,depth,True)
send_inference(gp_rep_str)
