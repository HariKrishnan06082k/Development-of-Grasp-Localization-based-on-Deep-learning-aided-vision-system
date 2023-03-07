# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:25:14 2022

@author: Admin01
"""

import socket            
import time
#from Grasp_rep_CV import get_grasp
#from mathFunc import *
import math
import numpy as np
#from scipy.spatial.transform import Rotation
  
# In[]
# Switching between clients
def connect_Rob(serverSocket,grasp,count):
    
    robIP = "192.168.125.1"
    clientSocket, clientAddr = serverSocket.accept()
        
    #Keep rejecting connection from DL until robot client requests for connection
    cnt=1
    while clientAddr[0] != robIP:
        cnt+=1
        print(cnt)
        clientSocket.shutdown(socket.SHUT_RDWR)
        clientSocket, clientAddr = serverSocket.accept()
        time.sleep(1)
        #clientSocket.close()
        
    #if clientAddr[0] != robIP:
        #clientSocket.shutdown(SHUT_RDWR)
        #clientSocket.close()
    #("Connected to: ",c)
    if clientAddr[0]==robIP:
        
        # do what ever you want here
        print('Connected to Robot',clientAddr)
    
        grasp = grasp.split(',')
    
        print("________________TASK-"+str(count)+"________________")
        trans = str(grasp[0])+","+str(grasp[1])+","+str(grasp[2])+","
        #print("Sending trans to robot...")
        #__________________________________________________S1
        print("Sending pose and robot configuration")
        clientSocket.send(trans.encode())
        # receive data from the server and decoding to get the string.
        #__________________________________________________R1
        x = clientSocket.recv(1024).decode()       #Receive confirmation for trans
        #print (x)
    
        rot =str(grasp[3])+","+str(grasp[4])+","+str(grasp[5])+","+str(grasp[6])+","
        #print("Sending rot to robot...")
        #__________________________________________________S2
        clientSocket.send(rot.encode())
        # receive data from the server and decoding to get the string.
        #__________________________________________________R2
        x = clientSocket.recv(1024).decode()       #Receive confirmation for rot
        #print (x)
        
        #__________________________________________________S3
        config_rot = str(grasp[7])+","+str(grasp[8])+","+str(grasp[9])+","+str(grasp[10])+","+str(grasp[11])+","
        
        clientSocket.send(config_rot.encode())
        # receive data from the server and decoding to get the string.
        #__________________________________________________R3
        x = clientSocket.recv(1024).decode()       #Receive confirmation for config and ext
        print ("Robot received pose")
        time.sleep(0.2)
        
        #__________________________________________________R3
        x = clientSocket.recv(1024).decode()       #Receive confirmation for reachability
        print (x)
        time.sleep(0.2)
        
        
        task = clientSocket.recv(1024).decode()       #Task done
        print(task)
    else:
        print("Unable to connect to Robot due to simultaneous client connection requests")
    
    
    return task,clientSocket

# DL and Cam client
def connect_DL(serverSocket):
    clientSocket, clientAddr = serverSocket.accept()
    print('Connected to DL and Cam',clientAddr)
    clientSocket.send('CAPTURE'.encode())
    time.sleep(0.1)
    #Keep checking for 'b'string!=0
    grasp = clientSocket.recv(1024).decode()
    print("Grasp received: ", grasp)
    time.sleep(0.1)
    clientSocket.close()
    return grasp

# CUSTOM MATH FUNCTIONS
def rtx2qt(rot):
    #print(rot)
    qw = math.sqrt(1+rot[0][0]+rot[1][1]+rot[2][2])/2
    qx = (rot[2][1]-rot[1][2])/(4*qw)
    qy = (rot[0][2]-rot[2][0])/(4*qw)
    qz = (rot[1][0]-rot[0][1])/(4*qw)
    quat = [qw,qx,qy,qz]
    #print(quat)
    return quat
def tfm2qt(rot):
    #print(rot)
    qw = math.sqrt(abs(1+rot[0][0]+rot[1][1]+rot[2][2]))/2
    qx = (rot[2][1]-rot[1][2])/(4*qw)
    qy = (rot[0][2]-rot[2][0])/(4*qw)
    qz = (rot[1][0]-rot[0][1])/(4*qw)
    den = abs(qw**2 + qx**2 + qy**2 + qz**2)
    mag = math.sqrt(den)
    quat = [qx/mag,qy/mag,qz/mag,qw/mag]
    #print(quat)
    return quat

def qt2rtx(qt):
    q0 = qt[3] # Real part
    q1 = qt[0] #x
    q2 = qt[1] #y
    q3 = qt[2] #z
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1) 
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1 
    # 3x3 rotation matrix
    rot = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    
    return rot

def toHomTrans(rot,trans):
    Tco = np.asarray([[rot[0][0], rot[0][1],rot[0][2],trans[0][0]],[rot[1][0], rot[1][1],rot[1][2],trans[1][0]],[rot[2][0], rot[2][1],rot[2][2],trans[2][0]],[0,0,0,1]])
    return Tco

def splitGraspString(grasp):
    grasp = grasp.split(",")
    trans = [[float(grasp[0])],[float(grasp[1])],[float(grasp[2])]]
    quat = [float(grasp[3]),float(grasp[4]),float(grasp[5]),float(grasp[6])]
    # FOR IN-PLANE ROTATION SCAM
    #axis[0] = 0
    #axis[1] = 0
    #axis[2] = -axis[2]
    #rz = np.asarray([[math.cos(axis[2]),-math.sin(axis[2]),0],[math.sin(axis[2]),math.cos(axis[2]),0],[0,0,1]])
    #ry = np.asarray([[math.cos(axis[1]),0,math.sin(axis[1])],[0,1,0],[-math.sin(axis[1]),0,math.cos(axis[1])]])
    #rx = np.asarray([[1,0,0],[0,math.cos(axis[0]),-math.sin(axis[0])],[0,math.sin(axis[0]),math.cos(axis[0])]])
    
    #R = rz
    R = qt2rtx(quat)
    
    return trans,R

def flexPoseToHomTrans(world_pose_wrt_robot):
    trans = [[-world_pose_wrt_robot[0]],[world_pose_wrt_robot[1]],[-world_pose_wrt_robot[2]]]
    qt = [world_pose_wrt_robot[3],world_pose_wrt_robot[4],world_pose_wrt_robot[5],world_pose_wrt_robot[6]]
    Trw = toHomTrans(qt2rtx(qt),trans)
    Trw[0][0] = 1
    Trw[0][1] = 0
    Trw[0][2] = 0
    Trw[1][0] = 0
    Trw[1][1] = 1
    Trw[1][2] = 0
    Trw[2][0] = 0
    Trw[2][1] = 0
    Trw[2][2] = 1
    return Trw
def qt2eul(quat):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    
    t0 = 2.0*(w*x+y*z)
    t1 = 1.- - 2.0*(x*x + y*y)
    roll_x = math.atan2(t0,t1)
    t2 = 2.0*(w*y-z*x)
    t2 = 1.0 if t2>1.0 else t2
    t2 = -1.0 if t2<-1.0 else t2
    pitch_y = math.asin(t2)
    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    yaw_z = math.atan2(t3,t4)
    return roll_x*180/math.pi,pitch_y*180/math.pi,yaw_z*180/math.pi

def inf2rob(grasp):
    grasp_rob_str=""
    trans,R = splitGraspString(grasp)
    #rot = qt2rtx(qt)
    #print(R)
    Tco = toHomTrans(R,trans)
    #print("C wr.t O",Tco)
    #print("Grasp wr.t Cam",Tco)
    #print("Grasp w.r.t Camera",Tco)
    #T_grasp_rot = np.asarray([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    #Tco = np.dot(Tco,T_grasp_rot) # Final grasp frame after re-orientation
    #______________________________________
    #Tcw = np.load('Tcw_closer.npy')
    #print("W wr.t c: \n",Tcw)
    #Twc = np.linalg.inv(Tcw)
    #print("C wr.t W: \n",Twc)
    #Trw = np.load('Trw_closer.npy')
    #print("W wr.t R: \n",Trw)
    #Two = np.dot(Twc,Tco)
    #print("O wr.t W: \n",Two)
    #Tro = np.dot(Trw,Two)
    #print("O wr.t W: \n",Tro)
    Trc = np.load('Trc.npy')
    Tro = np.dot(Trc,Tco)
    
    #Tro[0][0] = 0
    #Tro[0][1] = -1
    #Tro[0][2] = 0
    #Tro[1][0] = -1
    #Tro[1][1] = 0
    #Tro[1][2] = 0
    #Tro[2][0] = 0
    #Tro[2][1] = 0
    #Tro[2][2] = -1
    
    quat_rob= tfm2qt(Tro)
    
    #Rro_grasp = np.dot(R , qt2rtx(quat_rob))
    #quat_rob = rtx2qt(Rro_grasp)
    #print("Grasp orientation w.r.t Robot",quat_rob)
    conf=[-1,1,-1,4]
    #print(grasp_rob_str)
    grasp_rob_str=str(round(Tro[0][3],2))+","+str(round(Tro[1][3],2))+","+str(round(Tro[2][3],2))+","+str(round(quat_rob[3],6))+","+str(round(quat_rob[0],6))+","+str(round(quat_rob[1],6))+","+str(round(quat_rob[2],6))+","+str(conf[0])+","+str(conf[1])+","+str(conf[2])+","+str(conf[3])+"," 
    return grasp_rob_str
# In[]
def main():
    # Reserving port >1024
    port = 60064               
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind to the port
    serverSocket.bind(('', port))        
    print ("Socket binded to Port: %s" %(port))  
    # Server listen to requestscoming from other computers on the network
    serverSocket.listen(10)    
    print ("Server is listening") 
 
    connect = True
    count=0
    while connect:
        #print("Enter preference for Grasp inference: \n")
        #print("0 : CV\n")
#        print("1 : GQCNN\n")
#        inf = input()
#        if inf=="1":
#            pref = "GQCNN"
#        elif inf=="0":
#            pref = "CV"
        count+=1
        grasp = connect_DL(serverSocket)
        #grasp = float(grasp.split(","))
        #grasp = "411.06,0,40,0.01972,-0.85417,-0.51782,-0.04308,-1,-1,0,0,-1.74"
        print("Waiting for Robot")
        grasp4robot = inf2rob(grasp)
        print("GRASP REP TO ROBOT: ",grasp4robot)
        task,rob_clientSocket = connect_Rob(serverSocket,grasp4robot,count)
        if task=="TASK DONE":
            if count<2:
                rob_clientSocket.send(b'CONTINUE')
                rob_clientSocket.close()
            else:
                rob_clientSocket.send(b'STOP')
                rob_clientSocket.close()
                time.sleep(1)
                print("____________ ALL TASKS COMPLETED ___________")
                connect=False
               
        elif task=="CURRENT TASK TERMINATED":
            rob_clientSocket.send(b'STOP')
            rob_clientSocket.close()
            connect=False
            
    print("ALL TASKS COMPLETED")
    serverSocket.close()
    print("SERVER SOCKET CLOSED")

if __name__ == "__main__":
    main()

# In[]


