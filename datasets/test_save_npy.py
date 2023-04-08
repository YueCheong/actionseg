import numpy as np
import h5py
import os
import datetime
import matplotlib.pyplot as plt

from AS_base import SegDataset
from torch.utils.data import Dataset, DataLoader


dir_camera_param = "/temp/liuf/hoi4d/camera_params/"
camera_param1  = np.load(dir_camera_param+"ZY20210800001/intrin.npy")
print(camera_param1.shape)
print(camera_param1)
camera_param2  = np.load(dir_camera_param+"ZY20210800002/intrin.npy")
print(camera_param2.shape)
print(camera_param2)
camera_param3  = np.load(dir_camera_param+"ZY20210800003/intrin.npy")
print(camera_param3.shape)
print(camera_param3)
camera_param4  = np.load(dir_camera_param+"ZY20210800004/intrin.npy")
print(camera_param4.shape)
print(camera_param4)

CAM_WID, CAM_HGT = 1024, 768 #重新投影到深度图尺寸宽*高
CAM_FX1, CAM_FY1 = camera_param1[0][0], camera_param1[1][1]
CAM_CX1, CAM_CY1 = camera_param1[0][2], camera_param1[1][2]

CAM_FX2, CAM_FY2 = camera_param2[0][0], camera_param2[1][1]
CAM_CX2, CAM_CY2 = camera_param2[0][2], camera_param2[1][2]

CAM_FX3, CAM_FY3 = camera_param3[0][0], camera_param3[1][1]
CAM_CX3, CAM_CY3 = camera_param3[0][2], camera_param3[1][2]

CAM_FX4, CAM_FY4 = camera_param4[0][0], camera_param4[1][1]
CAM_CX4, CAM_CY4 = camera_param4[0][2], camera_param4[1][2]
# print("CAM_FX, CAM_FY", CAM_FX, CAM_FY)
# print("CAM_CX, CAM_CY", CAM_CX, CAM_CY)
EPS = 1.0e-16

dir_train_data = "/temp/liuf/hoi4d/action_seg/"

train_data_1 = h5py.File(dir_train_data+"train1.h5","r")
train_data_2 = h5py.File(dir_train_data+"train2.h5","r")
train_data_3 = h5py.File(dir_train_data+"train3.h5","r")
train_data_4 = h5py.File(dir_train_data+"train4.h5","r")

pc1 = train_data_1['pcd'][:]
pc2 = train_data_2['pcd'][:]
pc3 = train_data_3['pcd'][:]
pc4 = train_data_4['pcd'][:]
print("pc1.shape",pc1.shape)
print("pc2.shape",pc2.shape)
print("pc3.shape",pc3.shape)
print("pc4.shape",pc4.shape)

pc = np.concatenate((pc1,pc2,pc3,pc4), axis=0)
print("pc.shape",pc.shape)
#print(pc)   #(b, l, n, c) = (2971, 150, 2048, 3) 


label1 = train_data_1['label'][:]
label2 = train_data_2['label'][:]
label3 = train_data_3['label'][:]
label4 = train_data_4['label'][:]
print("label1.shape",label1.shape)
print("label2.shape",label2.shape)
print("label3.shape",label3.shape)
print("label4.shape",label4.shape)

label = np.concatenate((label1,label2,label3,label4), axis=0)
print("label.shape",label.shape)
print(label)   #(b, l) = (2971, 150) 


dir_save = "/temp/liuf/hoi4d/action_depth_npy/"

for clips in range(len(pc)): # 2971
    IMG=[]
    tmp=[]
    for frames in range(len(pc[0])): #150
        frame = pc[clips, frames, :, :]
        print(frame.shape)
        # #检查并清除镜头后的点
        valid = frame[:,2] >EPS
        z = frame[valid,2]
        print(z.shape)
         #点云反向映射到像素坐标位置
        if clips>=0 and clips<724: #第[0,724)个视频用的相机参数是ZY20210800004
            u = np.round(frame[valid,0] * CAM_CX4 / z + CAM_CX4).astype(int)
            v = np.round(frame[valid,1] * CAM_CY4 / z + CAM_CY4).astype(int)
        elif clips>=724 and clips<1255: #第[724,1255)个视频用的相机参数是ZY20210800003
            u = np.round(frame[valid,0] * CAM_CX3 / z + CAM_CX3).astype(int)
            v = np.round(frame[valid,1] * CAM_CY3 / z + CAM_CY3).astype(int)            
        elif clips>=1255 and clips<1950: #第[1255,1950)个视频用的相机参数是ZY20210800002
            u = np.round(frame[valid,0] * CAM_CX2 / z + CAM_CX2).astype(int)
            v = np.round(frame[valid,1] * CAM_CY2 / z + CAM_CY2).astype(int)              
        elif clips>=1950 and clips<2971: #第[1950,2971)个视频用的相机参数是ZY20210800001
            u = np.round(frame[valid,0] * CAM_CX1 / z + CAM_CX1).astype(int)
            v = np.round(frame[valid,1] * CAM_CY1 / z + CAM_CY1).astype(int)     
        else:
            print("out of the clips range!!!!")          

        
        #滤除超出图像尺寸的无效像素
        valid = np.bitwise_and(np.bitwise_and((u>=0), (u<CAM_WID)), np.bitwise_and((v>=0), (v<CAM_HGT)))
        u, v, z = u[valid], v[valid], z[valid]

        #按距离填充生成深度图，近距离覆盖远距离
        img_z = np.full((CAM_HGT, CAM_WID), np.inf) #初始化为inf
        for ui, vi, zi in zip(u,v,z):
            img_z[vi,ui] = min(img_z[vi,ui], zi) #近距离像素屏蔽远距离像素

        #小洞和"透射消除 axis=0，垂直滚动，1水平滚动，1个单位
        img_z_shift = np.array([img_z, 
                                np.roll(img_z, 1, axis=0),    
                                np.roll(img_z, -1, axis=0), 
                                np.roll(img_z, 1, axis=1), 
                                np.roll(img_z, -1, axis=1)])
        img_z = np.min(img_z_shift, axis=0) #保留滑动窗口内灰度最低的值
        # print('img_z.shape', img_z.shape)
        # print(img_z)
        tmp.append(img_z[np.newaxis,:]) 

    clip_imgs = np.concatenate(tmp,axis = 0)
    print("clip_imgs", clip_imgs.shape)  #(l, CAM_HGT, CAM_WID) = (150, 1080, 1920)
    
    np.save(dir_save+'depth_clip{}'.format(clips)+'.npy', clip_imgs)


        
        #保存重新投影生成的深度图dep_rot
        #np.savetxt('/home/dh/alisa/temp/actiondepth/test_depth_csv/depth_clip{}'.format(clips)+'_frame{}'.format(frames)+'.csv', img_z, fmt='%.12f', delimiter=',', newline='\n')

#         # #加载刚保存的深度图depth并显示
#         # img = np.genfromtxt('/home/dh/alisa/temp/actiondepth/test_depth_csv/depth_clip{}'.format(clips)+'_frame{}'.format(frames)+'.csv', delimiter=',').astype(np.float32)
#         # plt.imshow(img, cmap='jet')
#         # plt.show()
        




    













