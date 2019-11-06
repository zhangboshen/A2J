
import numpy as np
import matplotlib.image as mpimg
import h5py
import os
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
import math
import time
import cv2

depth_maps = h5py.File('/dataspace/zhangboshen/ITOP_LSTM/ITOP_NewBaseline/data/top_train/ITOP_top_train_depth_map.h5', 'r')
labels = h5py.File('/dataspace/zhangboshen/ITOP_LSTM/ITOP_NewBaseline/data/top_train/ITOP_top_train_labels.h5', 'r')
   
saveDir = '/dataspace/zhangboshen/ITOP_LSTM/ITOP_NewBaseline/data/top_train/depthImages/'
def GetDepthNormal(depth_maps,labels):
    DepthNormal = np.zeros((240,320,4),dtype='float32')
    count = 0
    for i in range(depth_maps['data'].shape[0]):
        if labels['is_valid'][i]:
            if count%1000 == 0:
                print(count)
            depth_map = depth_maps['data'][i].astype(np.float32)
            joints = labels['image_coordinates'][i]
            height, width = np.shape(depth_map)
            for x in range(1,height-1):
                for y in range(1,width-1):
                    dzdx = (depth_map[x+1,y] - depth_map[x-1,y]) / 2.0
                    dzdy = (depth_map[x,y+1] - depth_map[x,y-1]) / 2.0
        
                    DepthNormal[x,y,0] = -dzdx
                    DepthNormal[x,y,1] = -dzdy
                    DepthNormal[x,y,2] = 1
            DepthNormal[:,:,3] = depth_map[:,:]
            
            count = count+1
        
            scio.savemat(saveDir + str(count) + '.mat', {'DepthNormal':DepthNormal})
            
    return 0
 


if __name__ == '__main__':
    GetDepthNormal(depth_maps,labels)
    

