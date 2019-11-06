import numpy as np
import pandas as pd
import h5py
import cv2
import scipy.io as scio
from PIL import Image
import matplotlib.pyplot as plt

#train = pd.read_csv('./train_annos.txt')

trainList = []
with open('./test_annos.txt') as f:
    #line = f.readline()
    #while line:
    for line in f.readlines():
        trainList.append(line)

trainImNames = []
for i in range(len(trainList)):
    trainImNames.append(trainList[i][0:12])

imgDir = '/data05/zhangboshen/K2HPD/depth_data/depth_images/';
saveDir = '/data05/zhangboshen/K2HPD/depth_data/normal_Train/'
def GetDepthNormal(ImageNames):
    DepthNormal = np.zeros((212,256,4),dtype='float32')
    count = 0
    for i in range(len(ImageNames)):
        if i%1000 == 0:
            print i
        imName = ImageNames[i] 
        im = Image.open(imgDir + imName)
        im = np.asarray(im,dtype = 'float32')  # H*W*C
        im = im[:,:,0]
        depthTemp = np.ascontiguousarray(im, dtype=np.float32)
        imgTemp = cv2.normalize(depthTemp, depthTemp, 0, 1, cv2.NORM_MINMAX)
        imgTemp = np.array(imgTemp * 255, dtype = np.uint8)   # 240*320
        height, width = np.shape(im)
        for x in range(1,height-1):
            for y in range(1,width-1):
                dzdx = (im[x+1,y] - im[x-1,y]) / 2.0
                dzdy = (im[x,y+1] - im[x,y-1]) / 2.0
        
                DepthNormal[x,y,0] = -dzdx
                DepthNormal[x,y,1] = -dzdy
                DepthNormal[x,y,2] = 1
        normalTemp = np.ascontiguousarray(DepthNormal[:,:,0:3], dtype=np.float32)
        normal = cv2.normalize(normalTemp, normalTemp, 0, 1, cv2.NORM_MINMAX)
        normal = np.array(normal * 255, dtype = np.uint8)   # 240*320
        DepthNormal[:,:,0:3] = normal 
        DepthNormal[:,:,3] = imgTemp[:,:]
        
        count = count+1
        
        scio.savemat(saveDir + str(count) + '.mat', {'DepthNormal':DepthNormal})
            
    return 0

    
if __name__ == '__main__':
    GetDepthNormal(trainImNames)