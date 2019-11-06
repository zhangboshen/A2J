import cv2
import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import torch.utils.data
import sys
import model as model
import anchor as anchor
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# DataHyperParms 
keypointsNumber = 15
cropWidth = 288
cropHeight = 288
batch_size = 12
depthFactor = 50

save_dir = './result'

try:
    os.makedirs(save_dir)
except OSError:
    pass

testingImageDir = '/data/zhangboshen/CODE/Anchor_Pose_fpn/data/top_test/depthImages/'
keypointsfileTest = '../data/itop_top/itop_top_keypoints3D_test.mat'
bndbox_test = scio.loadmat('../data/itop_top/itop_top_bndbox_test.mat' )['FRbndbox_test']   
center_test = scio.loadmat('../data/itop_top/itop_top_center_test.mat')['centre_pixel']
Img_mean = np.load('../data/itop_top/itop_top_mean.npy')[3]
Img_std = np.load('../data/itop_top/itop_top_std.npy')[3]
model_dir = '../model/ITOP_top.pth'

def pixel2world(x,y,z):
    worldX = (x - 160.0)*z*0.0035
    worldY = (120.0 - y)*z*0.0035
    return worldX,worldY
    
def world2pixel(x,y,z):
    pixelX = 160.0 + x / (0.0035 * z)
    pixelY = 120.0 - y / (0.0035 * z)
    return pixelX,pixelY
    

keypointsWorldtest = scio.loadmat(keypointsfileTest)['keypoints3D']
keypointsPixeltest = np.ones((len(keypointsWorldtest),15,2),dtype='float32')
keypointsPixeltest_tuple = world2pixel(keypointsWorldtest[:,:,0],keypointsWorldtest[:,:,1],keypointsWorldtest[:,:,2])
keypointsPixeltest[:,:,0] = keypointsPixeltest_tuple[0]
keypointsPixeltest[:,:,1] = keypointsPixeltest_tuple[1]

joint_id_to_name = {
  0: 'Head',
  1: 'Neck',
  2: 'RShoulder',
  3: 'LShoulder',
  4: 'RElbow',
  5: 'LElbow',
  6: 'RHand',
  7: 'LHand',
  8: 'Torso',
  9: 'RHip',
  10: 'LHip',
  11: 'RKnee',
  12: 'LKnee',
  13: 'RFoot',
  14: 'LFoot',
}


def dataPreprocess(index, img, keypointsPixel, keypointsWorld, bndbox, center):
    
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 

    new_Xmin = max(bndbox[index][0], 0)
    new_Ymin = max(bndbox[index][1], 0)
    new_Xmax = min(bndbox[index][2], img.shape[1] - 1)
    new_Ymax = min(bndbox[index][3], img.shape[0] - 1)

    
    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C

    imgResize = (imgResize - Img_mean) / Img_std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    label_xy[:,0] = (keypointsPixel[index,:,0] - new_Xmin)*cropWidth/(new_Xmax - new_Xmin)
    label_xy[:,1] = (keypointsPixel[index,:,1] - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y
    
    imageOutputs[:,:,0] = imgResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1] 
    labelOutputs[:,2] = (keypointsWorld.copy()[index,:,2])*depthFactor  
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, trainingImageDir, bndbox, keypointsPixel, keypointsWorld, center):

        self.trainingImageDir = trainingImageDir
        self.mean = Img_mean
        self.std = Img_std
        self.bndbox = bndbox
        self.keypointsPixel = keypointsPixel
        self.keypointsWorld = keypointsWorld
        self.center = center
        
    def __getitem__(self, index):

        data4DTemp = scio.loadmat(self.trainingImageDir + str(index+1) + '.mat')['DepthNormal']       
        depthTemp = data4DTemp[:,:,3]
        data, label = dataPreprocess(index, depthTemp, self.keypointsPixel, self.keypointsWorld, self.bndbox, self.center)

        return data, label
    
    def __len__(self):
        return len(self.bndbox)

test_image_datasets = my_dataloader(testingImageDir, bndbox_test, keypointsPixeltest, keypointsWorldtest, center_test)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)
      
def main():   
    
    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load(model_dir)) 
    net = net.cuda()
    net.eval()
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)

    output = torch.FloatTensor()
        
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):     
        with torch.no_grad():

            img, label = img.cuda(), label.cuda()
            heads = net(img)  
            pred_keypoints = post_precess(heads,voting=False)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
    
    result = output.cpu().data.numpy()
    print('Accuracy:', evaluation10CMRule(result,keypointsWorldtest,bndbox_test, center_test))
    evaluation10CMRule_perJoint(result,keypointsWorldtest,bndbox_test, center_test)


def evaluation10CMRule(source, target, Bndbox, center):
    assert np.shape(source)==np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]
    
    for i in range(len(Test1_)):
             
        Test1[i,:,0] = Test1_[i,:,0]*(Bndbox[i,2]-Bndbox[i,0])/cropWidth + Bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(Bndbox[i,3]-Bndbox[i,1])/cropHeight + Bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor #+ center[i][0][2]
    TestWorld = np.ones((len(Test1),keypointsNumber,3))    
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])
    
    TestWorld[:,:,0] = TestWorld_tuple[0]
    TestWorld[:,:,1] = TestWorld_tuple[1]
    TestWorld[:,:,2] = Test1[:,:,2]

    count = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(TestWorld[i,j,0] - target[i,j,0]) + np.square(TestWorld[i,j,1] - target[i,j,1]) + np.square(TestWorld[i,j,2] - target[i,j,2])<np.square(0.1): #10cm   
                count = count + 1         
    accuracy = count/(len(source)*keypointsNumber)
    return accuracy
   

def evaluation10CMRule_perJoint(source, target, Bndbox, center):
    assert np.shape(source)==np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]
    
    for i in range(len(Test1_)):
             
        Test1[i,:,0] = Test1_[i,:,0]*(Bndbox[i,2]-Bndbox[i,0])/cropWidth + Bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(Bndbox[i,3]-Bndbox[i,1])/cropHeight + Bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor #+ center[i][0][2]
    TestWorld = np.ones((len(Test1),keypointsNumber,3))    
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])
    
    TestWorld[:,:,0] = TestWorld_tuple[0]
    TestWorld[:,:,1] = TestWorld_tuple[1]
    TestWorld[:,:,2] = Test1[:,:,2]

    count = 0
    accuracy = 0
    for j in range(keypointsNumber):
        for i in range(len(source)):      
            if np.square(TestWorld[i,j,0] - target[i,j,0]) + np.square(TestWorld[i,j,1] - target[i,j,1]) + np.square(TestWorld[i,j,2] - target[i,j,2])<np.square(0.1): #10cm   
                count = count + 1     

        accuracy = count/(len(source))
        print('joint_', j,joint_id_to_name[j], ', accuracy: ', accuracy)
        accuracy = 0
        count = 0

if __name__ == '__main__':
    main()
    
      
