import cv2
import torch
import torch.utils.data
from torch.autograd import Variable
import scipy.io as scio
import numpy as np
import os
from PIL import Image
import model as model
import anchor as anchor
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# DataHyperParms 
TrainImgNumber = 55943
TestImgNumber = 39732
keypointsNumber = 19
cropWidth = 288
cropHeight = 288
batch_size = 12

save_dir = './result/K2HPD'

try:
    os.makedirs(save_dir)
except OSError:
    pass


testingImageDir = '/data/zhangboshen/CODE/Anchor_Pose_fpn/data/K2HPD/normal_Test/'
keypointsTest = scio.loadmat('../data/k2hpd/k2hpd_keypoints_test.mat')['keypointsTest'].astype(np.float32)
bndboxTest = scio.loadmat('../data/k2hpd/k2hpd_bndbox_test.mat')['FRbndbox_test']
Img_mean = np.load('../data/k2hpd/k2hpd_mean.npy')
Img_std = np.load('../data/k2hpd/k2hpd_std.npy')
model_dir = '../model/K2HPD.pth'

def dataPreprocess(index, img, keypointsPixel, bndbox):
    
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 2), dtype = 'float32') 

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
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, trainingImageDir, bndbox, keypointsPixel):

        self.trainingImageDir = trainingImageDir
        self.mean = Img_mean
        self.std = Img_std
        self.bndbox = bndbox
        self.keypointsPixel = keypointsPixel

    def __getitem__(self, index):

        data4DTemp = scio.loadmat(self.trainingImageDir + str(index+1) + '.mat')['DepthNormal']       
        depthTemp = data4DTemp[:,:,3]
        
        data, label = dataPreprocess(index, depthTemp, self.keypointsPixel, self.bndbox)

        return data, label
    
    def __len__(self):
        return len(self.bndbox)


test_image_datasets = my_dataloader(testingImageDir, bndboxTest, keypointsTest)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)


def main():   
    
    net = model.A2J_model(num_classes = keypointsNumber, is_3D=False)
    net.load_state_dict(torch.load(model_dir)) 
    net = net.cuda()
    net.eval()
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None,is_3D=False)
    output = torch.FloatTensor()
        
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):
       
        with torch.no_grad():

            img, label = img.cuda(), label.cuda() 
            heads = net(img) 
            pred_keypoints = post_precess(heads,voting=False)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
            

    result = output.cpu().data.numpy()
    print('Accuracy 0.05:', evaluationPDJ(result.copy(), keypointsTest, bndboxTest, 0.05))
    print('Accuracy 0.10:', evaluationPDJ(result.copy(), keypointsTest, bndboxTest, 0.1))
    print('Accuracy 0.15:', evaluationPDJ(result.copy(), keypointsTest, bndboxTest, 0.15))
    print('Accuracy 0.20:', evaluationPDJ(result.copy(), keypointsTest, bndboxTest, 0.2))

def evaluationPDJ(source, target, Bndbox, PDJRatio):
    
    assert np.shape(source)==np.shape(target), "source has different shape with target"
    predUV = np.zeros((np.shape(source)))
    predUV[:,:,0] = source[:,:,1]
    predUV[:,:,1] = source[:,:,0]
    
    count = 0
    for i in range(len(source)):
        Xmin = Bndbox[i,0] 
        Ymin = Bndbox[i,1] 
        Xmax = Bndbox[i,2] 
        Ymax = Bndbox[i,3] 
        
        predUV[i,:,0] = predUV[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        predUV[i,:,1] = predUV[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
    
        torsoDiameter = np.sqrt(np.square(target[i,1,0] - (target[i,9,0]/2 + target[i,12,0]/2))+ np.square(target[i,1,1] - (target[i,9,1]/2 + target[i,12,1]/2)))
        for j in range(keypointsNumber):
            if np.sqrt(np.square(predUV[i,j,0] - target[i,j,0]) + np.square(predUV[i,j,1] - target[i,j,1])) <=  PDJRatio*torsoDiameter: #10cm   
                count = count + 1         
    accuracy = count/(len(source)*keypointsNumber)
    return accuracy


if __name__ == '__main__':
    main()
    
      
