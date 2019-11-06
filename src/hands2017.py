import cv2
import torch
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import scipy.io as scio
import os
from PIL import Image
import model as model
import anchor as anchor
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

fx = 475.065948
fy = 475.065857
u0 = 315.944855
v0 = 245.287079

TestImgFrames = 295510
keypointsNumber = 21
cropWidth = 176
cropHeight = 176
batch_size = 12
xy_thres = 100
depth_thres = 150

save_dir = './result/HANDS2017'

try:
    os.makedirs(save_dir)
except OSError:
    pass

testingImageDir = '/data/zhangboshen/CODE/Anchor_Pose_fpn/data/Hands2017/frame/images/'
center_file = '../data/hands2017/hands2017_center_test.mat'
MEAN = np.load('../data/hands2017/hands2017_mean.npy')
STD = np.load('../data/hands2017/hands2017_std.npy')
model_dir = '../model/HANDS2017.pth'
keypointsUVD_test = np.zeros((TestImgFrames,keypointsNumber,3),dtype=np.float32)
result_file = 'result_HANDS2017.txt'


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x

center_test = scio.loadmat(center_file)['centre_pixel']
center_test = center_test.astype(np.float32)

centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-xy_thres
centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+xy_thres
centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, xy_thres=100, depth_thres=150):
    
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
 
    new_Xmin = max(lefttop_pixel[index,0,0], 0)
    new_Ymin = max(rightbottom_pixel[index,0,1], 0)  
    new_Xmax = min(rightbottom_pixel[index,0,0], img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index,0,1], img.shape[0] - 1)

    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32') 

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2] 
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2] 
    imgResize = (imgResize - center[index][0][2])

    imgResize = (imgResize - mean) / std
    
    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    
    label_xy[:,0] = (keypointsUVD[index,:,0].copy() - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) 
    label_xy[:,1] = (keypointsUVD[index,:,1].copy() - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) 
    
    imageOutputs[:,:,0] = imgResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1] 
    labelOutputs[:,2] = (keypointsUVD[index,:,2] - center[index][0][2])   # Z  
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):
    def __init__(self, ImgDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD):

        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

    def __getitem__(self, index):
        depth = Image.open(self.ImgDir + 'image_D%.8d'%(index+1) + '.png') 
        depth = np.array(depth)
        
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, self.xy_thres, self.depth_thres)
            
        return data, label
    
    def __len__(self):
        return len(self.center)

      
test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel, test_rightbottom_pixel, keypointsUVD_test)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)
     
def main():   
    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load(model_dir)) 
    net = net.cuda()
    net.eval()
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    
    output = torch.FloatTensor()
    
    torch.cuda.synchronize() 
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):    
        with torch.no_grad():
            img, label = img.cuda(), label.cuda()             
            heads = net(img)  
            pred_keypoints = post_precess(heads,voting=False)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
        
    torch.cuda.synchronize()       
    result = output.cpu().data.numpy()
    writeTxt(result, center_test)
   

def writeTxt(result, center):

    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:,:,1]
    resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD = resultUVD_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)


    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(rightbottom_pixel[i,0,1], 0)  
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(lefttop_pixel[i,0,1], 240*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        resultUVD[i,:,2] = result[i,:,2] + center[i][0][2]

    resultXYD = pixel2world(resultUVD.copy(), fx, fy, u0, v0)

    resultReshape = resultXYD.reshape(len(resultXYD), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:     
        for i in tqdm(range(len(resultReshape))):
            f.write('frame/images/' + 'image_D%.8d'%(i+1) + '.png' + '\t')
            for j in range(keypointsNumber*3):
                f.write(str(resultReshape[i, j])+'\t')
            f.write('\n') 

    f.close()


if __name__ == '__main__':
    main()
 
