import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_anchors(P_h=None, P_w=None):
    if P_h is None:
        P_h = np.array([2,6,10,14])

    if P_w is None:
        P_w = np.array([2,6,10,14])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1  
    return anchors          

def shift(shape, stride, anchors):
    shift_h = np.arange(0, shape[0]) * stride
    shift_w = np.arange(0, shape[1]) * stride

    shift_h, shift_w = np.meshgrid(shift_h, shift_w)
    shifts = np.vstack((shift_h.ravel(), shift_w.ravel())).transpose()

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))

    return all_anchors

class post_process(nn.Module):
    def __init__(self, P_h=[2,6], P_w=[2,6], shape=[48,26], stride=8,thres = 8,is_3D=True):
        super(post_process, self).__init__()
        anchors = generate_anchors(P_h=P_h,P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape,stride,anchors)).cuda().float()
        self.thres = torch.from_numpy(np.array(thres)).cuda().float()
        self.is_3D = is_3D
    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0],b.shape[0]).cuda()
        for i in range(a.shape[1]):
            dis += torch.pow(torch.unsqueeze(a[:, i], dim=1) - b[:,i],0.5)
        return dis

    def forward(self,heads,voting=False):
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
        batch_size = classifications.shape[0]
        anchor = self.all_anchors #*(w*h*A)*2
        P_keys = []
        for j in range(batch_size):

            classification = classifications[j, :, :] #N*(w*h*A)*P
            regression = regressions[j, :, :, :] #N*(w*h*A)*P*2
            if self.is_3D:
                depthregression = depthregressions[j, :, :]#N*(w*h*A)*P
            reg = torch.unsqueeze(anchor,1) + regression
            reg_weight = F.softmax(classifications[j, :, :],dim=0) #(w*h*A)*P
            reg_weight_xy = torch.unsqueeze(reg_weight,2).expand(reg_weight.shape[0],reg_weight.shape[1],2)#(w*h*A)*P*2
            P_xy = (reg_weight_xy*reg).sum(0)
            if self.is_3D:
                P_depth = (reg_weight*depthregression).sum(0)
                P_depth = torch.unsqueeze(P_depth,1)
                P_key = torch.cat((P_xy,P_depth),1)            
                P_keys.append(P_key)
            else:
                P_keys.append(P_xy)
        return torch.stack(P_keys)

class A2J_loss(nn.Module):
    def __init__(self,P_h=[2,6], P_w=[2,6], shape=[8,4], stride=8,thres = [10.0,20.0],spatialFactor=0.1,img_shape=[0,0],is_3D=True):
        super(A2J_loss, self).__init__()
        anchors = generate_anchors(P_h=P_h, P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape,stride,anchors)).cuda().float()
        self.thres = torch.from_numpy(np.array(thres)).cuda().float()
        self.spatialFactor = spatialFactor
        self.img_shape = img_shape
        self.is_3D = is_3D
    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0],b.shape[0]).cuda()
        for i in range(a.shape[1]):
            dis += torch.abs(torch.unsqueeze(a[:, i], dim=1) - b[:,i])  
        return dis

    def forward(self, heads, annotations):
        alpha = 0.25
        gamma = 2.0
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
        #classifications,scalar,mu = classifications_tuple
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = self.all_anchors # num_anchors(w*h*A) x 2
        anchor_regression_loss_tuple = []

        for j in range(batch_size):

            classification = classifications[j, :, :] #N*(w*h*A)*P
            regression = regressions[j, :, :, :] #N*(w*h*A)*P*2
            if self.is_3D:
                depthregression = depthregressions[j, :, :]#N*(w*h*A)*P
            bbox_annotation = annotations[j, :, :]#N*P*3=>P*3
            reg_weight = F.softmax(classification,dim=0) #(w*h*A)*P
            reg_weight_xy = torch.unsqueeze(reg_weight,2).expand(reg_weight.shape[0],reg_weight.shape[1],2)#(w*h*A)*P*2         
            gt_xy = bbox_annotation[:,:2]#P*2 

            anchor_diff = torch.abs(gt_xy-(reg_weight_xy*torch.unsqueeze(anchor,1)).sum(0)) #P*2
            anchor_loss = torch.where(
                torch.le(anchor_diff, 1),
                0.5 * 1 * torch.pow(anchor_diff, 2),
                anchor_diff - 0.5 / 1
            )
            anchor_regression_loss = anchor_loss.mean()
            anchor_regression_loss_tuple.append(anchor_regression_loss)
#######################regression 4 spatial###################
            reg = torch.unsqueeze(anchor,1) + regression #(w*h*A)*P*2
            regression_diff = torch.abs(gt_xy-(reg_weight_xy*reg).sum(0)) #P*2
            regression_loss = torch.where(
                torch.le(regression_diff, 1),
                0.5 * 1 * torch.pow(regression_diff, 2),
                regression_diff - 0.5 / 1
                )
            regression_loss = regression_loss.mean()*self.spatialFactor
########################regression 4 depth###################
            if self.is_3D:
                gt_depth = bbox_annotation[:,2] #P
                regression_diff_depth = torch.abs(gt_depth - (reg_weight*depthregression).sum(0))#(w*h*A)*P       
                regression_loss_depth = torch.where(
                    torch.le(regression_diff_depth, 3),
                    0.5 * (1/3) * torch.pow(regression_diff_depth, 2),
                    regression_diff_depth - 0.5 / (1/3)
                    )
                regression_loss += regression_diff_depth.mean()           
############################################################
            regression_losses.append(regression_loss)
        return torch.stack(anchor_regression_loss_tuple).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)   
