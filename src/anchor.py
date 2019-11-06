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
