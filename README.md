[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a2j-anchor-to-joint-regression-network-for-3d/pose-estimation-on-hands-2017)](https://paperswithcode.com/sota/pose-estimation-on-hands-2017?p=a2j-anchor-to-joint-regression-network-for-3d) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a2j-anchor-to-joint-regression-network-for-3d/hand-pose-estimation-on-nyu-hands)](https://paperswithcode.com/sota/hand-pose-estimation-on-nyu-hands?p=a2j-anchor-to-joint-regression-network-for-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a2j-anchor-to-joint-regression-network-for-3d/hand-pose-estimation-on-icvl-hands)](https://paperswithcode.com/sota/hand-pose-estimation-on-icvl-hands?p=a2j-anchor-to-joint-regression-network-for-3d) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a2j-anchor-to-joint-regression-network-for-3d/pose-estimation-on-itop-front-view)](https://paperswithcode.com/sota/pose-estimation-on-itop-front-view?p=a2j-anchor-to-joint-regression-network-for-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a2j-anchor-to-joint-regression-network-for-3d/pose-estimation-on-itop-top-view)](https://paperswithcode.com/sota/pose-estimation-on-itop-top-view?p=a2j-anchor-to-joint-regression-network-for-3d)



# A2J: Anchor-to-Joint Regression Network for 3D Articulated Pose Estimation from a Single Depth Image

## Introduction
This is the official implementation for the paper, **"A2J: Anchor-to-Joint Regression Network for 3D Articulated Pose Estimation from a Single Depth Image"**, ICCV 2019. 

In this paper, we propose a simple and effective approach termed A2J, for 3D hand and human pose estimation from a single depth image. Wide-range evaluations on 5 datasets demonstrate A2J's superiority.

Please refer to our paper for more details, https://arxiv.org/abs/1908.09999.

![pipeline](https://github.com/zhangboshen/A2J/blob/master/fig/A2Jpipeline.png)

## Update (2021-9-28)
More details of A2J can be found in our slides (https://github.com/zhangboshen/A2J/blob/master/fig/A2J_Boshen_Zhang_public.pptx).
## Update (2020-6-16)
We upload A2J's prediction results in pixel coordinates (i.e., UVD format) for NYU and ICVL datasets: https://github.com/zhangboshen/A2J/tree/master/result_nyu_icvl,
Evaluation code (https://github.com/xinghaochen/awesome-hand-pose-estimation/tree/master/evaluation) can be applied for performance comparision among SoTA methods.

## Update (2020-3-23)
We released our training code [here](https://github.com/zhangboshen/A2J/tree/master/src_train). 


If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{A2J,
author = {Xiong, Fu and Zhang, Boshen and Xiao, Yang and Cao, Zhiguo and Yu, Taidong and Zhou Tianyi, Joey and Yuan, Junsong},
title = {A2J: Anchor-to-Joint Regression Network for 3D Articulated Pose Estimation from a Single Depth Image},
booktitle = {Proceedings of the IEEE Conference on International Conference on Computer Vision (ICCV)},
year = {2019}
}
```
## Comparison with state-of-the-art methods
![result_hand](https://github.com/zhangboshen/A2J/blob/master/fig/result_hand.png)
![result_body](https://github.com/zhangboshen/A2J/blob/master/fig/result_body.png)

## A2J achieves 2nd place in HANDS2019 3D hand pose estimation Challenge
#### Task 1: Depth-Based 3D Hand Pose Estimation
![T1](https://github.com/zhangboshen/A2J/blob/master/fig/T1.jpg)
#### Task 2: Depth-Based 3D Hand Pose Estimation while Interacting with Objects 
![T2](https://github.com/zhangboshen/A2J/blob/master/fig/T2.jpg)



# About our code 

## Dependencies
Our code is tested under Ubuntu 16.04 environment with NVIDIA 1080Ti GPU, both Pytorch0.4.1 and Pytorch1.2 work (Pytorch1.0/1.1 should also work).

## code
First clone this repository:  
```python
git clone https://github.com/zhangboshen/A2J
```

- `src` folder contains model definition, anchor, and test files for NYU, ICVL, HANDS2017, ITOP, K2HPD datasets.
- `data` folder contains center point, bounding box, mean/std, and GT keypoints files for 5 datasets.

Next you may download our pre-trained model files from:     
- Baidu Yun: https://pan.baidu.com/s/10QBT7mKEyypSkZSaFLo1Vw    
- Google Drive: https://drive.google.com/open?id=1fGe3K1mO934WPZEkHLCX7MNgmmgzRX4z

Directory structure of this code should look like:  
```
A2J
│   README.md
│   LICENSE.md  
│
└───src
│   │   ....py
└───data
│   │   hands2017
│   │   icvl
│   │   itop_side
│   │   itop_top
│   │   k2hpd
│   │   nyu
└───model
│   │   HANDS2017.pth
│   │   ICVL.pth
│   │   ITOP_side.pth
│   │   ITOP_top.pth
│   │   K2HPD.pth
│   │   NYU.pth
```

You may also have to download these datasets manually:  
- NYU Hand Pose Dataset [[link](https://jonathantompson.github.io/)]
- ICVL Hand Pose Dataset [[link](https://labicvl.github.io/hand.html)]
- HANDS2017 Hand Pose Dataset [[link](https://competitions.codalab.org/competitions/17356)]
- ITOP Body Pose Dataset [[link](https://www.alberthaque.com/projects/viewpoint_3d_pose/)]
- K2HPD Body Pose Dataset [[link](http://www.sysu-hcp.net/kinect2-human-pose-dataset-k2hpd/)]

After downloaded these datasets, you can follow the code from `data` folder (data_preprosess.py) to convert ICVL, NYU, ITOP, and K2HPD images to `.mat` files.

Finally, simply run DATASET_NAME.py in the `src` folder to test our model. For example, you can reproduce our HANDS2017 results by running:    
```python
python hands2017.py
```

There are some optional configurations you can adjust in the DATASET_NAME.py files.

Thanks *Gyeongsik et al.* for their nice work to provide precomputed center files (https://github.com/mks0601/V2V-PoseNet_RELEASE) for NYU, ICVL, HANDS2017 and ITOP datasets. This is really helpful to our work!



# Qualitative Results
#### [NYU](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) hand pose dataset:
![NYU_1](https://github.com/zhangboshen/A2J/blob/master/fig/NYU_1.png)
&nbsp;

#### [ITOP](https://www.alberthaque.com/projects/viewpoint_3d_pose/) body pose dataset:
![ITOP_1](https://github.com/zhangboshen/A2J/blob/master/fig/ITOP_1.png)

