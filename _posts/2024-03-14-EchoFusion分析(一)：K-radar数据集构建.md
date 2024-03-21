---
layout:     post
title:      EchoFusion分析(一)：K-radar数据集构建
subtitle:   Echofusion数据集处理
date:       2024-03-14
author:     LiuLiu
header-img: img/post-bg-debug.png
catalog: true
categories: 代码
tags:
    - EchoFusion
---
- [前言](#前言)
- [K-radar数据集基本结构](#k-radar数据集基本结构)
- [生成Sparse\_Radar\_Point\_Cloud数据](#生成sparse_radar_point_cloud数据)
- [数据集构建](#数据集构建)
  - [生成ImageSets](#生成imagesets)
  - [生成pickle文件](#生成pickle文件)

# 前言
在雷达-可见光融合目标检测中，需要解决的一大问题就是在恶劣天气情况下的目标检测。因此，需要对含有恶劣天气的数据集进行训练，因此需要对`K-radar`数据集进行构建。

# K-radar数据集基本结构
[K-radar数据集](https://github.com/kaist-avelab/K-Radar)主要分为两个部分：
1. 1-20 sequences
2. 21-58 sequences

因为原始数据集太大（12T左右），因此我们只分析`Google Drive`上的数据集。
对于前20个Sequences来说，含有Radar RT信息
```
K-radar
├── 1.zip
    ├──1_cam.zip
        ├── cam-front
        ├── cam-left
        ├── cam-rear
        ├── cam-right
    ├──1_lpc.zip
        ├── os1-128
        ├── os2-64
    ├──1_rt.zip
        ├── radar_zyx_cube
    ├──1_meta.zip
        ├── info_calib
        ├── info_label
        ├── time_info
        ├── description.txt
├── 2.zip
├── ...
```
而后38个Sequences并没有Radar RT信息，而且.zip文件中相对于前20个Sequences还嵌套了一层文件夹。因此在解压缩的过程中需要多一步`mv`的操作
```
K-radar
├── 38.zip
    ├──38_cam.zip
        ├──38_cam
            ├── cam-front
            ├── cam-left
            ├── cam-rear
            ├── cam-right
    ├──38_lpc.zip
        ├──38_lpc
            ├── os1-128
            ├── os2-64
    ├──38_meta.zip
        ├──38_meta
            ├── info_calib
            ├── info_label
            ├── time_info
            ├── description.txt
├── 39.zip
├── ...
```
我们对`K-radar`数据集进行解压，希望得到如下格式
```
K-radar
├──1
    ├── cam-front
    ├── cam-left
    ├── cam-rear
    ├── cam-right
    ├── description.txt
    ├── info_calib
    ├── info_label
    ├── os1-128
    ├── os2-64
    ├── radar_zyx_cube
    ├── time_info
├──2
├──...
```
代码如下
```python
from mmcv import ProgressBar
import pickle as pkl
import pandas as pd
import numpy as np
import time
import os
import argparse

parser = argparse.ArgumentParser(description='K-Radar data unzipper arg parser')
parser.add_argument(
    '--source-path',
    type=str,
    default='/home/home/raid/liuliu/21-37',
    help='specify the zip path of dataset')
parser.add_argument(
    '--target-dir',
    type=str,
    default='/home/home/raid/liuliu/k-radar-21-37',
    help='specify the output path of dataset')
args = parser.parse_args()


if __name__ == '__main__':
    src_dir = args.source_path
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # read all zip files in src_dir and sort them
    zip_files = os.listdir(src_dir)
    zip_files.sort()
    # 5, 7, 16, 17, 20

    # iterate on zip files
    for zip_file in zip_files:
        # extract index name from zip file name
        index = zip_file.split('_')[0]
        # unzip file to indexed file in target_dir
        zip_path = os.path.join(src_dir, zip_file)
        out_path = os.path.join(target_dir, index)
        os.system(f'unzip {zip_path} -d {out_path}')
        ##### 对嵌套文件夹进行处理 #####
        move_file = os.path.join(out_path,zip_file.split('.')[0])
        os.system(f'mv {move_file}/* {out_path}')
        print(f'move {move_file} done.')
        os.system(f'rm -rf {move_file}')
        print(f'rm {move_file} done.')
        ##############################
        # print progress
        print(f'{zip_file} done.')
    
    print('Done.')
```

# 生成Sparse_Radar_Point_Cloud数据
`K-radar`的Google Drive中存储了所有Sequences的Sparse_Radar_Point_Cloud数据,但也可以通过代码进行生成

但我发现，`K-radar`和`Echofusion`的sparse_radar生成config有一些不同，具体不同还需要分析。但这样的话就必须重新生成了，不能用现成的Sparse_Radar_Point_Cloud了。

但由于Sparse_Radar_Point_Cloud数据的生成需要用到Radar_RT数据，因此后28个Sequences不能生成，只能使用Google Drive中存储的Sparse_Radar_Point_Cloud。

代码分析之后解析

Google Drive中spr和代码生成的spr之间的区别：
自己生成的spr的范围更小，google drive中的点云范围更广
因此，如果要用到后38个sequence中的数据，需要对点晕进行filter
```python
>>> a = np.load("/media/liuliu/MyPassport/Datasets/K-Radar/spcube_00033.npy") # 自己生成的spr
>>> a
array([[ 7.00000048, -6.5999999 , -2.20000005,  0.05154945],
       [ 7.4000001 , -6.5999999 , -2.20000005,  0.04836387],
       [ 7.80000019, -6.5999999 , -2.20000005,  0.08280252],
       ...,
       [24.60000038,  5.80000067,  5.4000001 ,  0.44111619],
       [25.        ,  5.80000067,  5.4000001 ,  0.33331506],
       [25.39999962,  5.80000067,  5.4000001 ,  0.05038433]])
>>> b = np.load("/media/liuliu/MyPassport/Datasets/K-Radar/sprdr_00033.npy") # Google Drive中的spr
>>> b
array([[ 8.10000000e+01, -8.02000000e+01, -3.02000000e+01,
         2.93161599e-02],
       [ 8.14000000e+01, -7.98000000e+01, -3.02000000e+01,
         3.55426502e-02],
       [ 8.18000000e+01, -7.94000000e+01, -3.02000000e+01,
         4.22741320e-02],
       ...,
       [ 8.54000000e+01,  7.50000000e+01,  2.94000000e+01,
         4.14133753e-02],
       [ 8.50000000e+01,  7.54000000e+01,  2.94000000e+01,
         3.32191273e-02],
       [ 8.46000000e+01,  7.62000000e+01,  2.94000000e+01,
         3.55036603e-02]])
>>> a.shape
(11520, 4)
>>> b.shape
(150000, 4)
```

# 数据集构建
需要对K-radar数据集构建，生成包含数据集信息的pickle文件，才能对其进行训练
## 生成ImageSets
对于K-radar数据集，首先需要生成用于训练和测试的`ImageSets/*.txt`文件。对于数据集的划分，存储在`K-Radar-main/resources/split/train.txt`和`K-Radar-main/resources/split/test.txt`中。
## 生成pickle文件
