---
layout:     post
title:      EchoFusion分析(一)：RADIal数据集构建 (KITTI format)
subtitle:   Echofusion数据集处理
date:       2024-03-13
author:     LiuLiu
header-img: img/post-bg-debug.png
catalog: true
categories: 代码
tags:
    - EchoFusion
---
- [前言](#前言)
- [RADIal数据集简介](#radial数据集简介)
- [Covert Dataset](#covert-dataset)
- [数据集构建](#数据集构建)

# 前言
在目标检测算法中，很多算法对于数据集格式的输入有要求，通常采用`Kitti format`或者是`Nuscence format`形式。因此，其他数据集在训练之前，会有一个数据集构建的过程。这次记录一下`RADIal`数据集的构建过程。

# RADIal数据集简介
RADIal数据集的官方网址：https://github.com/valeoai/RADIal

首先了解一下`RADIal`数据集的基本结构
```
RADIal
├── camera: Front Camera images 
├── laser_PCL: Laser point cloud data
├── radar_FFT: Radar Range-Doppler data
├── radar_Freespace: Freespace in radar view
├── radar_PCL: Radar Sparse point cloud
├── labels.csv: Label of frames in csv (Center Points)
```
需要将RADIal数据集的结构转化为以下结构
```
RAIal
├── CalibrationTable.npy
├── calibs
├── images： Camera images
├── ImageSets: 
├── labels: Label of frames in txt (same as label.csv)
├── lidars: Laser point cloud data
├── radars_adc: Radar ADC Data
├── radars_pcd: Radar Point Cloud Data
├── radars_ra: Radar Range-Azimuth Data
├── radars_rd: Radar Range-Doppler Data
├── radars_rt: Radar Range-Time Data
```
其中，`radars_adc`、`radars_ra`、`radars_rd`和`radars_rt`都是由`RADIal`中的信号处理模块`DBReader`完成的

# Covert Dataset
首先是要将`RADIal`转化为`kitti`格式，主要有两个步骤
1. 提取相应frame数据
2. 信号处理模块


**1. 提取`label`数据**
```python
labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
    unique_ids = np.unique(labels[:,0]).tolist()
    label_dict = {}
    for i,ids in enumerate(unique_ids):
        sample_ids = np.where(labels[:,0]==ids)[0]
        label_dict[ids]=sample_ids
    sample_keys = list(label_dict.keys())
    pkbar = ProgressBar(len(unique_ids))
```

**2. 将`image`、`lidar`、`Imageset`、`label`数据进行提取**
```python
def map_func(frame_id):
        # From the sample id, retrieve all the labels ids
        entries_indexes = label_dict[frame_id]
        # Get the objects labels
        box_infos = labels[entries_indexes]

        record_name = box_infos[:, -3][0]
        data_root = os.path.join(root_dir, record_name)
        db = SyncReader(data_root, tolerance=20000, silent=True);

        idx = box_infos[:, -2][0]
        sample = db.GetSensorData(idx)

        # save ImageSets
        if record_name in Sequences['Validation']:
            set_name = 'val'
        elif record_name in Sequences['Test']:
            set_name = 'test'
        else:
            set_name = 'train'
        with open(os.path.join(out_dir, 'ImageSets', set_name + '.txt'), 'a') as f:
            f.write('%06d' % frame_id + '\n')

        # save image
        image = sample['camera']['data']  # load camera data, [1080, 1920, 3]
        image_name = os.path.join(out_dir, 'images', '%06d.png' % frame_id)
        cv2.imwrite(image_name, image)

        # save lidar as binary
        pts_lidar = sample['scala']['data'].astype(dtype=np.float32) # load lidar data, [15608, 11], no compensation
        lidar_name = os.path.join(out_dir, 'lidars', '%06d.bin' % frame_id)
        pts_lidar.tofile(lidar_name)

        # save labels as txt
        label_name = os.path.join(out_dir, 'labels', '%06d.txt' % frame_id)
        np.savetxt(label_name, box_infos, fmt='%d %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %s %d %d')
```

**3. 信号处理模块**
```python
        RSP = RadarSignalProcessing(clibration_path, method='ADC',device='cuda',silent=True)
        adc=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar data after range fft

        RSP = RadarSignalProcessing(clibration_path, method='RT',device='cuda',silent=True)
        rt=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar data after range fft
        
        RSP = RadarSignalProcessing(clibration_path, method='PC',silent=True)
        pcd=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']).astype(dtype=np.float32) # load radar pcd

        RSP = RadarSignalProcessing(clibration_path, method='RA',silent=True)
        ra=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar ra map
        
        RSP = RadarSignalProcessing(clibration_path, method='RD',silent=True)
        rd=RSP.run(sample['radar_ch0']['data'],sample['radar_ch1']['data'],
                sample['radar_ch2']['data'],sample['radar_ch3']['data']) # load radar ra map

        radar_name = os.path.join(out_dir, 'radars_adc', '%06d.bin' % frame_id)
        adc.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_rt', '%06d.bin' % frame_id)
        rt.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_pcd', '%06d.bin' % frame_id)
        pcd.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_ra', '%06d.bin' % frame_id)
        ra.tofile(radar_name)

        radar_name = os.path.join(out_dir, 'radars_rd', '%06d.bin' % frame_id)
        rd.tofile(radar_name)
```
# 数据集构建
数据集的构建主要是生成
1. 包含了数据集基本信息的`'.pkl'`文件
2. `2D annotation`，信息存储在pickle文件中

**1. 提取pkl文件基本信息**
```python
    def map_func(idx):
        info = {}
        pc_info = {'num_features': 11}
        radar_info = {'radar_shape': [512, 256, 16]}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if lidars:
            pc_info['lidars_path'] = get_lidars_info_path(
                idx, path, training, relative_path)
        if radars:
            radar_info['radars_rt_path'] = get_radars_info_path(
                idx, path, 'radars_rt', training, relative_path)
            radar_info['radars_pcd_path'] = get_radars_info_path(
                idx, path, 'radars_pcd', training, relative_path)
            radar_info['radars_ra_path'] = get_radars_info_path(
                idx, path, 'radars_ra', training, relative_path)
            
        image_info['image_path'] = get_image_path(idx, path, training,
                                                  relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info # image info
        info['radar'] = radar_info # radar info
        info['point_cloud'] = pc_info # lidar info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            calibration = np.load(calib_path,allow_pickle=True).item()

            calib_info['rvec'] = calibration['extrinsic']['rotation_vector']
            calib_info['tvec'] = calibration['extrinsic']['translation_vector']
            calib_info['cam_mat'] = calibration['intrinsic']['camera_matrix']
            calib_info['dist_coeffs'] = calibration['intrinsic']['distortion_coefficients']

            info['calib'] = calib_info # calib info

        if annotations is not None:
            info['annos'] = annotations # annotation info

        return info
```
其中，pkl中的信息包含5项
1. `image info`
2. `radar info`
3. `lidar info`
4. `calib info`
5. `annotation info`

其中，前4项直接调用文件信息即可
而`annotation info`需要将`RADIal`数据集中的`label`信息转化为`kitti`格式

**2. 生成`annotation`信息**
```python
def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'doppler': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()

    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[1] != '-1'])

    if num_objects == 0: # if there is no object
        content = []
    annotations['name'] = np.array(['Car' for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[1:5]]
                                    for x in content]).reshape(-1, 4)
    
    # fake dimensions in radar coordinates format(wlh).
    dim = [4, 1.8, 1.5]
    annotations['dimensions'] = np.array([dim for x in content]).reshape(-1, 3)

    # combine lidar and radar anno to get position annotation
    # 7:10 corresponds to lidar z and radar x, y
    annotations['location'] = np.array([[float(info) for info in x[7:10]]
                                    for x in content]).reshape(-1, 3)[:, [2, 1, 0]]
    # transform z to bottom center under radar coordinates, 
    # 0.42m and 0.8m are the height of lidar and radar
    annotations['location'][:, 2] = annotations['location'][:, 2] - 0.8 - dim[2] / 2
    annotations['location'][:, 1] = -annotations['location'][:, 1]
    annotations['location'][:, 0] = annotations['location'][:, 0] + dim[0] / 2

    # add fake yaw.
    annotations['rotation_y'] = np.array([0.0 for x in content]).reshape(-1)

    # record doppler, reflected power and difficulty
    annotations['doppler'] = np.array([float(x[12]) for x in content])
    annotations['difficulty'] = np.array([int(x[-1]) for x in content])

    # deal with truncated, occluded and alpha
    locations = annotations['location']
    annotations['alpha'] = np.array([-10 for loc in locations])
    annotations['truncated'] = np.array([0.0 for x in content])
    annotations['occluded'] = np.array([0 for x in content])

    index = list(range(num_objects))
    annotations['index'] = np.array(index, dtype=np.int32)
    
    return annotations
```
kitti数据集的[annotation格式](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt)为

```
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```

由于该数据集的原始label为center label，且固定`wlh`，因此`dimention`固定为`[4, 1.8, 1.5]`

对于`alpha`、`occluded`和`truncated`、`rotation_y`来说，由于`RADIal`数据集中并没有相关`label`，因此赋值为常数。

对于`location`，需要将`label`中的数据


