---
layout:     post
title:      Echofusion
subtitle:   
date:       2024-03-12
author:     LiuLiu
header-img: img/post-bg-debug.png
catalog: true
tags:
    - Code
---



# 1 自己实现
1. 将PolarFormer的MutiCamera改为Single。将载入的nuScenes数据格式改为RADIal

``/home/liuliu/PolarFormer/mmdetection3d/mmdet3d/datasets`

mmdet3d中的数据集处理主要依赖于`mmdetection3d/mmdet3d/datasets`
https://www.cnblogs.com/notesbyY/p/13564454.html



1. 数据集构建
2. annofile生成.pkl文件（create data）
3. dataset建立
4. pipeline建立
5. datalodaer建立

## 1.1 数据集构建
将RADIal数据集的格式改为Kitti的格式。
```cardlink
url: https://mmdetection3d.readthedocs.io/zh-cn/stable/advanced_guides/customize_dataset.html
title: "自定义数据集 — MMDetection3D 1.3.0 文档"
host: mmdetection3d.readthedocs.io
```


```cardlink
url: https://mmdetection3d.readthedocs.io/en/v0.17.3/data_preparation.html
title: "Dataset Preparation — MMDetection3D 0.17.3 documentation"
host: mmdetection3d.readthedocs.io
```

按照官方提供的自定义数据集构建
按照框架top-down进行数据集转换代码构建
```python
# 生成文件结构
def generate_file_structure():

# 生成ImageSets的txt文件
def generate_ImageSets_txt():

# 生成calibration.txt文件
def generate_calibration_txt():

# 将image和fft中的图片复制到images文件夹中
def copy_images():

# 将label.csv文件转化为txt文件
def convert_label_csv_to_txt():
```


## 1.2 Modal
### 1.2.1 Bug

```cardlink
url: https://github.com/facebookresearch/detectron2/issues/3972
title: "RuntimeError: Default process group has not been initialized, please make sure to call init_process_group. · Issue #3972 · facebookresearch/detectron2"
description: "I have tried to train detectron2 using LazyConfig on single GPU but I encountered File \"/home/user/.conda/envs/default/lib/python3.9/site-packages/torch/distributed/distributed_c10d.py\", line 358, ..."
host: github.com
favicon: https://github.githubassets.com/favicons/favicon.svg
image: https://opengraph.githubassets.com/c38a64f60867dd92fcfabbf10be653d34b5474f368a552f387a8ad5a88636311/facebookresearch/detectron2/issues/3972
```


```cardlink
url: https://blog.csdn.net/weixin_44246009/article/details/119426147
title: "【PyTorch】RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasSgemm()_Anova.YJ的博客-CSDN博客"
description: "文章浏览阅读1.2w次，点赞14次，收藏7次。CUDA error: CUBLAS_STATUS_INVALID_VALUE_cuda error: cublas_status_invalid_value when calling `cublassgemm( handle, opa, opb, m, n, k, α, a, lda, b, ldb, β, c, ldc)`"
host: blog.csdn.net
```

```cardlink
url: https://zhuanlan.zhihu.com/p/77644792
title: "Deformable Convolution  v1, v2 总结"
description: "最近在复现其它文章时，发现他们用了 DCN 的网络。这里就总结下 Jifeng Dai 的关于 Deformable Convolution 的这两篇文章。文章挺有 insight 的。 Deformable Conv v1这篇文章其实比较老了，是 2017 年 5 月出的 1…"
host: zhuanlan.zhihu.com
image: https://pic1.zhimg.com/v2-b277c8ef06fcbf5012ca5d411aca67af_720w.jpg?source=172ae18b
```

# 2 官方代码
## 2.1 环境构建

按照官方代码中的`install.md`进行环境构建
* 创建虚拟环境
* 下载`pytorch`
* 下载`mmcv`、`mmdetection`、`mmsegmentation`、`mmdetection3d`
* 下载`RADIal`信号处理模块
* 下载修改后的目标检测`Evaluation`模块
* 下载其他依赖

### 2.1.1 出现问题：
#### 2.1.1.1 依赖不匹配
在下载`mmdetection3d`和`cupy-cuda111`时出现了环境依赖不匹配问题。`mmdetection3d`所需要的`numpy`版本在`1.20.0`以下，而`cupy-cuda111`所需要的版本在`1.20.0`以上。然后环境分别用`pip`和`conda`下载了两个不同版本的`numpy`

```cardlink
url: https://blog.csdn.net/qq_49030008/article/details/124708108
title: "利用torch安装CuPy，使得numpy(np)在CUDA(gpu)上加速。_numpy cuda-CSDN博客"
description: "文章浏览阅读7.3k次，点赞6次，收藏32次。前言标题名字是很奇怪，没关系，重点看内容。正常安装CupPy需要较为复杂的cuda环境配置，可以参考文章——UDA环境配置。如果你觉得去官网安装CUDA Toolkit太麻烦，那我们可以利用pyotch的cudatookit来保证Cupy的正常运行。正文CuPy官网	官网给出了详细的安装操作，但是需要手动安装CUDA Toolkit，如果你没有实践过，这也许会比较难。官网给出了相应版本对应的安装命令：我电脑是11.1，所以这里执行下面命令即可pip install cupy-cuda_numpy cuda"
host: blog.csdn.net
```


#### 2.1.1.2 版本不匹配
在运行数据集准备时出现bug
```bash
(echofusion) liuliu@liuliu-Legion-Y9000P-IAH7H:~/EchoFusion$ python tools/radial/convert_radial.py --root-path ./data/radial --out-dir data/radial_kitti_format
Traceback (most recent call last):
  File "/home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy/__init__.py", line 18, in <module>
    from cupy import _core  # NOQA
  File "/home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy/_core/__init__.py", line 3, in <module>
    from cupy._core import core  # NOQA
  File "cupy/_core/core.pyx", line 1, in init cupy._core.core
  File "/home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy/cuda/__init__.py", line 8, in <module>
    from cupy.cuda import compiler  # NOQA
  File "/home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy/cuda/compiler.py", line 13, in <module>
    from cupy.cuda import device
  File "cupy/cuda/device.pyx", line 1, in init cupy.cuda.device
ImportError: /home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy_backends/cuda/api/runtime.cpython-37m-x86_64-linux-gnu.so: undefined symbol: cudaMemPoolCreate, version libcudart.so.11.0

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "tools/radial/convert_radial.py", line 11, in <module>
    from rpl import RadarSignalProcessing
  File "/home/liuliu/EchoFusion/tools/radial/rpl.py", line 2, in <module>
    import cupy as cp
  File "/home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy/__init__.py", line 27, in <module>
    ''') from exc
ImportError: 
================================================================
Failed to import CuPy.

If you installed CuPy via wheels (cupy-cudaXXX or cupy-rocm-X-X), make sure that the package matches with the version of CUDA or ROCm installed.

On Linux, you may need to set LD_LIBRARY_PATH environment variable depending on how you installed CUDA/ROCm.
On Windows, try setting CUDA_PATH environment variable.

Check the Installation Guide for details:
  https://docs.cupy.dev/en/latest/install.html

Original error:
  ImportError: /home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy_backends/cuda/api/runtime.cpython-37m-x86_64-linux-gnu.so: undefined symbol: cudaMemPoolCreate, version libcudart.so.11.0
================================================================
```
查看mmcv常见问题

```cardlink
url: https://mmcv.readthedocs.io/zh-cn/latest/faq.html
title: "常见问题 — mmcv 2.1.0 文档"
host: mmcv.readthedocs.io
```

文档中显示
```ad-note
- “undefined symbol” 或者 “cannot open xxx.so”
    
    1. 如果符号和 CUDA/C++ 相关（例如：libcudart.so 或者 GLIBCXX），请检查 CUDA/GCC 运行时的版本是否和编译 mmcv 的一致
        
    2. 如果符号和 PyTorch 相关（例如：符号包含 caffe、aten 和 TH），请检查 PyTorch 运行时的版本是否和编译 mmcv 的一致
        
    3. 运行 `python mmdet/utils/collect_env.py` 以检查 PyTorch、torchvision 和 MMCV 构建和运行的环境是否相同
```

而我的问题是
```bash
Original error:
  ImportError: /home/anaconda3/envs/echofusion/lib/python3.7/site-packages/cupy_backends/cuda/api/runtime.cpython-37m-x86_64-linux-gnu.so: undefined symbol: cudaMemPoolCreate, version libcudart.so.11.0
```
我的bug符号为`at`而不是`aten`，但查找了半天资料也没有找到关于`at`的bug信息。因此以`pytorch`版本不匹配尝试解决bug

查看`pytorch`版本号和mmcv版本号
发现`pytorch`版本号玩儿哦`1.9.1`，而`mmcv`版本号所依赖的`pytorch`版本号为`1.9.0`，所以将`pytorch`卸载重新安装。
安装后发现[[#2.1.1.1 依赖不匹配]] 依然没有解决，希望能够安装版本更新的`pytorch`

查看自己的[[cuda版本]]之间的关系之后
首先查看自己的driver cuda版本
```bash
(base) liuliu@liuliu-Legion-Y9000P-IAH7H:~/EchoFusion$ nvidia-smi
Mon Jan 15 11:14:41 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.223.02   Driver Version: 470.223.02   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   51C    P0    26W /  N/A |   2898MiB /  5921MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

```

查看自己的Runtime cuda版本
```bash
(base) liuliu@liuliu-Legion-Y9000P-IAH7H:~/EchoFusion$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:15:46_PDT_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0
```

查看自己的`pytorch cuda`版本
```bash
(echofusion) liuliu@liuliu-Legion-Y9000P-IAH7H:~/EchoFusion$ python
Python 3.7.13 (default, Oct 20 2022, 14:56:53) 
[GCC 11.2.0] :: Intel Corporation on linux
Type "help", "copyright", "credits" or "license" for more information.
Intel(R) Distribution for Python is brought to you by Intel Corporation.
Please check out: https://software.intel.com/en-us/python-distribution
>>> import torch
>>> torch.version.cuda
'11.1'
```

发现自己的`pytorch cuda`版本与`nvcc`不同，为保险，希望更改为`pytorch+cuda113`版本
因此，重新新建环境

```bash
conda create -n echo python=3.7 -y 
conda activate echofusion
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```


之后根据`pytorch`与`mmcv`匹配版本对`mmcv`进行安装，由于代码安装的是`mmcv1.4.0`，因此要根据`mmcv`版本去找合适的`pytorch`版本

[版本匹配](https://www.zywvvd.com/notes/environment/cuda/mmcv-package-install/mmcv-package-install/)查找

| CUDA | torch 1.11 | torch 1.10 | torch 1.9 | torch 1.8 | torch 1.7 | torch 1.6 | torch 1.5 |
| ---- | ---------- | ---------- | --------- | --------- | --------- | --------- | --------- |
| 11.5 | √          |            |           |           |           |           |           |
| 11.3 | √          | √          |           |           |           |           |           |
| 11.1 |            | √          | √         | √         |           |           |           |
| 11.0 |            |            |           |           | √         |           |           |
| 10.2 | √          | √          | √         | √         | √         | √         | √         |
| 10.1 |            |            |           | √         | √         | √         | √         |
| 9.2  |            |            |           |           | √         | √         | √         |
| cpu  | √          | √          | √         | √         | √         | √         | √         |

本来准备安装`pytorch1.11.0`的，结果在[网址](https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html)中查看没有`mmcv1.4.0`版本，因此安装`pytorch1.10.0`

之后按教程安装，比较顺利。注意`cupy`安装`cupy-cuda113`
## 2.2 数据集准备
### 2.2.1 出现问题
#### 2.2.1.1 mmdetection3d报错
```bash
echo) liuliu@liuliu-Legion-Y9000P-IAH7H:~/EchoFusion$ python tools/radial/create_data.py radial --root-path /media/liuliu/MyPassport/radial_kitti_format
Traceback (most recent call last):
  File "tools/radial/create_data.py", line 5, in <module>
    from tools.data_converter import kitti_converter as kitti
  File "/home/liuliu/EchoFusion/mmdetection3d/tools/data_converter/kitti_converter.py", line 8, in <module>
    from mmdet3d.core.bbox import box_np_ops
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/core/__init__.py", line 3, in <module>
    from .bbox import *  # noqa: F401, F403
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/core/bbox/__init__.py", line 5, in <module>
    from .iou_calculators import (AxisAlignedBboxOverlaps3D, BboxOverlaps3D,
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/core/bbox/iou_calculators/__init__.py", line 2, in <module>
    from .iou3d_calculator import (AxisAlignedBboxOverlaps3D, BboxOverlaps3D,
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py", line 6, in <module>
    from ..structures import get_box_type
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/core/bbox/structures/__init__.py", line 2, in <module>
    from .base_box3d import BaseInstance3DBoxes
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/core/bbox/structures/base_box3d.py", line 6, in <module>
    from mmdet3d.ops.iou3d import iou3d_cuda
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/ops/__init__.py", line 6, in <module>
    from .ball_query import ball_query
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/ops/ball_query/__init__.py", line 1, in <module>
    from .ball_query import ball_query
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/ops/ball_query/ball_query.py", line 4, in <module>
    from . import ball_query_ext
ImportError: /home/liuliu/EchoFusion/mmdetection3d/mmdet3d/ops/ball_query/ball_query_ext.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZNK2at6Tensor8data_ptrIfEEPT_v
```

发现报错，尝试卸载重新安装numpy，因为之前的numpy仍然有版本不匹配问题
发现
```bash
(echo) liuliu@liuliu-Legion-Y9000P-IAH7H:~/EchoFusion$ pip install numpy==1.19.3
Collecting numpy==1.19.3
  Using cached numpy-1.19.3-cp37-cp37m-manylinux2010_x86_64.whl (14.9 MB)
WARNING: Error parsing requirements for numpy: [Errno 2] No such file or directory: '/home/anaconda3/envs/echo/lib/python3.7/site-packages/numpy-1.21.6.dist-info/METADATA'
Installing collected packages: numpy
  Attempting uninstall: numpy
    WARNING: No metadata found in /home/anaconda3/envs/echo/lib/python3.7/site-packages
    Found existing installation: numpy 1.21.6
ERROR: Cannot uninstall numpy 1.21.6, RECORD file not found. You might be able to recover from this via: 'pip install --force-reinstall --no-deps numpy==1.21.6'.
```
无法卸载numpy，也无法重新安装
查询资料

```cardlink
url: https://stackoverflow.com/questions/68886239/cannot-uninstall-numpy-1-21-2-record-file-not-found
title: "Cannot uninstall numpy 1.21.2, RECORD file not found"
description: "I encountered a problem while installingpip install pytorch-nlpThe erro is as follow:ERROR: Could n`ot install packages due to an OSError: [Errno 2] No such file or directory: 'c:\\users\\pcpcpc\\"
host: stackoverflow.com
image: https://cdn.sstatic.net/Sites/stackoverflow/Img/apple-touch-icon@2.png?v=73d79a89bded
```
重新安装后依然没有解决问题
决定重新编译mmdetection3d，编译后问题解决。

## 2.3 模型训练
### 2.3.1 单机多卡更改为单机单卡
[[Pytorch分布式训练]]

新建`run_radial_onegpu.sh`和`train.sh`
```
Traceback (most recent call last):
  File "tools/train.py", line 257, in <module>
    main()
  File "tools/train.py", line 167, in main
    init_dist(args.launcher, **cfg.dist_params)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/dist_utils.py", line 18, in init_dist
    _init_dist_pytorch(backend, **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/dist_utils.py", line 29, in _init_dist_pytorch
    rank = int(os.environ['RANK'])
  File "/home/anaconda3/envs/echo/lib/python3.7/os.py", line 681, in __getitem__
    raise KeyError(key) from None
KeyError: 'RANK'
```

### 2.3.2 Bug
#### 2.3.2.1 第一个
```
(echo) liuliu@liuliu-Legion-Y9000P-IAH7H:~/EchoFusion$ bash run_radial_onegpu.sh 
projects.mmdet3d_plugin
Traceback (most recent call last):
  File "tools/train.py", line 257, in <module>
    main()
  File "tools/train.py", line 175, in main
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/utils/config.py", line 541, in dump
    f.write(self.pretty_text)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/utils/config.py", line 496, in pretty_text
    text, _ = FormatCode(text, style_config=yapf_style, verify=True)
TypeError: FormatCode() got an unexpected keyword argument 'verify'

```
[Fetching Data#has9](https://github.com/open-mmlab/mmdetection/issues/10962)

#### 2.3.2.2 第二个
```
Traceback (most recent call last):
  File "tools/train.py", line 257, in <module>
    main()
  File "tools/train.py", line 253, in main
    meta=meta)
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/apis/train.py", line 35, in train_model
    meta=meta)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmdet/apis/train.py", line 203, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py", line 108, in run
    self.call_hook('before_run')
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/base_runner.py", line 307, in call_hook
    getattr(hook, fn_name)(self)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/dist_utils.py", line 94, in wrapper
    return func(*args, **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/hooks/logger/tensorboard.py", line 35, in before_run
    from torch.utils.tensorboard import SummaryWriter
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module>
    LooseVersion = distutils.version.LooseVersion
AttributeError: module 'distutils' has no attribute 'version'

```

```cardlink
url: https://github.com/pytorch/pytorch/issues/69894
title: "AttributeError: module 'distutils' has no attribute 'version' : with setuptools 59.6.0 · Issue #69894 · pytorch/pytorch"
description: "🐛 Describe the bug # python3 -m pip install --upgrade setuptools torch tensorboard` # python3 Python 3.8.10 (default, Sep 28 2021, 16:10:42) [GCC 9.3.0] on linux Type \"help\", \"copyright\", \"credits\"..."
host: github.com
favicon: https://github.githubassets.com/favicons/favicon.svg
image: https://opengraph.githubassets.com/ee740c5396599341885d334219b3c1489ad0e309d6a73d102b26611a29215f77/pytorch/pytorch/issues/69894
```

```cardlink
url: https://github.com/pytorch/pytorch/pull/69823
title: "Use `packaging.version` instead of `distutils.version` by asi1024 · Pull Request #69823 · pytorch/pytorch"
description: "This PR fixes to use pkg_resources.packaging.version instead of distutils.version.distutils deprecated version classes from setuptools 59.6.0 (pypa/distutils#75), and PyTorch raises an error: Attr..."
host: github.com
favicon: https://github.githubassets.com/favicons/favicon.svg
image: https://opengraph.githubassets.com/aff54101108943d759dab7cbb25efd4f6c38a80d6189e343da8adbc104c48781/pytorch/pytorch/pull/69823
```

#### 2.3.2.3 第三个
```
raceback (most recent call last):
  File "tools/train.py", line 257, in <module>
    main()
  File "tools/train.py", line 253, in main
    meta=meta)
  File "/home/liuliu/EchoFusion/mmdetection3d/mmdet3d/apis/train.py", line 35, in train_model
    meta=meta)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmdet/apis/train.py", line 203, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py", line 50, in train
    self.run_iter(data_batch, train_mode=True, **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py", line 30, in run_iter
    **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmdet/models/detectors/base.py", line 248, in train_step
    losses = self(**data)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 186, in new_func
    return old_func(*args, **kwargs)
  File "/home/liuliu/EchoFusion/projects/mmdet3d_plugin/models/detectors/echofusion_rt_img.py", line 146, in forward
    return self.forward_train(**kwargs)
  File "/home/liuliu/EchoFusion/projects/mmdet3d_plugin/models/detectors/echofusion_rt_img.py", line 190, in forward_train
    img_feats = self.extract_feat(img=img, img_metas=img_metas, radars_rt=radars_rt, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 98, in new_func
    return old_func(*args, **kwargs)
  File "/home/liuliu/EchoFusion/projects/mmdet3d_plugin/models/detectors/echofusion_rt_img.py", line 106, in extract_feat
    radar_feats = self.extract_radar_feat(radars_rt[0]) # use only one radar
  File "/home/liuliu/EchoFusion/projects/mmdet3d_plugin/models/detectors/echofusion_rt_img.py", line 71, in extract_radar_feat
    radar_feats = self.pts_backbone(radars_rt)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/mmdet/models/backbones/resnet.py", line 637, in forward
    x = self.norm1(x)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/nn/modules/batchnorm.py", line 732, in forward
    world_size = torch.distributed.get_world_size(process_group)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 845, in get_world_size
    return _get_group_size(group)
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 306, in _get_group_size
    default_pg = _get_default_group()
  File "/home/anaconda3/envs/echo/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 411, in _get_default_group
    "Default process group has not been initialized, "
RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.

```
[Fetching Data#7ih0](https://github.com/facebookresearch/detectron2/issues/3972)

解决之后发现自己的计算器显存不够，只能用服务器来跑

## 2.4 服务器环境构建
### 2.4.1 代码
### 2.4.2 环境
#### 2.4.2.1 环境迁移

```cardlink
url: https://zhuanlan.zhihu.com/p/562302448
title: "使用conda pack打包并迁移现有环境到新服务器上"
description: "在源服务器上关于conda pack的使用可以参考： https://conda.github.io/conda-pack/安装conda packpip install conda-pack注意，这时候如果报错，比如： FileNotFoundError: [Errno 2] No usable temporary direct…"
host: zhuanlan.zhihu.com
```

```cardlink
url: https://blog.csdn.net/weixin_48030475/article/details/135155483?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-135155483-blog-134430975.235%5Ev40%5Epc_relevant_3m_sort_dl_base2&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-135155483-blog-134430975.235%5Ev40%5Epc_relevant_3m_sort_dl_base2&utm_relevant_index=1
title: "conda安装慢：\ WARNING conda.models.version:get_matcher(528): Using .* with relational operator is super_using .* with relational operator is superfluous-CSDN博客"
description: "文章浏览阅读633次，点赞10次，收藏9次。conda安装pytorch时特别慢,出现。_using .* with relational operator is superfluous"
host: blog.csdn.net
```
发现cuda不匹配
```bash
    RuntimeError:
    The detected CUDA version (10.1) mismatches the version that was used to compile
    PyTorch (11.3). Please make sure to use the same CUDA versions.

```
#### 2.4.2.2 重建环境

```cardlink
url: https://stackoverflow.com/questions/74781771/how-we-can-resolve-solving-environment-failed-with-initial-frozen-solve-retry
title: "How we can resolve \"Solving environment: failed with initial frozen solve. Retrying with flexible solve.\" issue while installing the new conda package"
description: "I have tried to install new package in conda for windows using the following command:conda install -c conda-forge python-pdfkitbut got the following error:Collecting package metadata ("
host: stackoverflow.com
image: https://cdn.sstatic.net/Sites/stackoverflow/Img/apple-touch-icon@2.png?v=73d79a89bded
```

```cardlink
url: https://blog.csdn.net/wacebb/article/details/133419770
title: "【已解决】WARNING conda.models.version:get_matcher(556): Using .* with relational operator is superfluous-CSDN博客"
description: "文章浏览阅读6.9k次，点赞9次，收藏8次。用conda安装东西很慢，并且出现以下信息。_warning conda.models.version:get_matcher(556): using .* with relational oper"
host: blog.csdn.net
```

```cardlink
url: https://pytorch.org/get-started/previous-versions/
title: "Previous PyTorch Versions"
description: "Installing previous versions of PyTorch"
host: pytorch.org
image: https://pytorch.org/assets/images/social-share.jpg
```

```cardlink
url: https://stackoverflow.com/questions/18356502/github-failed-to-connect-to-github-443-windows-failed-to-connect-to-github
title: "GitHub - failed to connect to github 443 windows/ Failed to connect to gitHub - No Error"
description: "I installed Git to get the latest version of Angular. When I tried to rungit clone https://github.com/angular/angular-phonecat.gitI got:failed to connect to github 443 errorI even triedgit ..."
host: stackoverflow.com
image: https://cdn.sstatic.net/Sites/stackoverflow/Img/apple-touch-icon@2.png?v=73d79a89bded
```
[Fetching Data#swtr](https://github.com/Turoad/lanedet/issues/29)
conda安装报错，只能用pip安装

```cardlink
url: https://stackoverflow.com/questions/74781771/how-we-can-resolve-solving-environment-failed-with-initial-frozen-solve-retry
title: "How we can resolve \"Solving environment: failed with initial frozen solve. Retrying with flexible solve.\" issue while installing the new conda package"
description: "I have tried to install new package in conda for windows using the following command:conda install -c conda-forge python-pdfkitbut got the following error:Collecting package metadata ("
host: stackoverflow.com
image: https://cdn.sstatic.net/Sites/stackoverflow/Img/apple-touch-icon@2.png?v=73d79a89bded
```

20240116 21：25 安装到`conda install -c intel intel-aikit-modin`
##### 2.4.2.2.1 安装git

```cardlink
url: https://pip.pypa.io/en/stable/topics/vcs-support/
title: "VCS Support - pip documentation v23.3.2"
host: pip.pypa.io
```

```bash
(echofusion) liuliu2023@ubuntu:~/EchoFusion$ pip install --upgrade git+https://github.com/klintan/pypcd.git
Collecting git+https://github.com/klintan/pypcd.git
  Cloning https://github.com/klintan/pypcd.git to /tmp/pip-req-build-kt37p9c2
  Running command git clone --filter=blob:none --quiet https://github.com/klintan/pypcd.git /tmp/pip-req-build-kt37p9c2
  fatal: unable to access 'https://github.com/klintan/pypcd.git/': Failed to connect to github.com port 443: Connection timed out
  error: subprocess-exited-with-error
  
  × git clone --filter=blob:none --quiet https://github.com/klintan/pypcd.git /tmp/pip-req-build-kt37p9c2 did not run successfully.
  │ exit code: 128
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× git clone --filter=blob:none --quiet https://github.com/klintan/pypcd.git /tmp/pip-req-build-kt37p9c2 did not run successfully.
│ exit code: 128
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```

```cardlink
url: https://blog.csdn.net/qq_50935651/article/details/123059602
title: "pip install git+https:XXX 安装失败_.在下载的文件夹下执行 :python setup.py sdist-CSDN博客"
description: "文章浏览阅读1w次，点赞20次，收藏60次。在没有vpn无法访问外网的时候，pip install git+https:XXX通常无法完成安装。下面是我尝试后成功的一种安装方式：1.先要下载好原文件（这时候文件夹里往往有一个setup.py文件，但是有些时候并不能简单的python setup.py）2.在下载的文件夹下执行 ：pythonsetup.pysdist3.然后会多出一个dist的文件夹，打开文件夹便可以看到一个打包好的你需要安装的项目xxx.tar.gz4.然后再 pip install xxx.tar.gz .._.在下载的文件夹下执行 :python setup.py sdist"
host: blog.csdn.net
```

```cardlink
url: https://blog.csdn.net/weixin_41997940/article/details/123607318
title: "Linux设置密钥登录（非root用户）-CSDN博客"
description: "文章浏览阅读3.9k次，点赞2次，收藏7次。第一步 生成密钥对ssh-keygen -t rsa    生成了.ssh文件，路径：/home/用户名/第二步 检查文件进入.ssh文件夹，检查authorized_keys文件是否存在，如果有不用操作，没有则创建一个，将id_rsa追加进去。cd .sshtouch authorized_keys			cat id_rsa.pub >> authorized_keys  第三步 修改权限   修改.ssh..."
host: blog.csdn.net
```

```cardlink
url: https://blog.csdn.net/hamupp/article/details/114581036
title: "GIT检查是否SSH通畅的指令_查看ssh与git是否建立连接-CSDN博客"
description: "文章浏览阅读4k次。命令行输入以下指令（windows、mac同）。ssh -T -v git@github.com将会打印出一串日志。如果是通的，将会在最后显示你的github账号名称（绿色框），如下：_查看ssh与git是否建立连接"
host: blog.csdn.net
```

问题：在安装
```bash
conda install -c intel intel-aikit-modin
pip3 install --upgrade git+https://github.com/klintan/pypcd.git
```
时出错

解决方法
pypcd：下载tar.gz传输到服务器上install的
itel：
```cardlink
url: https://anaconda.org/intel/intel-aikit-modin
title: "Intel Aikit Modin :: Anaconda.org"
host: anaconda.org
```
```bash
(echofusion) liuliu2023@ubuntu:/raid/liuliu$ conda install intel/label/validation::intel-aikit-modin
Collecting package metadata (current_repodata.json): done
Solving environment: \ 
The environment is inconsistent, please check the package plan carefully
The following packages are causing the inconsistency:

  - conda-forge/linux-64::python==3.7.12=hf930737_100_cpython
  - conda-forge/noarch::wheel==0.42.0=pyhd8ed1ab_0
  - conda-forge/noarch::pip==23.3.2=pyhd8ed1ab_0
\ failed with initial frozen solve. Retrying with flexible solve.
```
最后是下载file传输上去安装的
file搜索：

```cardlink
url: https://anaconda.org/
title: ":: Anaconda.org"
host: anaconda.org
```


#### 2.4.2.3 mmdetection3d编译

[Fetching Data#cxqt](https://github.com/pytorch/vision/issues/3261)

```bash
/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/torch/utils/cpp_extension.py:782: UserWarning: The detected CUDA version (10.1) has a minor version mismatch with the version that was used to compile PyTorch (10.2). Most likely this shouldn't be a problem.
```
看起来没问题，但是在运行代码的时候就有问题
```bash
(echofusion) liuliu2023@ubuntu:~/EchoFusion$ bash run_radial_onegpu.sh 
Traceback (most recent call last):
  File "tools/train.py", line 18, in <module>
    from mmdet3d.apis import train_model
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/apis/__init__.py", line 2, in <module>
    from .inference import (convert_SyncBN, inference_detector,
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/apis/inference.py", line 11, in <module>
    from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/core/__init__.py", line 3, in <module>
    from .bbox import *  # noqa: F401, F403
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/core/bbox/__init__.py", line 5, in <module>
    from .iou_calculators import (AxisAlignedBboxOverlaps3D, BboxOverlaps3D,
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/core/bbox/iou_calculators/__init__.py", line 2, in <module>
    from .iou3d_calculator import (AxisAlignedBboxOverlaps3D, BboxOverlaps3D,
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py", line 6, in <module>
    from ..structures import get_box_type
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/core/bbox/structures/__init__.py", line 2, in <module>
    from .base_box3d import BaseInstance3DBoxes
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/core/bbox/structures/base_box3d.py", line 6, in <module>
    from mmdet3d.ops.iou3d import iou3d_cuda
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/ops/__init__.py", line 6, in <module>
    from .ball_query import ball_query
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/ops/ball_query/__init__.py", line 1, in <module>
    from .ball_query import ball_query
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/ops/ball_query/ball_query.py", line 4, in <module>
    from . import ball_query_ext
ImportError: libcudart.so.10.1: cannot open shared object file: No such file or directory

```
查看环境版本
```bash
(echofusion) liuliu2023@ubuntu:~/EchoFusion/mmdetection3d$ python mmdet3d/utils/collect_env.py 
sys.platform: linux
Python: 3.7.12 | packaged by conda-forge | (default, Oct 26 2021, 06:08:21) [GCC 9.4.0]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: Tesla V100-SXM2-32GB
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 10.1, V10.1.243
GCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
PyTorch: 1.10.0+cu102
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 10.2
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70
  - CuDNN 7.6.5
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=10.2, CUDNN_VERSION=7.6.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 

TorchVision: 0.11.0+cu102
OpenCV: 4.9.0
MMCV: 1.4.0
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 10.2
MMDetection: 2.19.0
MMSegmentation: 0.20.0
MMDetection3D: 0.17.3+940e429
```
以为是GCC问题，因为按道理来说，CUDA Compile为10.1和RunTime CUDA 为10.2不矛盾。(？)
然后查了很多资料，怎么在虚拟环境中更改GCC的版本，因为
`Python: 3.7.12 | packaged by conda-forge | (default, Oct 26 2021, 06:08:21) [GCC 9.4.0]`显示GCC版本为9,高于CUDA10.1所兼容的最高版本。

| CUDA version     | max supported GCC version |
| ---------------- | ------------------------- |
| 11.1, 11.2, 11.3 | 10                        |
| 11               | 9                         |
| 10.1, 10.2       | 8                         |
| 9.2, 10.0        | 7                         |
| 9.0, 9.1         | 6                         |
| 8                | 5.3                       |
| 7                | 4.9                       |
| 5.5, 6           | 4.8                       |
| 4.2, 5           | 4.6                       |
| 4.1              | 4.5                       |
| 4.0              | 4.4                       |
然后尝试：
1.[非root用户改变gcc版本](https://blog.csdn.net/Fhujinwu/article/details/113786909)
其中gcc版本网站为(https://ftp.gnu.org/gnu/gcc/)
2.[conda虚拟环境安装gcc](https://blog.csdn.net/qq_35752161/article/details/111345572)

然后其实不是这个问题，还是CUDA版本不匹配，但CUDA10.1不适配Pytorch1.9.0以上的版本。因此，准备在服务器上进行[非root用户安装CUDA](https://zhuanlan.zhihu.com/p/198161777)
更改环境变量的时候参考这篇：

```cardlink
url: https://www.cnblogs.com/wuliytTaotao/p/12169315.html
title: "【tf.keras】Linux 非 root 用户安装 CUDA 和 cuDNN - wuliytTaotao - 博客园"
description: "TF 2.0 for Linux 使用时报错：Loaded runtime CuDNN library: 7.4.1 but source was compiled with: 7.6.0.  解决方法：升级 cuDNN。非 root 用户可以在自己目录下安装 CUDA 和新版本的 cuDNN 来解"
host: www.cnblogs.com
```

由于之前配置的Pytorch、mmcv环境都是基于CUDA10.2的，因此准备先配置CUDA10.2的环境
不同版本的CUDA Toolkit：[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

### 2.4.3 数据
scp传输
但好慢，可以查一下有没有更快的传输方式

# 3 代码解析
## 3.1 数据集处理
![[out.png]]
### 3.1.1 将radial转化为kitti格式
#### 3.1.1.1 导入包库
```python
from DBReader.DBReader import SyncReader # 读取同步后的数据
from mmcv import ProgressBar
from concurrent import futures as futures # 提供异步执行可调用对象高层接口
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
import os
import argparse
from rpl import RadarSignalProcessing # 雷达信号处理

Sequences = {'Validation':['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
            'Test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}

```
#### 3.1.1.2 生成文件夹
```python
def create_dir(out_dir, clear=False): # 生成kitti格式的文件夹
    """Create data structure."""
    # check and create files
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'ImageSets')):
        os.makedirs(os.path.join(out_dir, 'ImageSets'))
    if not os.path.exists(os.path.join(out_dir, 'images')):
        os.makedirs(os.path.join(out_dir, 'images'))
    if not os.path.exists(os.path.join(out_dir, 'radars_adc')):
        os.makedirs(os.path.join(out_dir, 'radars_adc'))
    if not os.path.exists(os.path.join(out_dir, 'radars_rt')):
        os.makedirs(os.path.join(out_dir, 'radars_rt'))
    if not os.path.exists(os.path.join(out_dir, 'radars_pcd')):
        os.makedirs(os.path.join(out_dir, 'radars_pcd'))
    if not os.path.exists(os.path.join(out_dir, 'radars_ra')):
        os.makedirs(os.path.join(out_dir, 'radars_ra'))
    if not os.path.exists(os.path.join(out_dir, 'radars_rd')):
        os.makedirs(os.path.join(out_dir, 'radars_rd'))
    if not os.path.exists(os.path.join(out_dir, 'lidars')):
        os.makedirs(os.path.join(out_dir, 'lidars'))
    if not os.path.exists(os.path.join(out_dir, 'labels')):
        os.makedirs(os.path.join(out_dir, 'labels'))
    
    if clear:
        # clear txt in ImageSets
        for file in os.listdir(os.path.join(out_dir, 'ImageSets')):
            os.remove(os.path.join(out_dir, 'ImageSets', file))
```
#### 3.1.1.3 数据转换
```python
def convert(root_dir, 
            out_dir,
            clear=False,
            num_worker=8): # 并行处理
    """ Parallelized conversion process. """
    root_dir = args.root_path
    out_dir = args.out_dir
    create_dir(out_dir, clear)  # note that ImageSets will be cleared

    # read labels from csv
    labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
    unique_ids = np.unique(labels[:,0]).tolist()
    label_dict = {}
    for i,ids in enumerate(unique_ids):
        sample_ids = np.where(labels[:,0]==ids)[0]
        label_dict[ids]=sample_ids
    sample_keys = list(label_dict.keys())
    pkbar = ProgressBar(len(unique_ids))

    def map_func(frame_id):
        # From the sample id, retrieve all the labels ids
        entries_indexes = label_dict[frame_id]
        # Get the objects labels
        box_infos = labels[entries_indexes]

        record_name = box_infos[:, -3][0] #记录名称
        data_root = os.path.join(root_dir, record_name)
        db = SyncReader(data_root, tolerance=20000, silent=True); # 读取该记录名称的所有同步数据

        idx = box_infos[:, -2][0] # 读取sample的id
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

        # save radar as binary
        clibration_path = os.path.join(root_dir, 'CalibrationTable.npy')

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

        # save labels as txt
        label_name = os.path.join(out_dir, 'labels', '%06d.txt' % frame_id)
        np.savetxt(label_name, box_infos, fmt='%d %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %s %d %d')

        pkbar.update()

    with futures.ThreadPoolExecutor(num_worker) as executor: # 异步执行
        image_infos = executor.map(map_func, unique_ids)
    # for frame_id in unique_ids:
    #     map_func(frame_id)

    print('\nConversion Done!')
```
其中雷达信号处理模块在rpl.py中，属于radial数据集中的处理模块
ADC和RD处理模块已经看懂，PCD处理模块已经看到CA_CFAR部分，而找到Tx_0的位置还存在疑惑，设计到信号处理和MIMO原理。
### 3.1.2 生成信息pickle文件
生成pickle文件一共由三个py文件组成
`create_data.py` 

### 3.1.3 生成3D label
与2Dlabel的区别在于，没有2dbox，dimention、location、rotation都有了更精确的数值。

**疑问：如何产生函数调用图，光用脑壳记不住啊**
1. **pyreverse(pylint)+graphviz**   感觉好像不太行
2. 画时序图
3. doxygen+graphviz：doxygen好像是写文档注释的！
4. **pycallgraph**：貌似不太支持python3,且对于包库引用不太友好
5. **code2flow**
ps:标粗的包库都可以使用，但最丝滑的是code2flow

```cardlink
url: https://www.cnblogs.com/54chensongxia/p/13236965.html
title: "程序员必备画图技能之——时序图 - 程序员自由之路 - 博客园"
description: "什么是时序图 时序图(Sequence Diagram)，又名序列图、循序图，是一种UML交互图。它通过描述对象之间发送消息的时间顺序显示多个对象之间的动态协作。 使用场景 时序图的使用场景非常广泛，几乎各行各业都可以使用。当然，作为一个软件工作者，我这边主要列举和软件开发有关的场景。 1. 梳理业"
host: www.cnblogs.com
```

```cardlink
url: https://blog.csdn.net/benkaoya/article/details/79750745
title: "绘制函数调用图（call graph）（1）：专栏开篇-CSDN博客"
description: "文章浏览阅读3.3w次，点赞11次，收藏59次。绘制函数调用关系图（call graph），对开发人员理解源码有非常大的帮助，特别是在以下情况：大型项目，庞杂的代码量；项目文档缺失，特别是设计文档、流程图的缺失；第三方代码库，如开源项目；检查实际函数调用关系跟规划的设计是否一致，以免出错。绘制函数调用关系图的途径主要有两种，一种是人工手动绘制（很多人应该都有一边看代码（或借助调试工具单步跟踪），一边在纸上画函数调用关系图的经历..._函数调用图"
host: blog.csdn.net
```

```cardlink
url: https://www.doxygen.nl/manual/install.html
title: "My Project: Installation"
host: www.doxygen.nl
favicon: doxygen.ico
```

```cardlink
url: https://stackoverflow.com/questions/13963321/build-a-call-graph-in-python-including-modules-and-functions
title: "Build a Call graph in python including modules and functions?"
description: "I have a bunch of scripts to perform a task. And I really need to know the call graph of the project because it is very confusing. I am not able to execute the code because it needs extra HW and SW..."
host: stackoverflow.com
image: https://cdn.sstatic.net/Sites/stackoverflow/Img/apple-touch-icon@2.png?v=73d79a89bded
```

## 3.2 Train.py
```python
# 导入第三方包库
from __future__ import division # 能够使python3使用python2的包库

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

import sys # vscode调试专用，在终端跑程序时不需要此代码
sys.path.append('/home/liuliu/EchoFusion')
```

```python
#读取参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path') #程序config文件，在泼辣人transformer中使用的是py文件
    parser.add_argument('--work-dir', help='the dir to save logs and models') #程序的log文件放置目录
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='whether to auto resume from the latest checkpoint') # 是否在前序checkpoint基础上进行训练
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group() # 创建互斥组，同一时间只能有一个参数生效
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.') # 是否为 CUDNN 后端设置确定性选项
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args
```

```python
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config) # mmcv 中的配置文件读取函数
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options) # 从命令行参数中更新配置文件
    # import modules from string list.
    if cfg.get('custom_imports', None): # 如果配置文件中有 custom_imports 字段
        from mmcv.utils import import_modules_from_strings # 从字符串中导入模块
        import_modules_from_strings(**cfg['custom_imports']) # 导入模块

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib # 动态导入模块
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path) #  import mmdet3d_plugin
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False): # 如果配置文件中有 cudnn_benchmark 字段
        torch.backends.cudnn.benchmark = True # 优化卷积网络的运行效率 A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # auto-resume
    if args.resume_from is None and args.auto_resume:
        resume_from = find_latest_checkpoint(cfg.work_dir)
        cfg.resume_from = resume_from
    
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config))) # 将配置文件保存到工作目录下
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env() # collect 环境
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')) # 网络实例化 Registry.build() 会根据配置文件中的 type 字段找到对应的类并实例化
    model.init_weights() # 初始化网络参数权重

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
```
没有分析到distributed，因为本机上无法分布式训练

### 3.2.1 加载数据集
pipelines 的逻辑，由三部分组成，load，transforms，和 format
load 相关的 LoadImageFromFile，LoadAnnotations都是字典 results 进去，字典 results 出来。具体代码看下便知，LoadImageFromFile 增加了’filename’，’img’，’img_shape’，’ori_shape’,’pad_shape’,’scale_factor’,’img_norm_cfg’ 字段。其中 img 是 numpy 格式。LoadAnnotations 从 results[’ann_info’] 中解析出 bboxs,masks,labels 等信息。注意 coco 格式的原始解析来自 pycocotools，包括其评估方法，这里关键是字典结构 (这个和模型损失函数，评估等相关，统一结构，使得代码统一)。transforms 中的类作用于字典的 values，也即数据增强。format 中的 DefaultFormatBundle 是将数据转成 mmcv 扩展的容器类格式 DataContainer。另外 Collect 会根据不同任务的不同配置，从 results 中选取只含 keys 的信息生成新的字典，具体看下该类帮助文档。

```cardlink
url: https://nicehuster.github.io/2020/09/04/mmdetection-3/
title: "mmdetection详解指北 (三)"
description: "数据处理数据处理可能是炼丹师接触最为密集的了，因为通常情况，除了数据的离线处理，写个数据类，就可以炼丹了。但本节主要涉及数据的在线处理，更进一步应该是检测分割数据的 pytorch 处理方式。虽然 mmdet 将常用的数据都实现了，而且也实现了中间通用数据格式，但，这和模型，损失函数，性能评估的实现也相关，比如你想把官网的 centernet 完整的改成 mmdet风格，就能看到 (看起来没必要)"
host: nicehuster.github.io
```
疑问：DataContainer是什么？
permute的具体实现，可视化？
### 3.2.2 训练
#### 3.2.2.1 ectract feature
疑问：Mask掩码的作用？
##### 3.2.2.1.1 RADAR_RT
如何解开RT信息?
1. 解开RT信息
2. RadarResNet
3. FPN

##### 3.2.2.1.2 IMG
1. Mask掩码处理
2. ResNet
3. FPN
4. Neck
	1. single image process
		1. position encoding
			1. 疑问 torch.permute 和stack
		2. 三个Transformer Decoder
		3. 转换到BEV视角
	2. 和Radar第一步处理结果一起做三个Transformer（只有Decoder，结构与单独image一样）

#### 3.2.2.2 Fusion

Transformer
	Encoder: BaseTransformerLayer   6layer
	Decoder:DetrTransformerLayer     6layer
