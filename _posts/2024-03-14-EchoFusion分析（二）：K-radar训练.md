---
layout:     post
title:      EchoFusion分析（二）：K-radar训练
subtitle:   "ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 1"
date:       2024-03-14
author:     LiuLiu
header-img: img/post-bg-debug.png
catalog: true
categories: 代码
tags:
    - EchoFusion
---
# 前言
在对K-radar数据集进行训练的时候，发现在第一个epoch训练完进行evaluation的时候，总是会[报错](https://github.com/open-mmlab/mmcv/issues/1969)
```python
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 1
```
但是由于Pytorch分布式DDP训练的时候没有具体报错信息，因此用VSCode进行Dbug。
更改Vscode的配置`launch.json`文件
```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python 调试程序: 当前文件",
            "type": "debugpy",
            "request": "launch",
            // "program": "${file}",
            "program": "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/torch/distributed/launch.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "python": "/home/liuliu2023/anaconda3/envs/echofusion/bin/python",
            // "args":[]
            "args": [
                "--nproc_per_node=3",
                "tools/train.py",
                "./projects/configs/kradar/ra_img_echofusion_kradar_r50_trainval_24e.py",
                "--launcher",
                "pytorch"
            ],
            "env": {"CUDA_VISIBLE_DEVICES":"1,2,3"}
        }
    ]
}
```
# Dbug
由于K-radar的训练时间非常长，而bug信息其实出现在evaluation上，也就是一个epoch train完之后，因此希望程序不进行训练，之间进入到evaluation阶段。但由于底层调用的是mmcv，因此需要修改mmcv代码。

首先将`launch.json`文件中的`“justMycode":"true”`更改为`“justMycode":"false”`

之后将`/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py`中`def train`代码注释掉
```python
def train(self, data_loader, **kwargs):
    self.model.train()
    self.mode = 'train'
    self.data_loader = data_loader
    self._max_iters = self._max_epochs * len(self.data_loader)
    self.call_hook('before_train_epoch')
    time.sleep(2)  # Prevent possible deadlock during epoch transition
    # for i, data_batch in enumerate(self.data_loader):
    #     self._inner_iter = i
    #     self.call_hook('before_train_iter')
    #     self.run_iter(data_batch, train_mode=True, **kwargs)
    #     self.call_hook('after_train_iter')
    #     self._iter += 1

    self.call_hook('after_train_epoch')
    self._epoch += 1
```
Bbug完了之后才更改回来。

## numba与numpy不匹配
首先报错的是
```python
expected dtype object, got 'numpy.dtype[float64]'
```
这个报错似乎跟`numba`这个包库有关，根据[网上](https://stackoverflow.com/questions/75177195/typeerror-expected-dtype-object-got-numpy-dtypefloat32-when-running-statsfo)的说法，是`numba`与`numpy`的版本不匹配有关

有关numba的介绍可以参考：https://lulaoshi.info/gpu/python-cuda/cuda-intro.html

因此，尝试升级`numba`
```
pip install --upgrade numba
```
将`numba`由`0.48.0`升级到了`0.56.4`。但升级后报错
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
mmdet3d 0.17.3 requires numba==0.48.0, but you have numba 0.56.4 which is incompatible.
mmdet3d 0.17.3 requires numpy<1.20.0, but you have numpy 1.20.3 which is incompatible.
```
因此又将`numba`的版本降回来。
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
mmdet3d 0.17.3 requires numpy<1.20.0, but you have numpy 1.20.3 which is incompatible.
```

发现`numpy`的版本也有错，因此查看`numpy`

```bash
(echofusion) (base) liuliu2023@user-NF5468M5:~/EchoFusion$ conda list numpy
# packages in environment at /home/liuliu2023/anaconda3/envs/echofusion:
#
# Name                    Version                   Build  Channel
numpy                     1.19.5                   pypi_0    pypi
numpy-base                1.20.3           py37hf707ed8_2    intel
```

发现`numpy`与`numpy-base`的版本不匹配，重新下载`numpy-base`

两者区别参考：https://stackoverflow.com/questions/50699252/anaconda-environment-installing-packages-numpy-base
```bash
(echofusion) conda install numpy-base==1.19.5
(echofusion) liuliu2023@user-NF5468M5:~/EchoFusion$ conda list numpy
# packages in environment at /home/liuliu2023/anaconda3/envs/echofusion:
#
# Name                    Version                   Build  Channel
numpy                     1.19.5           py37hd5178e2_4  
numpy-base                1.19.5           py37h622ebfc_4 
```

之后Bug `expected dtype object, got 'numpy.dtype[float64]'`被解决

## cuda版本与numba不匹配
报错信息：

```bash
<unnamed> (66, 23): parse expected comma after load's type
...
...
Error numba.cuda.cudadrv.error.NvvmError: Failed to compile, while testing SECOND model with pre-trained model and Config file
```

<details>
<summary>具体报错信息：</summary>

```python
Traceback (most recent call last):
  File "/home/liuliu2023/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/pydevd.py", line 3489, in <module>
    main()
  File "/home/liuliu2023/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/pydevd.py", line 3482, in main
    globals = debugger.run(setup['file'], None, None, is_module)
  File "/home/liuliu2023/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/pydevd.py", line 2510, in run
    return self._exec(is_module, entry_point_fn, module_name, file, globals, locals)
  File "/home/liuliu2023/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/pydevd.py", line 2517, in _exec
    globals = pydevd_runpy.run_path(file, globals, '__main__')
  File "/home/liuliu2023/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 322, in run_path
    pkg_name=pkg_name, script_name=fname)
  File "/home/liuliu2023/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 136, in _run_module_code
    mod_name, mod_spec, pkg_name, script_name)
  File "/home/liuliu2023/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 124, in _run_code
    exec(code, run_globals)
  File "tools/train.py", line 260, in <module>
    main()
  File "tools/train.py", line 256, in main
    meta=meta)
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/apis/train.py", line 35, in train_model
    meta=meta)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmdet/apis/train.py", line 203, in train_detector
    runner.run(data_loaders, cfg.workflow)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py", line 127, in run
    epoch_runner(data_loaders[i], **kwargs)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py", line 54, in train
    self.call_hook('after_train_epoch')
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmcv/runner/base_runner.py", line 307, in call_hook
    getattr(hook, fn_name)(self)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmcv/runner/hooks/evaluation.py", line 267, in after_train_epoch
    self._do_evaluate(runner)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmdet/core/evaluation/eval_hooks.py", line 123, in _do_evaluate
    key_score = self.evaluate(runner, results)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmcv/runner/hooks/evaluation.py", line 362, in evaluate
    results, logger=runner.logger, **self.eval_kwargs)
  File "/home/liuliu2023/EchoFusion/projects/mmdet3d_plugin/datasets/kradar_dataset.py", line 359, in evaluate
    eval_types=eval_types)
  File "/home/liuliu2023/EchoFusion/projects/mmdet3d_plugin/datasets/utils/evaluate_kradar.py", line 771, in kradar_eval
    eval_types)
  File "/home/liuliu2023/EchoFusion/projects/mmdet3d_plugin/datasets/utils/evaluate_kradar.py", line 661, in do_eval
    min_overlaps)
  File "/home/liuliu2023/EchoFusion/projects/mmdet3d_plugin/datasets/utils/evaluate_kradar.py", line 498, in eval_class
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
  File "/home/liuliu2023/EchoFusion/projects/mmdet3d_plugin/datasets/utils/evaluate_kradar.py", line 400, in calculate_iou_partly
    dt_boxes).astype(np.float64)
  File "/home/liuliu2023/EchoFusion/projects/mmdet3d_plugin/datasets/utils/evaluate_kradar.py", line 122, in bev_box_overlap
    from mmdet3d.core.evaluation.kitti_utils.rotate_iou import rotate_iou_gpu_eval
  File "/home/liuliu2023/EchoFusion/mmdetection3d/mmdet3d/core/evaluation/kitti_utils/rotate_iou.py", line 293, in <module>
    criterion=-1):
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/decorators.py", line 101, in kernel_jit
    kernel.bind()
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/compiler.py", line 548, in bind
    self._func.get()
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/compiler.py", line 426, in get
    ptx = self.ptx.get()
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/compiler.py", line 397, in get
    **self._extra_options)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/cudadrv/nvvm.py", line 496, in llvm_to_ptx
    ptx = cu.compile(**opts)
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/cudadrv/nvvm.py", line 233, in compile
    self._try_error(err, 'Failed to compile\n')
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/cudadrv/nvvm.py", line 251, in _try_error
    self.driver.check_error(err, "%s\n%s" % (msg, self.get_log()))
  File "/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/numba/cuda/cudadrv/nvvm.py", line 141, in check_error
    raise exc
numba.cuda.cudadrv.error.NvvmError: Failed to compile

<unnamed> (66, 23): parse expected comma after load's type
NVVM_ERROR_COMPILATION
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 1123433 closing signal SIGTERM
WARNING:torch.distributed.elastic.multiprocessing.api:Sending process 1123434 closing signal SIGTERM
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 1123432) of binary: /home/liuliu2023/anaconda3/envs/echofusion/bin/python
```
</details>

根据[issue](https://github.com/numba/numba/issues/6607)信息，`numba`最后支持的`CUDA`版本为`11.1.x`，因此需要降低`CUDA`的版本。

目前的环境信息
```bash
(echofusion) liuliu2023@user-NF5468M5:~/EchoFusion/mmdetection3d/mmdet3d/utils$ python collect_env.py 
sys.platform: linux
Python: 3.7.16 (default, Jan 17 2023, 22:20:44) [GCC 11.2.0]
CUDA available: True
GPU 0,1,2,3: Tesla V100-SXM2-32GB
CUDA_HOME: /home/liuliu2023/cuda-11.3
NVCC: Build cuda_11.3.r11.3/compiler.29745058_0
GCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
PyTorch: 1.10.0+cu113
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX512
  - CUDA Runtime 11.3
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
  - CuDNN 8.2
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 

TorchVision: 0.11.0+cu113
OpenCV: 4.9.0
MMCV: 1.4.0
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 11.3
MMDetection: 2.19.0
MMSegmentation: 0.20.0
MMDetection3D: 0.17.3+940e429
```

因此，需要重建一个基于`CUDA11.1`的环境。由于是在服务器上，所以首先要在服务器上安装一个新的`CUDA Compile`。

安装完毕后，安装基于CUDA11.1的新环境，重新运行代码

* 一个小bug
```python
TypeError: FormatCode() got an unexpected keyword argument 'verify'
```
[解决方法](https://github.com/open-mmlab/mmdetection/issues/10962)：
```
pip install yapf==0.40.1
```

之后将/home/liuliu2023/anaconda3/envs/echofusion/lib/python3.7/site-packages/mmcv/runner/epoch_based_runner.py中def train代码还原，发现在`train`的过程中有了报错，发现是mmdetection3d编译问题，但对其重新编译发现报错，无法重新编译。



