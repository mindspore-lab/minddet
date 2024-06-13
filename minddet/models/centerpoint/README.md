# 内容

* [Centerpoint描述](#描述)
* [网络结构](#网络结构)
* [环境依赖](#环境依赖)
* [数据集](#数据集)
* [快速开始](#快速开始)



## [Centerpoint描述](#contents)

三维物体通常在点云中以3D盒子的形式表示。这种表示模仿了已经研究充分的基于图像的2D边界框检测，但带来了额外的挑战。3D世界中的物体并不遵循任何特定的方向，基于盒子的检测器在列举所有方向或适应旋转物体的轴对齐边界框方面存在困难。在本文中，我们提出了一种不同的方法，即以点的形式来表示、检测和跟踪3D物体。我们的框架，CenterPoint，首先使用关键点检测器检测物体的中心，并回归到其他属性，包括3D尺寸、3D方向和速度。在第二阶段，它使用物体上的额外点特征来细化这些估计。在CenterPoint中，3D物体跟踪简化为贪婪最近点匹配。所得到的检测和跟踪算法简单、高效且有效。CenterPoint在nuScenes基准测试中实现了最先进的性能，对于单一模型，3D检测和跟踪的NDS为65.5，AMOTA为63.8。在Waymo开放数据集上，CenterPoint在所有单一模型方法中表现最佳，并且所有仅使用激光雷达的提交中排名第一。



## [网络结构](#contents)

以PointPillar作为网络的backbone。



## [环境依赖](#contents)

* 硬件（Ascend）

  * 910系列昇腾卡

* 软件

  * Mindspore 2.3
  * Python 3.7

* 安装依赖库

  * ```shell
    pip install -r requirements.txt
    export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
    ```



## [数据集](#contents)

此网络使用nuScenes数据集。

### 准备数据

下载数据集，并将数据集整理成如下目录结构：

```shell
# For nuScenes Dataset
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

创建指向数据集根目录的软链接：

```shell
mkdir data && cd data
ln -s DATA_ROOT
mv DATA_ROOT nuScenes # rename to nuScenes
```

### 制作数据

代码指令：

 ```shell
 # nuScenes
 python tools_ms/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
 ```

数据制作完成后，整体代码路径格式应为：

```shell
# For nuScenes Dataset
└── CenterPoint
       └── data
              └── nuScenes
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database
```

### 准备mindrecord数据
修改配置文件中参数
```python
# 训练集路径
train_mindrecord_dir = "path/to/train_mindrecord"
# 测试集路径
test_mindrecord_dir = "path/to/test_mindrecord"
```

## [快速开始](#contents)

### 训练

#### 单卡训练

```shell
export DEVICE_ID=X
python tools_ms/train.py --train_url work_dirs/SAVE_CKPT_DIR
```

#### 多卡训练

```shell
cd tools_ms
bash dist_train.sh SAVE_CKPT_DIR # SAVE_CKPT_DIR为绝对路径
```



### 评估

```shell
export DEVICE_ID=X
python -m tools_ms.eval --checkpoint CKPT_ABSOLUTE_PATH
```
