# Contents

- [Contents](#contents)
    - [PointPillars description](#pointpillars-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment requirements](#environment-requirements)
    - [Quick start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Dataset Preparation](#dataset-preparation)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)

## [PointPillars description](#contents)

PointPillars is a method for object detection in 3D that enables end-to-end learning with only 2D convolutional layers.
PointPillars uses a novel encoder that learn features on pillars (vertical columns) of the point cloud to predict 3D oriented boxes for objects.
There are several advantages of this approach.
First, by learning features instead of relying on fixed encoders, PointPillars can leverage the full information represented by the point cloud.
Further, by operating on pillars instead of voxels there is no need to tune the binning of the vertical direction by hand.
Finally, pillars are highly efficient because all key operations can be formulated as 2D convolutions which are extremely efficient to compute on a GPU.
An additional benefit of learning features is that PointPillars requires no hand-tuning to use different point cloud configurations.
For example, it can easily incorporate multiple lidar scans, or even radar point clouds.

> [Paper](https://arxiv.org/abs/1812.05784):  PointPillars: Fast Encoders for Object Detection from Point Clouds.
> Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom, 2018.

## [Model architecture](#contents)

The main components of the network are a Pillar Feature Network, Backbone, and SSD detection head.
The raw point cloud is converted to a stacked pillar tensor and pillar index tensor.
The encoder uses the stacked pillars to learn a set of features that can be scattered back to a 2D pseudo-image for a convolutional neural network.
The features from the backbone are used by the detection head to predict 3D bounding boxes for objects.
![network.png](.\src\data\img\network.png)
## [Dataset](#contents)

Dataset used: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

Data was collected with using a standard station wagon with two high-resolution color and grayscale video cameras.
Accurate ground truth is provided by a Velodyne laser scanner and a GPS localization system.
Dataset was captured by driving around the mid-size city of Karlsruhe, in rural areas and on highways.
Up to 15 cars and 30 pedestrians are visible per image. The 3D object detection benchmark consists of 7481 images.

## [Environment requirements](#contents)

- Hardware（Ascend）

    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)
- Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), data from [ImageSets](https://github.com/traveller59/second.pytorch/tree/master/second/data/ImageSets), put files from `ImageSets` into `pointpillars/src/data/ImageSets/`

  For the dataset preparation instructions see the [Dataset preparation](#dataset-preparation) section.

## [Quick start](#contents)

After preparing the dataset you can start training and evaluation as follows：

### [Running on Ascend](#contents)

#### Train

```shell
# standalone train
bash ./scripts/run_standalone_train.sh [CFG_PATH] [SAVE_PATH] [DEVICE_ID]

# distribute train
bash ./scripts/run_distribute_train.sh [CFG_PATH] [SAVE_PATH] [RANK_SIZE] [RANK_TABLE]
```

Example:

```shell
# standalone train
bash ./scripts/run_standalone_train.sh ./configs/car_xyres16.yaml ./experiments/car/ 0

# distribute train (8p)
bash ./scripts/run_distribute_train.sh ./configs/car_xyres16.yaml ./experiments/car/ 8 /home/hccl_8p_01234567_192.168.88.13.json
```

#### Evaluate

```shell
# evaluate
bash ./scripts/run_eval.sh [CFG_PATH] [CKPT_PATH] [DEVICE_ID]
```

Example:

```shell
# evaluate
bash ./scripts/run_eval.sh ./configs/car_xyres16.yaml ./experiments/car/poinpitllars-160_37120.ckpt 0
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└── pointpillars
    ├── configs
    │    ├── car_xyres16.yaml  # config for car detection
    │    └── ped_cycle_xyres16.yaml  # config for pedestrian and cyclist detection
    ├── scripts
    │    ├── run_eval.sh
    │    └── run_standalone_train.sh
    ├── src
    │    ├── builder
    │    │    ├── __init__.py                       # init file
    │    │    ├── anchor_generator_builder.py       # builder for anchor generator
    │    │    ├── box_coder_builder.py              # builder for box coder
    │    │    ├── dataset_builder.py                # builder for dataset
    │    │    ├── dbsampler_builder.py              # builder for db sampler
    │    │    ├── model_builder.py                  # builder for model
    │    │    ├── preprocess_builder.py             # builder for preprocess
    │    │    ├── similarity_calculator_builder.py # builder for similarity calculator
    │    │    ├── target_assigner_builder.py        # builder for target assigner
    │    │    └── voxel_builder.py                  # builder for voxel generator
    │    ├── core
    │    │    ├── point_cloud
    │    │    │    ├── __init__.py                  # init file
    │    │    │    ├── bev_ops.py                   # ops for bev
    │    │    │    └── point_cloud_ops.py           # ops for point clouds
    │    │    ├── __init__.py                       # init file
    │    │    ├── anchor_generator.py               # anchor generator
    │    │    ├── box_coders.py                     # box coders
    │    │    ├── box_np_ops.py                     # box ops with numpy
    │    │    ├── box_ops.py                        # box ops with mindspore
    │    │    ├── einsum.py                         # einstein sum
    │    │    ├── eval_utils.py                     # utils for evaluate
    │    │    ├── geometry.py                       # geometry
    │    │    ├── losses.py                         # losses
    │    │    ├── nms.py                            # nms
    │    │    ├── preprocess.py                     # preprocess operations
    │    │    ├── region_similarity.py              # region similarity calculator
    │    │    ├── sample_ops.py                     # ops for sample data
    │    │    ├── target_assigner.py                # target assigner
    │    │    └── voxel_generator.py                # voxel generator
    │    ├── data
    │    │    ├── ImageSets                         # splits for train and val
    │    │    ├── __init__.py                       # init file
    │    │    ├── dataset.py                        # kitti dataset
    │    │    └── kitti_common.py                   # auxiliary file for kitti
    │    ├── __init__.py                            # init file
    │    ├── create_data.py                         # create dataset for train model
    │    ├── pointpillars.py                        # pointpillars model
    │    ├── utils.py                               # utilities functions
    │    └── predict.py                             # postprocessing pointpillars`s output
    ├── __init__.py                                 # init file
    ├── eval.py                                     # evaluate mindspore model
    ├── README.md                                   # readme file
    ├── requirements.txt                            # requirements
    └── train.py                                    # train mindspore model
```

### [Script Parameters](#contents)

Training parameters can be configured in `car_xyres16.yaml` for car detection or `ped_cycle_xyres16.yaml` for pedestrian
and cyclist detection.

```text
"initial_learning_rate": 0.0002,        # learning rate
"max_num_epochs": 80,                  # number of training epochs
"weight_decay": 0.0001,                 # weight decay
"batch_size": 4,                        # batch size
"max_number_of_voxels": 16000,          # mux number of voxels in one pillar
"max_number_of_points_per_voxel": 100   # max number of points per voxel
```

For more parameters refer to the contents of `car_xyres16.yaml` or `ped_cycle_xyres16.yaml`.

### [Dataset Preparation](#contents)

1. Add `/path/to/pointpillars/` to your `PYTHONPATH`

```text
export PYTHONPATH=/path/to/pointpillars/:$PYTHONPATH
```

2. Download KITTI dataset into one folder:
- [Download left color images of object data set (12 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
- [Download camera calibration matrices of object data set (16 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- [Download training labels of object data set (5 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
- [Download Velodyne point clouds, if you want to use laser information (29 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
3. Unzip all downloaded archives.
4. Directory structure is as follows:

```text
└── KITTI
       ├── training
       │   ├── image_2 <-- for visualization
       │   ├── calib
       │   ├── label_2
       │   ├── velodyne
       │   └── velodyne_reduced <-- create this empty directory
       └── testing
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- create this empty directory
```

5. Download [ImageSets](https://github.com/traveller59/second.pytorch/tree/master/second/data/ImageSets), put files from `ImageSets` into `pointpillars/src/data/ImageSets/`

6. Create KITTI infos:

```shell
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

7. Create reduced point cloud:

```shell
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

8. Create groundtruth-database infos:

```shell
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

9. The config file `car_xyres16.yaml` or `ped_cycle_xyres16.yaml` needs to be edited to point to the above datasets:

```text
train_input_reader:
  ...
  database_sampler:
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
...
eval_input_reader:
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
```

### [Training Process](#contents)

#### [Run on Ascend](#contents)

##### Standalone training

```shell
bash ./scripts/run_standalone_train.sh ./configs/car_xyres16.yaml ./output/car/ 0
```

Logs will be saved to `./output/cars/log.txt`

Result:

```text
2024-03-13 23:12:26 epoch:6, iter:5600,  loss:1.1716965, fps:24.77 imgs/sec,  step time: 0.16150619983673095 s
2024-03-13 23:12:33 epoch:6, iter:5650,  loss:1.011474, fps:25.4 imgs/sec,  step time: 0.15746017456054687 s
2024-03-13 23:12:41 epoch:6, iter:5700,  loss:0.7482436, fps:25.05 imgs/sec,  step time: 0.15965424060821534 s
2024-03-13 23:12:50 epoch:6, iter:5750,  loss:0.58538055, fps:24.35 imgs/sec,  step time: 0.16429409503936768 s
2024-03-13 23:12:58 epoch:6, iter:5800,  loss:0.5734662, fps:23.72 imgs/sec,  step time: 0.16863516807556153 s
2024-03-13 23:13:06 epoch:6, iter:5850,  loss:0.77734077, fps:24.59 imgs/sec,  step time: 0.16267478942871094 s
2024-03-13 23:13:14 epoch:6, iter:5900,  loss:0.7195258, fps:24.65 imgs/sec,  step time: 0.16228347301483154 s
2024-03-13 23:13:22 epoch:6, iter:5950,  loss:0.6363396, fps:24.55 imgs/sec,  step time: 0.16291052341461182 s
2024-03-13 23:13:31 epoch:6, iter:6000,  loss:0.7434416, fps:24.52 imgs/sec,  step time: 0.1631227684020996 s
2024-03-13 23:13:39 epoch:6, iter:6050,  loss:0.7142066, fps:24.28 imgs/sec,  step time: 0.16477130889892577 s
2024-03-13 23:13:47 epoch:6, iter:6100,  loss:0.62609035, fps:24.31 imgs/sec,  step time: 0.16455113410949707 s
2024-03-13 23:13:55 epoch:6, iter:6150,  loss:0.77837485, fps:24.21 imgs/sec,  step time: 0.1652225112915039 s
2024-03-13 23:14:04 epoch:6, iter:6200,  loss:0.70987016, fps:24.15 imgs/sec,  step time: 0.16564173221588135 s
2024-03-13 23:14:12 epoch:6, iter:6250,  loss:0.889233, fps:23.58 imgs/sec,  step time: 0.16960060119628906 s
2024-03-13 23:14:20 epoch:6, iter:6300,  loss:0.62696505, fps:24.69 imgs/sec,  step time: 0.16198619365692138 s
2024-03-13 23:14:28 epoch:6, iter:6350,  loss:0.6138692, fps:24.32 imgs/sec,  step time: 0.16444008827209472 s
2024-03-13 23:14:37 epoch:6, iter:6400,  loss:0.8882336, fps:23.92 imgs/sec,  step time: 0.16720683097839356 s
2024-03-13 23:14:45 epoch:6, iter:6450,  loss:0.73737246, fps:24.47 imgs/sec,  step time: 0.16349081039428712 s

```

##### Distribute training (8p)

```shell
bash ./scripts/run_distribute_train.sh /home/group1/pointpillars/
configs/car_xyres16.yaml ./output/dist/ 8 /home/hccl_8p_01234567_192.168.88.13.json
```
the config path and the hccl file path need to use absolute path. 
Logs will be saved to `./device0/train_log0.txt`


### [Evaluation Process](#contents)


#### Ascend

```shell
bash ./scripts/run_eval.sh [CFG_PATH] [CKPT_PATH] [DEVICE_ID]
```

Example:

```shell
bash ./scripts/run_eval.sh ./configs/car_xyres16.yaml ./experiments/car/pointpillars-160_37120.ckpt 0
```

Result:

Here is model for cars detection as an example，you can view the result in log file `./experiments/car/log_eval.txt`：

```text
        Easy   Mod    Hard
Car AP@0.70, 0.70, 0.70:
bbox AP:93.63, 88.72, 87.29
```

Here is result for pedestrian and cyclist detection：

```text
        Easy   Mod    Hard
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:86.46, 67.37, 64.18
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:67.38, 62.54, 59.27
```
