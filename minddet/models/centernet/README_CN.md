# 内容

- [CenterNet 描述](#描述)
- [网络结构](#网络结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本](#脚本)
    - [脚本参数](#脚本参数)
    - [训练](#训练)
    - [推理](#推理)
- [实验结果](#实验结果)
    - [表现](#表现)
        - [昇腾910上的训练结果](#昇腾910上的训练结果)
        - [昇腾910上的推理结果](#昇腾910上的推理结果)

# [CenterNet 描述](#contents)

CenterNet 是一种新颖实用的无锚点方法，用于对象检测、3D 检测和姿态估计，可检测将对象识别为图像中轴对齐的框。检测器使用关键点估计来查找中心点，并回归到所有其他对象属性，例如大小、3D 位置、方向甚至姿势。在本质上，它是一种单阶段方法，相较于其他同类网络，CenterNet可以同时预测中心点和边框，并有着出更快更准确的表现

[Paper](https://arxiv.org/pdf/1904.07850.pdf): Objects as Points. 2019.
Xingyi Zhou(UT Austin) and Dequan Wang(UC Berkeley) and Philipp Krahenbuhl(UT Austin)

# [Model 网络结构](#contents)

使用带有DCN layer的resnet18作为网络的backbone

# [数据集](#contents)

注意，你可以基于原始论文中提到的数据集或在相关域/网络架构中广泛使用的数据集运行脚本。在以下各节中，我们将介绍如何使用下面的相关数据集运行脚本。

使用数据集: [COCO2017](https://cocodataset.org/)

- 数据集大小：26G
    - 训练集：19G，118000 images  
    - 评估集：0.8G，5000 images
    - 测试集: 6.3G, 40000 images
    - 标注数据：808M，instances，captions etc
- 数据格式：image and json files

- 注意：所有数据的处理都放在了dataset.py中

- 目录结构如下，目录和文件的名称由用户定义：

    ```path
    .
    ├── dataset
        ├── centernet
            ├── annotations
            │   ├─ train.json
            │   └─ val.json
            └─ images
                ├─ train
                │    └─images
                │       ├─class1_image_folder
                │       ├─ ...
                │       └─classn_image_folder
                └─ val
                │    └─images
                │       ├─class1_image_folder
                │       ├─ ...
                │       └─classn_image_folder
                └─ test
                      └─images
                        ├─class1_image_folder
                        ├─ ...
                        └─classn_image_folder
    ```

# [环境要求](#contents)

- 硬件（Ascend）
    - 910系列昇腾卡
- 框架
    MindSpore 2.3.0
    - [MindSpore](https://www.mindspore.cn/install/en)
- 更多关于mindspore的教程可以参考下面文档
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/zh-CN/r2.2/index.html)
- 下载 COCO2017.
- 我们使用COCO2017作为实验数据集.

    1. 如果使用 coco 数据集。运行脚本时选择要coco的数据集。
        安装 Cython and pycocotool

        ```pip
        pip install Cython

        pip install pycocotools
        ```

        并更改`default_config.yaml`中需要的COCO_ROOT和其他设置. 目录结构如下:

        ```path
        ├── data    
            └─coco
                ├─annotations
                    ├─instance_train2017.json
                    └─instance_val2017.json
                ├─val2017
                └─train2017

        ```

    2. 如果使用自己的数据集。运行脚本时选择数据集到其他数据集。
        数据集信息的组织格式与COCO相同。

# [快速开始](#contents)

- 本地运行

    通过官网安装好MindSpore之后，可以按照下列步骤开始运行

    注意:
    1.MINDRECORD_DATASET_PATH 是mindrecord 数据目录.
    2.对于 `train.py`, LOAD_CHECKPOINT_PATH 是预训练好的或者需要resume的CenterNet模型ckpt，一般设置为" "；如果想要基于预训练好的backbone进行训练，需要设置'default_config.yaml'中的'load_backbone_path'.
    3.对于 `eval.py`, LOAD_CHECKPOINT_PATH 是需要进行评估的ckpt.

    ```shell
    cd minddet/models/centernet
    # 将训练数据集转换为mind record格式
    bash scripts/convert_dataset_to_mindrecord.sh [COCO_DATASET_DIR] [MINDRECORD_DATASET_DIR]

    # 单卡训练
    bash scripts/run_standalone_train_ascend.sh [DEVICE_ID] [MINDRECORD_DATASET_PATH] [LOAD_CHECKPOINT_PATH](optional)

    # 分布式训练
    bash scripts/run_distributed_train_ascend.sh [MINDRECORD_DATASET_PATH] [RANK_TABLE_FILE] [LOAD_CHECKPOINT_PATH](optional)

    # 单卡推理
    bash tools/eval.py --config_path=./default_config.yaml
    ```

# [脚本](#contents)

```path
.
├── cv
    ├── centernet_resnet50
        ├── train.py                     // training scripts
        ├── eval.py                      // testing and evaluation outputs
        ├── export.py                    // convert mindspore model to mindir model
        ├── README.md                    // descriptions about centernet_resnet50
        ├── default_config.yaml          // parameter configuration
        ├── preprocess.py                // preprocess scripts
        ├── postprocess.py               // postprocess scripts
        ├── scripts
        │   ├── ascend_distributed_launcher
        │   │    ├── __init__.py
        │   │    ├── hyper_parameter_config.ini         // hyper parameter for distributed training
        │   │    ├── get_distribute_train_cmd.py        // script for distributed training
        │   │    ├── README.md
        │   ├── convert_dataset_to_mindrecord.sh        // shell script for converting coco type dataset to mindrecord
        │   ├── run_standalone_train_ascend.sh          // shell script for standalone training on ascend
        │   ├── run_distributed_train_ascend.sh         // shell script for distributed training on ascend
        │   ├── run_standalone_eval_ascend.sh           // shell script for standalone evaluation on ascend
        └── src
            ├── model_utils
            │   ├── config.py            // parsing parameter configuration file of "*.yaml"
            │   ├── device_adapter.py    // local or ModelArts training
            │   ├── local_adapter.py     // get related environment variables on local
            │   └── moxing_adapter.py    // get related environment variables abd transfer data on ModelArts
            ├── __init__.py
            ├── centernet_det.py          // centernet networks, training entry
            ├── dataset.py                // generate dataloader and data processing entry
            ├── decode.py                 // decode the head features
            ├── resnet50.py              // resnet50 backbone
            ├── image.py                  // image preprocess functions
            ├── post_process.py           // post-process functions after decode in inference
            ├── utils.py                  // auxiliary functions for train, to log and preload
            └── visual.py                 // visualization image, bbox, score and keypoints
```

## [脚本参数](#contents)

### 转换为mind record

```text
usage: dataset.py  [--coco_data_dir COCO_DATA_DIR]
                   [--mindrecord_dir MINDRECORD_DIR]
                   [--mindrecord_prefix MINDRECORD_PREFIX]

options:
    --coco_data_dir            path to coco dataset directory: PATH, default is ""
    --mindrecord_dir           path to mindrecord dataset directory: PATH, default is ""
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_det.train.mind"
```

### train

```text
usage: train.py  [--device_target DEVICE_TARGET] [--distribute DISTRIBUTE]
                 [--need_profiler NEED_PROFILER] [--profiler_path PROFILER_PATH]
                 [--epoch_size EPOCH_SIZE] [--train_steps TRAIN_STEPS]  [device_id DEVICE_ID]
                 [--device_num DEVICE_NUM] [--do_shuffle DO_SHUFFLE]
                 [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                 [--enable_save_ckpt ENABLE_SAVE_CKPT]
                 [--save_checkpoint_path SAVE_CHECKPOINT_PATH]
                 [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                 [--save_checkpoint_steps N] [--save_checkpoint_num N]
                 [--mindrecord_dir MINDRECORD_DIR]
                 [--mindrecord_prefix MINDRECORD_PREFIX]
                 [--save_result_dir SAVE_RESULT_DIR]

options:
    --device_target            device where the code will be implemented: "Ascend"
    --distribute               training by several devices: "true"(training by more than 1 device) | "false", default is "true"
    --need profiler            whether to use the profiling tools: "true" | "false", default is "false"
    --profiler_path            path to save the profiling results: PATH, default is ""
    --epoch_size               epoch size: N, default is 1
    --train_steps              training Steps: N, default is -1
    --device_id                device id: N, default is 0
    --device_num               number of used devices: N, default is 1
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_lossscale         enable lossscale: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --enable_save_ckpt         enable save checkpoint: "true" | "false", default is "true"
    --save_checkpoint_path     path to save checkpoint files: PATH, default is ""
    --load_checkpoint_path     path to load checkpoint files: PATH, default is ""
    --save_checkpoint_steps    steps for saving checkpoint files: N, default is 1000
    --save_checkpoint_num      number for saving checkpoint files: N, default is 1
    --mindrecord_dir           path to mindrecord dataset directory: PATH, default is ""
    --mindrecord_prefix        prefix of MindRecord dataset filename: STR, default is "coco_det.train.mind"
    --save_result_dir          path to save the visualization results: PATH, default is ""
```

### eval

```text
usage: eval.py  [--device_target DEVICE_TARGET] [--device_id N]
                [--load_checkpoint_path LOAD_CHECKPOINT_PATH]
                [--data_dir DATA_DIR] [--run_mode RUN_MODE]
                [--visual_image VISUAL_IMAGE]
                [--enable_eval ENABLE_EVAL] [--save_result_dir SAVE_RESULT_DIR]
options:
    --device_target              device where the code will be implemented: "Ascend"
    --device_id                  device id to run task, default is 0
    --load_checkpoint_path       initial checkpoint (usually from a pre-trained CenterNet model): PATH, default is ""
    --data_dir                   validation or test dataset dir: PATH, default is ""
    --run_mode                   inference mode: "val" | "test", default is "val"
    --visual_image               whether visualize the image and annotation info: "true" | "false", default is "false"
    --save_result_dir            path to save the visualization and inference results: PATH, default is ""
```

### 参数悬梁

Parameters for training and evaluation can be set in file `default_config.yaml`.

#### 选项

```text
train_config.
    batch_size: 16                  // batch size of input dataset: N, default is 16 for 8 NPU, 114 for 1 NPU
    loss_scale_value: 1024          // initial value of loss scale: N, default is 1024
    optimizer: 'Adam'               // optimizer used in the network: Adam, default is Adam
    lr_schedule: 'MultiDecay'       // schedules to get the learning rate
```

```text
config for evaluation.
    SOFT_NMS: True                  // nms after decode: True | False, default is True
    keep_res: True                  // keep original or fix resolution: True | False, default is True
    multi_scales: [1.0]             // use multi-scales of image: List, default is [1.0]
    K: 100                          // number of bboxes to be computed by TopK, default is 100
    score_thresh: 0.3               // threshold of score when visualize image and annotation info,default is 0.3
```

#### 参数

```text
Parameters for dataset (Training/Evaluation):
    num_classes                     number of categories: N, default is 80
    max_objs                        maximum numbers of objects labeled in each image,default is 128
    input_res_train                 train input resolution, default is [512, 512]
    output_res                      output resolution, default is [128, 128]
    input_res_test                  test input resolution, default is [680, 680]
    rand_crop                       whether crop image in random during data augmenation: True | False, default is True
    shift                           maximum value of image shift during data augmenation: N, default is 0.1
    scale                           maximum value of image scale times during data augmenation: N, default is 0.4
    aug_rot                         properbility of image rotation during data augmenation: N, default is 0.0
    rotate                          maximum value of rotation angle during data augmentation: N, default is 0.0
    flip_prop                       properbility of image flip during data augmenation: N, default is 0.5
    color_aug                       color augmentation of RGB image, default is True
    coco_classes                    name of categories in COCO2017
    mean                            mean value of RGB image
    std                             variance of RGB image
    eig_vec                         eigenvectors of RGB image
    eig_val                         eigenvalues of RGB image

Parameters for network (Training/Evaluation):
    num_stacks         　　　　　　　 the number of stacked resnet network,default is 1
    down_ratio                      the ratio of input and output resolution during training, default is 4
    head_conv                       the input dimension of resnet network,default is 64
    block_class                     block for network,default is [3, 4, 23, 3]
    dense_hp                        whether apply weighted pose regression near center point: True | False,default is True
    dense_wh                        apply weighted regression near center or just apply regression on center point
    cat_spec_wh                     category specific bounding box size
    reg_offset                      regress local offset or not: True | False,default is True
    hm_weight                       loss weight for keypoint heatmaps: N, default is 1.0
    off_weight                      loss weight for keypoint local offsets: N,default is 1
    wh_weight                       loss weight for bounding box size: N, default is 0.1
    mse_loss                        use mse loss or focal loss to train keypoint heatmaps: True | False,default is False
    reg_loss                        l1 or smooth l1 for regression loss: 'l1' | 'sl1', default is 'l1'

Parameters for optimizer and learning rate:
    Adam:
    weight_decay                    weight decay: Q
    decay_filer                     lamda expression to specify which param will be decayed

    PolyDecay:
    learning_rate                   initial value of learning rate: Q
    end_learning_rate               final value of learning rate: Q
    power                           learning rate decay factor
    eps                             normalization parameter
    warmup_steps                    number of warmup_steps

    MultiDecay:
    learning_rate                   initial value of learning rate: Q
    eps                             normalization parameter
    warmup_steps                    number of warmup_steps
    multi_epochs                    list of epoch numbers after which the lr will be decayed
    factor                          learning rate decay factor
```

## [训练](#contents)

在第一次训练之前，需要将 coco 类型的数据集转换为 mindrecord 文件以提高性能。

```shell
bash scripts/convert_dataset_to_mindrecord.sh /path/coco_dataset_dir /path/mindrecord_dataset_dir
```

上面的命令将在后台运行，转换后的mindrecord文件将位于你指定的路径中。

### 分布式训练

```shell
bash scripts/run_distributed_train_ascend.sh /path/mindrecord_dataset /path/hccl.json /path/load_ckpt(optional)
```

上面的命令将在后台运行，您可以在 LOG*/training_log.txt 和 LOG*/ms_log/ 中查看训练日志。训练完成后，默认情况下，你将在 LOG*/ckpt_0 文件夹下获得ckpt。loss值将显示如下
```text
# grep "epoch" LOG0/training_log.txt
epoch: 139 | current epoch percent: 0.988 | step: 128229 | loss 2.2666364 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.989 | step: 128230 | loss 2.1323793 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.990 | step: 128231 | loss 1.5310838 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.991 | step: 128232 | loss 1.5261183 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.992 | step: 128233 | loss 1.8693255 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.993 | step: 128234 | loss 2.2504547 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.995 | step: 128235 | loss 1.9662838 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.996 | step: 128236 | loss 2.482753 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.997 | step: 128237 | loss 1.7604492 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.998 | step: 128238 | loss 1.8683317 | overflow False | scaling_sens 1024.0 | lr 5e-06
epoch: 139 | current epoch percent: 0.999 | step: 128239 | loss 1.4165485 | overflow False | scaling_sens 1024.0 | lr 5e-06
```

如果想基于官方提供的预训练好的resnet18进行训练, 下载ckpt [models](https://download.pytorch.org/models/resnet18-5c106cde.pth), 然后执行下列命令进行权重转换：
```python
python tools/convert_resnet18.py -ckpt_file ckpt_file --torch_name_file torch_name --ms_name_file ms_name --output_file output_file
```
设置'default_config.yaml'中的'load_backbone_path'，执行训练脚本

## [推理](#contents)

### 评估
CenterNet使用NMS进行后处理，需要先安装nms
```shell
    git clone https://github.com/xingyizhou/CenterNet.git
    cd CenterNet/src/lib/external/
    make
    python setup.py install
    cd - || exit
    rm -rf CenterNet
```
```shell
# Evaluation base on validation dataset
# On Ascend
python tools/eval.py --config_path=./default_config.yaml
```

可以看到评估结果如下：

```log
overall performance on coco2017 validation dataset
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.287
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.293
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.260
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.420
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.653
```

## [实验结果](#contents)

### 昇腾910上的训练结果

CenterNet on 11.8K images(The annotation and data format must be the same as coco)

| Parameters             | CenterNet_ResNet18_DCN                                           |
| ---------------------- | ------------------------------------------------------------ |
| Resource               | Ascend 910; CPU 2.60GHz, 192cores; Memory, 1511G              |
| uploaded Date          | 03/08/2024 (month/day/year)                                  |
| MindSpore Version      | 2.3.0                                                        |
| Dataset                | COCO2017/train2017                                                     |
| Training Parameters    | 8p, epoch=140, steps=128240, batch_size = 16, lr=5e-4      |
| Optimizer              | Adam                                                         |
| Loss Function          | Focal Loss, L1 Loss, RegLoss                                 |
| outputs                | detections                                                   |
| Loss                   | 1.5-2.0                                                      |
| Speed                  | 8p 590 img/s                                                  |
| Total time: training   | 8p: 8 h                                                     |
| Checkpoint             | 166MB (.ckpt file)                                           |
| Scripts                | run_distributed_train_ascend.sh                                          |

### I昇腾910上的推理结果

CenterNet on validation(5K images)

| Parameters           | CenterNet_ResNet18_DCN                                           |
| -------------------- | ------------------------------------------------------------ |
| Resource             | Ascend 910; CPU 2.60GHz, 192cores; Memory, 1511G             |
| uploaded Date        | 03/08/2024 (month/day/year)                                  |
| MindSpore Version    | 2.3.0                                                        |
| Dataset              | COCO2017/val2017                                                     |
| batch_size           | 1                                                            |
| outputs              | mAP                                                          |
| Accuracy(validation) | MAP: 28.7%, AP50: 47.2%, AP75: 29.3%, Medium: 31.5%, Large: 42.6% |

# [随机情况的说明](#contents)

In run_distributed_train_ascend.sh, we set do_shuffle to True to shuffle the dataset by default.
In train.py, we set a random seed to make sure that each node has the same initial weight in distribute training.

# FAQ

