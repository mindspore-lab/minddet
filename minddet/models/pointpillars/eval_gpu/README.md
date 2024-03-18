# kitti-object-eval-python
**Note**: This is borrowed from [traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)

Fast kitti object detection eval in python(finish eval in less than 10 second), support 2d/bev/3d/aos. , support coco-style AP. If you use command line interface, numba need some time to compile jit functions.
## Dependencies
Only support python 3.6+, need `numpy`, `skimage`, `numba`, `fire`. If you have Anaconda, just install `cudatoolkit` in anaconda. Otherwise, please reference to this [page](https://github.com/numba/numba#custom-python-environments) to set up llvm and cuda for numba.
* Install by conda:
```
conda install -c numba cudatoolkit=x.x  (8.0, 9.0, 9.1, depend on your environment)
```
## Usage
* commandline interface:
```
python evaluate.py evaluate --label_path=/path/to/out_gt_annos.pkl --result_path=/path/to/out_dt_annos.pkl --current_class=0
```
current_class=0 car

current_class=[1,2] pedestrian and cyclist
