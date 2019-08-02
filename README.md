# Contextual Affinity Understanding for Low Shot Object Detection

By anonymous authors.

### Introduction
Inspired by the structure of Receptive Fields (RFs) in human visual systems, we propose a novel RF Block (RFB) module, which takes the relationship between the size and eccentricity of RFs into account, to enhance the discriminability and robustness of features. We further  assemble the RFB module to the top of SSD with a lightweight CNN model, constructing the RFB Net detector. You can use the code to train/evaluate the RFB Net for object detection. For more details, please refer to our [ECCV paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Songtao_Liu_Receptive_Field_Block_ECCV_2018_paper.pdf). 

<img align="right" src="https://github.com/Ze-Yang/CAUNet/blob/master/doc/CAU.png">

&nbsp;
&nbsp;

### COCO60 to VOC20 results (Transfer Setting)
| Method |  *1shot* | *5shot* |
|:-------|:-----:|:-------:|
| [Prototype](https://github.com/ShaoqingRen/faster_rcnn) | 22.8 | 39.8 |
| [Imprinted](http://pjreddie.com/darknet/yolo/) | 24.5 | 40.9 |
| [Non-local](https://github.com/daijifeng001/R-FCN)| 25.2 | 41.0 |
| our CAU-Net | **27.0** | **43.8** |


### VOC15 to VOC20 (Incremental Setting)
| System |  *test-dev mAP* | **Time** (Titan X Maxwell) |
|:-------|:-----:|:-------:|
| [Faster R-CNN++ (ResNet-101)](https://github.com/KaimingHe/deep-residual-networks) | 34.9 | 3.36s | 
| [YOLOv2 (Darknet-19)](http://pjreddie.com/darknet/yolo/) | 21.6 | 25ms| 
| [SSD300* (VGG16)](https://github.com/weiliu89/caffe/tree/ssd) | 25.1 | 22ms |
| [SSD512* (VGG16)](https://github.com/weiliu89/caffe/tree/ssd) | 28.8 | 53ms |
| [RetinaNet500 (ResNet-101-FPN)](https://arxiv.org/pdf/1708.02002.pdf) | 34.4| 90ms|
| RFBNet300 (VGG16) | **30.3** |**15ms** | 
| RFBNet512 (VGG16) | **33.8** | **30ms** |
| RFBNet512-E (VGG16) | **34.4** | **33ms** |  


### MobileNet
|System |COCO *minival mAP*| **\#parameters**|
|:-------|:-----:|:-------:|
|[SSD MobileNet](https://arxiv.org/abs/1704.04861)| 19.3| 6.8M|
|RFB MobileNet| 20.7 | 7.4M|


### Citing RFB Net
Please cite our paper in your publications if it helps your research:

    @InProceedings{Liu_2018_ECCV,
    author = {Liu, Songtao and Huang, Di and Wang, andYunhong},
    title = {Receptive Field Block Net for Accurate and Fast Object Detection},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    month = {September},
    year = {2018}
    }

### Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Models](#models)

## Installation
- Install [Anaconda3-5.2.0](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh).
- Clone this repository. This repository is mainly based on [RFBNet](https://github.com/ruinmessi/RFBNet), many thanks to them.
  * Note: We currently only support PyTorch-1.0.1 and Python 3+.
- Run this command to build up the environment.
``` 
conda env create -n CAU -f env.yaml
```

- Compile the nms and coco tools:

```Shell
sh make.sh
```

*Note*: Check you GPU architecture support in utils/build.py, line 131. Default is:
``` 
'nvcc': ['-arch=sm_61',
``` 

## Datasets
### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:    https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
or from [BaiduYun Driver](https://pan.baidu.com/s/1jIP86jW) 

- Download the [RFBNet models](https://pan.baidu.com/s/1aW73KRm3anrX0ulcadQZMg), which are pretrained on COCO60, VOC15_split1, VOC15_split2, VOC15_split3 dataset.

By default, we assume you have downloaded the files in the `CAUNet/weights` dir.

### Transfer Setting (COCO60 to VOC20)
- To train CAUNet under transfer setting, use the `train_RFB_CAU.py` script (or `train_SSD_CAU.py` if you want to use SSD framework), simply specify the parameters listed in `train_RFB_CAU.py` as a flag or manually change them.
```Shell
python train_RFB_CAU.py --n_shot_task 5 --resume_net weights/RFB_COCO60_pretrain.pth --save_folder weights/VOC_CAU_5shot/
```

### Incremental Setting (VOC15 to VOC20)
- To train CAUNet under incremental setting, use the `train_RFB_CAU_incre.py` script, simply specify the parameters listed in `train_RFB_CAU_incre.py` as a flag or manually change them.
```Shell
python train_RFB_CAU_incre.py --n_shot_task 5 --resume_net weights/split1_pretrain.pth --save_folder weights/VOC_split1_5shot/
```

Note:
  * --n_shot_task: specify n shot task (n=1, 2, 3, 5 ,10)
  * If you want to reproduce the results in the paper, feel free to reset the HEAD to corresponding commit, e.g., reset the HEAD to `CAU_5shot` for 5 shot task, all the training settings will be ready for you.

## Evaluation
- To evaluate a trained network under transfer setting:
```Shell
python test_RFB.py --method CAU -m weights/VOC_CAU_5shot/Final_RFB_vgg_VOC_CAU.pth --save_folder eval/VOC_CAU_5shot/40ep/
```

- To evaluate a trained network under incremental setting:
```Shell
python test_RFB.py --method CAU_incre -m weights/VOC_split1_5shot/RFB_vgg_VOC_CAU_incre_epoches_4.pth --save_folder eval/VOC_split1_5shot/4ep/
```
By default, it will directly output the mAP results on VOC2007 *test*.

