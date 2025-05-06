DETR Implementation in Pytorch
========

This repository implements DETR, with training, inference and mAP evaluation in PyTorch.
The repo provides code to train on voc dataset. Specifically I trained on trainval images of VOC 2007+2012 dataset and for testing, I use VOC2007 test set.
For easy and quicker training/testing, I use fixed-size images of 640x640(unlike official repo which uses multi-scale training and tests on 800 sized images).

This repo was only meant for better understanding of DETR. Though the overall flow and some of the code(like matching) is exactly like the official repo(with some refactor) but still, for getting best results please use [official implementation](https://github.com/facebookresearch/detr) itself.

## DETR Explanation and Implementation Video
<a href="https://www.youtube.com/watch?v=v900ZFKkWxA">
   <img alt="DETR Explanation Video" src="https://github.com/user-attachments/assets/3e47bf7d-0e7b-46d7-9d1b-945eaf4d76de" width="400">
</a>
<a href="https://www.youtube.com/watch?v=NG09OJQPWWQ">
   <img alt="DETR Implementation Video" src="https://github.com/user-attachments/assets/99bea538-6062-44dd-8752-982839a5f893" width="400">
</a>



## Result by training DETR on VOC 2007 dataset 
I have used frozen Resnet 34 backbone with 25 query objects. Using this configuration, one should be able to get **65% mAP(with NMS) and 60%(without)** by training on VOC 2007+2012 trainval images.
To get better results, use trainable Resnet-50/101 with 100 query objects.

<img src="https://github.com/user-attachments/assets/9f2a0e2d-8b82-45b3-8566-fcc2f9313bbd" width="250">
<img src="https://github.com/user-attachments/assets/762ec1f2-bd50-43ee-ab88-6cd439425f96" width="250">
</br>
<img src="https://github.com/user-attachments/assets/07a7ffa1-7ae3-422d-ae4d-7f5aa6c59151" width="250">
<img src="https://github.com/user-attachments/assets/6c0aa6c4-ffe4-4661-bbc0-76de2aacc511" width="250">

Here's an evaluation result(AP50 with NMS) that I got after training ~250 epochs.
```
Class Wise Average Precisions
AP for class background = nan
AP for class aeroplane = 0.6873
AP for class bicycle = 0.7786
AP for class bird = 0.6163
AP for class boat = 0.5471
AP for class bottle = 0.2270
AP for class bus = 0.7405
AP for class car = 0.7188
AP for class cat = 0.8861
AP for class chair = 0.3762
AP for class cow = 0.6069
AP for class diningtable = 0.7945
AP for class dog = 0.8495
AP for class horse = 0.8521
AP for class motorbike = 0.7301
AP for class person = 0.5499
AP for class pottedplant = 0.2318
AP for class sheep = 0.5267
AP for class sofa = 0.9662
AP for class train = 0.8352
AP for class tvmonitor = 0.5966
Mean Average Precision : 0.6559
```


## Data preparation
For setting up the VOC 2007 dataset:
* Create a data directory inside DETR-PyTorch
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the `VOC2007` directory inside `data` directory
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2007` directory and name it as `VOC2007-test` directory inside `data`
* For using 2012 trainval images as well, download VOC 2012 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2012` directory inside `data`
  * Ensure to place all the directories inside the data folder of repo according to below structure
      ```
      DETR-Pytorch
          -> data
              -> VOC2007
                  -> JPEGImages
                  -> Annotations
                  -> ImageSets
              -> VOC2007-test
                  -> JPEGImages
                  -> Annotations
              -> VOC2012 
                  -> JPEGImages
                  -> Annotations
                  -> ImageSets
          -> tools
              -> train.py
              -> infer.py
          -> config
              -> voc.yaml
          -> model
              -> detr.py 
          -> dataset
              -> voc.py
      ```

## For training on your own dataset

* Update the path for `train_im_sets`, `test_im_sets` in config
* If you want to train on 2007+2012 trainval then have `train_im_sets` as `['data/VOC2007', 'data/VOC2012'] `
* Modify dataset file `dataset/voc.py` to load images and annotations accordingly specifically `load_images_and_anns` method
* Update the class list of your dataset in the dataset file.
* Dataset class should return the following:
    ```
  im_tensor(C x H x W) , 
  target{
        'boxes': Number of Gts x 4 (this is in x1y1x2y2 format normalized from 0-1)
        'labels': Number of Gts,
        'difficult': Number of Gts,
        }
  file_path
  ```


## For modifications 
* In case you have GPU which does not support large batch size, you can use a smaller batch size like 2 and then have `acc_steps` in config set as 4(to mimic 8 batch size training).
* For using a different backbone you would have to change the following:
  * Change the backbone layers in initialization of DETR model
  * Ensure the `backbone_channels` is correctly set in config, this is the number of channels in final feature map returned by backbone 

# Quickstart
* Create a new conda environment with python 3.10 then run below commands
* ```git clone https://github.com/explainingai-code/DETR-PyTorch.git```
* ```cd DETR-PyTorch```
* ```pip install -r requirements.txt```
* For training/inference use the below commands passing the desired configuration file as the config argument in case you want to play with it. 
* ```python -m tools.train``` for training DETR on VOC dataset
* ```python -m tools.infer --evaluate False --infer_samples True``` for generating inference predictions
* ```python -m tools.infer --evaluate True --infer_samples False``` for evaluating on test dataset

## Configuration
* ```config/voc.yaml``` - Allows you to play with different components of DETR on voc dataset  


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of `task_name` key in config will be created

During training of DETR, the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During inference the following output will be saved
* Sample prediction outputs for images in ```task_name/samples```

## Citations
```
@misc{carion2020endtoendobjectdetectiontransformers,
      title={End-to-End Object Detection with Transformers}, 
      author={Nicolas Carion and Francisco Massa and Gabriel Synnaeve and Nicolas Usunier and Alexander Kirillov and Sergey Zagoruyko},
      year={2020},
      eprint={2005.12872},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2005.12872}, 
}
```
