# Setting Up Datasets
Note that all datasets are downloaded on the TDT4265 compute server and the tulipan/cybele computers (saved under /work/datasets).

Index
- Datasets used for assignment 4:
    - [PASCAL VOC](#pascal-voc)
    - MNIST dataset will be downloaded automatic. For information, see: [hukkelas/MNIST-ObjectDetection](github.com/hukkelas/MNIST-ObjectDetection)

---


## Pascal VOC
For Pascal VOC dataset, make the folder structure like this:
```
VOCdevkit
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```

You can download this if you are on the school network directly as a .zip file (Note that this dataset should be only used for educational/academic purposes).

With scp: 
```
scp [ntnu-username]@oppdal.idi.ntnu.no:/work/datasets/VOC.zip .

unzip VOC.zip
```


Or you can download it from the PASCAL VOC website:
http://host.robots.ox.ac.uk/pascal/VOC/

Note that we are using the VOC2007 TRAIN/VAL + VOC2012 TRAIN/VAL as the train set.

We use VOC2007 Test as the validation set.


## TDT4265 Dataset
To take a peek at the dataset, take a look at [visualize_dataset.ipynb](../notebooks/visualize_dataset.ipynb).

#### Getting started
**Download/Setup:**
```
python3 scripts/update_tdt4265_dataset.py
```
This will automatically download the dataset on local computers and create symlinks to the folder data if you are on TDT4265 compute servers.

#### Dataset Information
Label format follows the standard COCO format (see [here](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) for more info).
