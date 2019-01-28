Keypoint Detections of Cars and Humans
======================


## Quick start guide
Follow these steps to set up a simple example:

### 1. Check out the codebase
```
git clone https://github.com/CMU-ILIM/Combined-Keypoints
cd Combined-Keypoints
```

### 2. Install Required Libraries
Install required python libraries
```
virtualenv Combined-Keypoints -p python3
source Combined-Keypoints/bin/activate
pip3 install -r requirements.txt

```

Compile required detector libraries
```
cd Detector/lib
sh make.sh

```

### 3. To run the detectors on Images or Videos
 
## Pretrained Model

I use ImageNet pretrained weights from Caffe for the backbone networks.

- [ResNet50](https://drive.google.com/open?id=1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1), [ResNet101](https://drive.google.com/open?id=1x2fTMqLrn63EMW0VuK4GEa2eQKzvJ_7l), [ResNet152](https://drive.google.com/open?id=1NSCycOb7pU0KzluH326zmyMFUU55JslF)

Download them and put them into the `{repo_root}/data/pretrained_model`.

You can the following command to download them all:

- extra required packages: `argparse_color_formater`, `colorama`, `requests`

```
python tools/download_imagenet_weights.py
```

**NOTE**: Caffe pretrained weights have slightly better performance than Pytorch pretrained. Suggest to use Caffe pretrained models from the above link to reproduce the results. By the way, Detectron also use pretrained weights from Caffe.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data preprocessing (minus mean and normalize) as used in Pytorch pretrained model.**

Download the trained Keypoint models for cars and persons and place them in the home folder

- [Cars](https://drive.google.com/open?id=1wHhtmYiBZexR2UMjBNuV-1J9ELZ9NV7n), [Persons](https://drive.google.com/open?id=13Dn9_K-DvElBKGpc_AcwRNr6gmSEC6cR)



## Excecuting on a video
Run the following commands to run the detector on a video
```
sh test.sh 0 0 demo/
```

### 4. Sample Results on the Demo videos

<p align="center">
    <img src="demo/a.gif", width="480">
    <br>
</p>


