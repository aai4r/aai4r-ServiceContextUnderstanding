# Service Context Understanding Module

This is an implementation of Service Context Understanding Module in [Cloud Robot Project](https://github.com/aai4r/aai4r-master).
The module has two main parts, an object detector and a food classifier.
From the input image, object detector detects three categories, food, tableware, and drink.
Food classifier crops the food area and classify it.
The classification class is 150 Korean foods ([Kfood](https://www.aihub.or.kr/)).
We will provide the model trained on [Food101](https://www.kaggle.com/dansbecker/food-101), [FoodX251](https://github.com/karansikka1/iFood_2019).

We borrowed some code from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). 

### Environment
* python 3.6
* pytorch 1.3.0
* pytorchvision 0.4.1

### Installation
0. Setup the environment
    ```bash
    conda create -n py36torch13 python=3.6 
    conda activate py36torch13
    conda install pytorch=1.3.0 torchvision=0.4.1 -c pytorch
    ```

1. Clone this repository.
    ```bash
    git clone https://github.com/aai4r/aai4r-ServiceContextUnderstanding
    cd aai4r-ServiceContextUnderstanding
    ```

2. Install required modules
    ```bash
    pip install pretrainedmodels
    pip install opencv-python
    pip install numpy
    pip install imageio
    pip install easydict
    pip install pyyaml
    ```

3. Make output folder and download [all weight files (detection and classification)](https://drive.google.com/drive/folders/1rT2DYaiywGt8gqdl2YGnd6RLP1rxZV9I?usp=sharing) and move them to output folder.
    ```bash
    mkdir output
    ```
    ```bash
    ├── output
    │   ├── class_info_Food101.pkl
    │   ├── class_info_FoodX251.pkl
    │   ├── class_info_Kfood.pkl
    │   ├── faster_rcnn_1_7_9999.pth
    │   └── model_best.pth.tar
    ```
 
4. Download [nanumgothic.ttf](https://fonts.google.com/download?family=Nanum%20Gothic) and Install
   ```
   Copyright (c) 2010, NAVER Corporation (https://www.navercorp.com/),

   with Reserved Font Name Nanum, Naver Nanum, NanumGothic, Naver NanumGothic, NanumMyeongjo, 
   Naver NanumMyeongjo, NanumBrush, Naver NanumBrush, NanumPen, Naver NanumPen, Naver NanumGothicEco, 
   NanumGothicEco, Naver NanumMyeongjoEco, NanumMyeongjoEco, Naver NanumGothicLight, NanumGothicLight, 
   NanumBarunGothic, Naver NanumBarunGothic, NanumSquareRound, NanumBarunPen, MaruBuri

   This Font Software is licensed under the SIL Open Font License, Version 1.1.
   This license is copied below, and is also available with a FAQ at: http://scripts.sil.org/OFL

   SIL OPEN FONT LICENSE
   Version 1.1 - 26 February 2007 
   ```
   
### Run
#### Webcamera demo

Run the demo code with the webcam number (we used 0).
   ```bash
   python my_demo_det_multi.py --webcam_num 0
   ```
   
#### Image demo

Run the demo code with the sample images.
   ```bash
   python my_demo_det_multi.py --image_dir sample_images
   ```
   
#### CLOi demo

The code will be updated.
