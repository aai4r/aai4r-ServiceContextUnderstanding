# Service Context Understanding Module

This is an implementation of Service Context Understanding Module in [Cloud Robot Project](https://github.com/aai4r/aai4r-master).
The module has two main parts, an object detector and a food classifier.
From the input image, object detector detects three categories, food, tableware, and drink.
Food classifier crops the food area and classify it.
The classification class is 150 Korean foods ([Kfood](https://www.aihub.or.kr/)).
We will provide the model trained on [Food101](https://www.kaggle.com/dansbecker/food-101), [FoodX251](https://github.com/karansikka1/iFood_2019).

### Environment
Follow the environment in [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).

### Installation

1. Clone this repository.
    ```
    git clone https://github.com/aai4r/aai4r-ServiceContextUnderstanding
    cd aai4r-ServiceContextUnderstanding
    ```

2. Clone [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and install on faster-rcnn.pytorch folder.
    ```
    git clone https://github.com/jwyang/faster-rcnn.pytorch.git
    cd faster-rcnn.pytorch && mkdir data
    pip install -r requirements.txt
    cd lib
    sh make.sh 
    ```

3. Make output folder and copy all faster-rcnn files
    ```
    cd ../.. && mkdir output
    cp faster-rcnn.pytorch/* .
    ```

4. Download all weight files(detection and classification) from [this link](https://drive.google.com/drive/folders/1rT2DYaiywGt8gqdl2YGnd6RLP1rxZV9I?usp=sharing) and move them to output folder.

5. Update model files to faster-rcnn.pytorch
    ```
    cp lib/model/faster_rcnn/faster_rcnn.py faster-rcnn.pytorch/lib/model/faster_rcnn/faster_rcnn.py
    cp lib/model/faster_rcnn/resnet.py faster-rcnn.pytorch/lib/model/faster_rcnn/resnet.py
    ```
   
   
### Run
#### Webcamera demo

Run the demo code with the webcam number (we used 0).
   ```bash
   python my_demo_det_cls.py --webcam_num 0
   ```
   
#### Image demo

Run the demo code with the sample images.
   ```bash
   python my_demo_det_cls.py --image_dir sample_images
   ```
   
#### CLOi demo

The code will be updated.
