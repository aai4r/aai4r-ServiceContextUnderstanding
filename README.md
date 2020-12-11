# Service Context Understanding Module

This is an implementation of Service Context Understanding Module Module in Cloud Robot Project.
The module has two main parts, object detector and food classifier.
The object detector is based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and the food classifier utilizes SENet. 

### Environment
Follow the environment in [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).

### Installation

1. Clone this repository.
    ```bash
    git clone https://github.com/yochin/aai4r-ServiceConetxtUnderstanding
    cd aai4r-ServiceConetxtUnderstanding
    ```

2. Clone [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and install on faster-rcnn.pytorch folder.
    ```bash
    git clone https://github.com/jwyang/faster-rcnn.pytorch.git
    cd faster-rcnn.pytorch && mkdir data
    pip install -r requirements.txt
    cd lib
    sh make.sh 
    ```

3. Make model folder and copy faster-rcnn.pytorch/cfgs folder to models/
    ```bash
    cd ../.. && mkdir models
    mkdir models/InstModel
    cp -r faster-rcnn.pytorch/cfgs models
    ```

4. Download [the model file](https://www.dropbox.com/s/be0isevd22eikqb/faster_rcnn_1_10_14657.pth?dl=0) from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and move to models folder.
   
   
### Run
#### Object instance registration

1. Run the demo code.
   ```bash
   python demo.py
   ```
2. Press key 'r' and type two information, category_name and instance_name. 
3. Show the object instance until predefined number of images are captured. 
   
#### Object instance detection and recognition
1. Run the demo code.
   ```bash
   python demo.py
   ```
2. Press key 'd' and see the result.

### License

See LICENSE.md
