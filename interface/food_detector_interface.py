import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO


# TODO: implement FoodDetector
class FoodDetector(object):
    def __init__(self, model_path):
        self.load(model_path)

    def load(self, path):
        print('- models loaded from {}'.format(path))

    def detect(self, img):
        print('- input image shape: {}'.format(img.shape))
        results = [[100,100,200,200,123],[200,200,300,300,111]]
        return results


class FoodDetectionRequestHandler(object):

    def __init__(self, model_path):
        self.food_detector = FoodDetector(model_path)

    def process_inference_request(self, img_url):
        # 1. Read an image and convert it to a numpy array
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        pixels = np.array(img)

        # 2. Perform food detection
        # - result is a list of [x1,y1,x2,y2,class_id]
        # - ex) result = [(100,100,200,200,154), (200,300,200,300,12)]
        results = self.food_detector.detect(pixels)

        # 3. Print the result
        print("Detection Result:")
        for result in results:
            print("  BBox(x1={},y1={},x2={},y2={}) => {}".format(result[0],result[1],result[2],result[3],result[4]))


if __name__ == '__main__':
    image_url = 'https://ppss.kr/wp-content/uploads/2019/08/03-62-540x362.jpg'
    # TODO: set model_path
    model_path = '/abc/de'
    handler = FoodDetectionRequestHandler(model_path)
    handler.process_inference_request(image_url)