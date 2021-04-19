from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO

import numpy as np
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections_korean_ext2
from model.utils.blob import im_list_to_blob
# from model.utils.parser_func import set_dataset_args
from torchvision.ops import nms
import network
import pretrained_utils_v2 as utils
import torchvision
import pickle

# from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet

import pdb

class FoodClassifier:
    # eval_crop_type: 'TenCrop' or 'CenterCrop'
    def __init__(self, net, dbname, eval_crop_type, ck_file_folder):
        self.eval_crop_type = eval_crop_type
        # load class info
        path_class_to_idx = os.path.join(ck_file_folder, 'class_info_%s.pkl' % dbname)
        if os.path.exists(path_class_to_idx):
            fid = open(path_class_to_idx, 'rb')
            self.class_to_idx = pickle.load(fid)
            fid.close()
        else:
            raise AssertionError('%s file is not exists' % path_class_to_idx)

        self.idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())

        # create model
        self.model = network.pret_torch_nets(model_name=net, pretrained=True, class_num=len(self.class_to_idx))
        self.test_transform = utils.TransformImage(self.model.model, crop_type=eval_crop_type,
                                          rescale_input_size=1.0)
        checkpoint = torch.load(os.path.join(ck_file_folder, 'model_best_{}_{}.pth.tar'.format(net, dbname)))
        self.model.load_state_dict(checkpoint['state_dict'])

    def classify(self, image):
        self.model.eval()
        output = self.model(image)

        return output


# TODO: implement FoodDetector
class FoodDetector(object):
    def __init__(self, model_path='output', use_cude=True):
        self.load(model_path, use_cude)

    def load(self, path, use_cude):
        # define options
        path_model_detector = os.path.join(path, 'faster_rcnn_1_7_9999.pth')    # model for detector (food, tableware, drink)

        dataset = 'OpenImageSimpleCategory_test'
        load_name = path_model_detector

        net = 'resnet50'
        prep_type = 'caffe'
        cfg_file = os.path.join(model_path, '{}.yml'.format(net))

        self.cuda = use_cude
        self.class_agnostic = False
        dataset_t = ''  # assign dummy naming

        self.vis = True # generate debug images

        # Load food classifier
        # possible dbname='FoodX251', 'Food101', 'Kfood'
        # possible eval_crop_type='CenterCrop', 'TenCrop'
        self.food_classifier = FoodClassifier(net='senet154',
                                         dbname='Kfood',
                                         eval_crop_type='CenterCrop',
                                         ck_file_folder=model_path)

        # print('Called with args:')
        # args = set_dataset_args(args, test=True)
        imdb_name = imdbval_name = "OpenImageSimpleCategory_test"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']


        if cfg_file is not None:
            cfg_from_file(cfg_file)
        if set_cfgs is not None:
            cfg_from_list(set_cfgs)

        cfg.USE_GPU_NMS = self.cuda

        print('Using config:')
        pprint.pprint(cfg)

        # train set
        # -- Note: Use validation set and disable the flipped to enable faster loading.
        input_dir = load_name
        if not os.path.exists(input_dir):
            raise Exception('There is no input directory for loading network from ' + input_dir)

        if dataset == 'OpenImageSimpleCategory_test' or dataset == 'OpenImageSimpleCategory_validation':
            self.pascal_classes = np.asarray(['__background__',  # always index 0
                                         '음식',
                                         '식기',
                                         '음료'])
            self.thresh = [0., 0.9, 0.5, 0.7]

            self.list_box_color = [(0, 0, 0),
                              (0, 0, 200),
                              (0, 200, 0),
                              (200, 0, 0)]
        else:
            raise AssertionError('check dataset')

        pathOutputSave = load_name
        pathOutputSave = '/'.join(os.path.split(pathOutputSave)[:-1])
        pathOutputSave = pathOutputSave + '/' + dataset
        pathOutputSaveImages = os.path.join(pathOutputSave, 'det_cls_images')
        if not os.path.exists(pathOutputSaveImages):
            print('make a directory for save images: %s' % pathOutputSaveImages)
            os.makedirs(pathOutputSaveImages)

        if net == 'resnet50':
            self.fasterRCNN = resnet(self.pascal_classes, use_pretrained=False, num_layers=50, class_agnostic=self.class_agnostic)
        else:
            raise AssertionError("network is not defined")

        self.fasterRCNN.create_architecture()

        print("load checkpoint %s" % (load_name))
        if self.cuda > 0:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')

        print("load checkpoint %s" % (load_name))

        # initilize the tensor holder here.
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if self.cuda > 0:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        # make variable
        self.im_data = Variable(self.im_data, volatile=True)
        self.im_info = Variable(self.im_info, volatile=True)
        self.num_boxes = Variable(self.num_boxes, volatile=True)
        self.gt_boxes = Variable(self.gt_boxes, volatile=True)

        if self.cuda:
            cfg.CUDA = True

        if self.cuda:
            self.fasterRCNN.cuda()

        self.fasterRCNN.eval()

        print('- models loaded from {}'.format(path))

    def detect(self, cv_img, is_rgb=False):
        # - image shape is (height,width,no_channels)
        print('- input image shape: {}'.format(cv_img.shape))
        # - result is a list of [x1,y1,x2,y2,class_id]
        results = []

        im_in = np.array(cv_img)

        if is_rgb:
            im = im_in[:, :, ::-1]      # rgb -> bgr
        else:
            im = im_in

        blobs, im_scales = self._get_image_blob(im)

        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 5).zero_()
            self.num_boxes.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    if self.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if self.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        if self.vis:
            im2show = np.copy(im)

        im_pil = torchvision.transforms.ToPILImage(mode=None)(im[:, :, ::-1])
        im_width, im_height = im_pil.size

        for j in range(1, len(self.pascal_classes)):
            inds = torch.nonzero(scores[:, j] > self.thresh[j]).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                # im: original image, (768, 1024, 3)
                # im_data: blob image, (1, 3, 600, 800)
                # cls_dets: x1, y1, x2, y2, score

                # crop and feed to classifier
                # im_pil.save(osp.join(pathOutputSaveImages, 'debug_input.png'))
                if j == 1:  # food
                    for k in range(cls_dets.shape[0]):
                        crop_margin_ratio = 0.1

                        x1 = int(cls_dets[k, 0])
                        y1 = int(cls_dets[k, 1])
                        x2 = int(cls_dets[k, 2])
                        y2 = int(cls_dets[k, 3])

                        crop_h_margin = (y2 - y1) * crop_margin_ratio/2.
                        crop_w_margin = (x2 - x1) * crop_margin_ratio/2.

                        x1 = x1 - crop_w_margin
                        y1 = y1 - crop_h_margin
                        x2 = x2 + crop_w_margin
                        y2 = y2 + crop_h_margin

                        if x1 < 0: x1 = 0
                        if y1 < 0: y1 = 0
                        if x2 > im_width-1: x2 = im_width-1
                        if y2 > im_height-1: y2 = im_height-1

                        im_crop = im_pil.crop((x1, y1, x2, y2))
                        # im_crop.save(osp.join(pathOutputSaveImages, 'debug_crop.png'))

                        im_crop = self.food_classifier.test_transform(im_crop)
                        im_crop = torch.unsqueeze(im_crop, dim=0)

                        if self.food_classifier.eval_crop_type == 'TenCrop':
                            bs, ncrops, c, h, w = im_crop.size()
                            im_crop = im_crop.view(-1, c, h, w)

                        food_output = self.food_classifier.classify(im_crop)

                        if self.food_classifier.eval_crop_type == 'TenCrop':
                            food_output = food_output.view(bs, ncrops, -1).mean(1)  # avg over crops

                        topk_score, topk_index = torch.topk(food_output, 5, dim=1)

                        food_class = [self.food_classifier.idx_to_class[topk_index[0][l].item()] for l in range(5)]
                        food_score = torch.nn.functional.softmax(topk_score[0], dim=0)

                        # print(food_class, food_score)

                        # - result is a list of [x1,y1,x2,y2,class_id]
                        bbox_draw = cls_dets.cpu().numpy()[k:k + 1, :]
                        class_name_w_food = '%s (%s)' % (self.pascal_classes[j], food_class[0])
                        results.append([int(bbox_draw[0][0]), int(bbox_draw[0][1]), int(bbox_draw[0][2]), int(bbox_draw[0][3]), class_name_w_food])

                        if self.vis:
                            # class_name_w_food = '%s (%s: %.2f)'%(pascal_classes[j], food_class[0], food_score[0].item())
                            im2show = vis_detections_korean_ext2(im2show, class_name_w_food, bbox_draw,
                                                            box_color=self.list_box_color[j], text_color=(255, 255, 255),
                                                                 text_bg_color=self.list_box_color[j], fontsize=20, thresh=self.thresh[j],
                                                                 draw_score=False, draw_text_out_of_box=True)
                else:
                    # - result is a list of [x1,y1,x2,y2,class_id]
                    bbox_draw = cls_dets.cpu().numpy()
                    results.append([int(bbox_draw[0][0]), int(bbox_draw[0][1]), int(bbox_draw[0][2]), int(bbox_draw[0][3]), self.pascal_classes[j]])

                    if self.vis:
                        im2show = vis_detections_korean_ext2(im2show, self.pascal_classes[j], bbox_draw,
                                                            box_color=self.list_box_color[j], text_color=(255, 255, 255),
                                                             text_bg_color=self.list_box_color[j], fontsize=20, thresh=self.thresh[j],
                                                             draw_score=False, draw_text_out_of_box=True)

            if self.vis:
                cv2.imwrite('debug.png', im2show)
                print('debug.png is savedc')

        return results

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)


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
        results = self.food_detector.detect(pixels, is_rgb=True)

        # 3. Print the result
        print("Detection Result:")
        for result in results:
            print("  BBox(x1={},y1={},x2={},y2={}) => {}".format(result[0],result[1],result[2],result[3],result[4]))

if __name__ == '__main__':
    image_url = 'https://ppss.kr/wp-content/uploads/2019/08/03-62-540x362.jpg'
    # TODO: set model_path
    model_path = './output'
    handler = FoodDetectionRequestHandler(model_path)   # init
    print('FoodDetectionRequestHandler is initialized!')
    handler.process_inference_request(image_url)        # request
    print('FoodDetectionRequestHandler request is processed!')