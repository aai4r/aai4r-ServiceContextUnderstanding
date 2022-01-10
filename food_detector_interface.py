from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import requests
import json
from PIL import Image
from io import BytesIO

#import _init_paths
import os
import os.path as osp
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable

# from scipy.misc import imread   # scipy.__version__ < 1.2.0
from imageio import imread  # new
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections_korean_ext2, vis_detections_korean_ext2_wShare
from model.utils.blob import im_list_to_blob
# from model.utils.parser_func import set_dataset_args
from torchvision.ops import nms
import network
import pretrained_utils_v2 as utils
import torchvision
import pickle
import copy
import pdb

# from lib.model.faster_rcnn.vgg16 import vgg16
#from lib.model.faster_rcnn.resnet import resnet

import pdb

class FoodClassifier:
    # eval_crop_type: 'TenCrop' or 'CenterCrop'
    def __init__(self, net, dbname, eval_crop_type, ck_file_folder, use_cuda=True, pretrained=True):
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
        self.model = network.pret_torch_nets(model_name=net, pretrained=pretrained, class_num=len(self.class_to_idx))
        if pretrained == False:
            self.model.model.input_size = [3, 224, 224]
            self.model.model.input_space = 'RGB'
            self.model.model.input_range = [0, 1]
            self.model.model.mean = [0.485, 0.456, 0.406]
            self.model.model.std = [0.229, 0.224, 0.225]
        self.test_transform = utils.TransformImage(self.model.model, crop_type=eval_crop_type, rescale_input_size=1.0)

        if use_cuda is False:
            checkpoint = torch.load(os.path.join(ck_file_folder, 'model_best_{}_{}.pth.tar'.format(net, dbname)), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(os.path.join(ck_file_folder, 'model_best_{}_{}.pth.tar'.format(net, dbname)))
        self.model.load_state_dict(checkpoint['state_dict'])

    def classify(self, image):
        self.model.eval()
        output = self.model(image)

        return output


# TODO: implement FoodDetector
class FoodDetector(object):
    def __init__(self, model_path='output', use_cuda=True):
        self.load(model_path, use_cuda)
        self.vis_th = 0.8
        self.save_result = True
        self.thresh = 0.05

    def load(self, path, use_cuda):
        # define options
        path_model_detector = os.path.join(path, 'fpn101_1_10_9999.pth')    # model for detector (food, tableware, drink)

        dataset = 'CloudStatus_val'
        imdb_name2 = 'CloudTableThings_val'
        total_imdb_name = 'CloudStatusTableThings_val'
        load_name = path_model_detector

        self.use_share_regress = True
        self.use_progress = True

        net = 'resnet101'
        self.cuda = use_cuda
        self.class_agnostic = False
        self.att_type = 'None'

        self.vis = True # generate debug images

        # Load food classifier
        # possible dbname='FoodX251', 'Food101', 'Kfood'
        # possible eval_crop_type='CenterCrop', 'TenCrop'
        self.food_classifier = FoodClassifier(net='senet154',
                                              dbname='Kfood',
                                              eval_crop_type='CenterCrop',
                                              ck_file_folder=path,
                                              use_cuda=use_cuda,
                                              pretrained=False)

        cfg_file = os.path.join(path, '{}_ls.yml'.format(net))
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']


        if cfg_file is not None:
            cfg_from_file(cfg_file)
        if set_cfgs is not None: 
            cfg_from_list(set_cfgs)

        USE_GPU_NMS = self.cuda

        print('Using config:')
        pprint.pprint(cfg)

        # train set
        # -- Note: Use validation set and disable the flipped to enable faster loading.
        input_dir = load_name
        if not os.path.exists(input_dir):
            raise Exception('There is no input directory for loading network from ' + input_dir)

        self.list_box_color = [(0, 0, 0),
                        (0, 0, 200),
                        (0, 200, 0),
                        (200, 200, 0),
                        (200, 0, 200),
                        (0, 200, 200),
                        (200, 0, 200)]

        self.classes0 = self.get_class_list(dataset)
        self.classes1 = self.get_class_list(imdb_name2)
        self.classes_total = self.get_class_list(total_imdb_name)

        from model.fpn.resnet_multi_CBAM import resnet
        self.fasterRCNN = resnet(self.classes0, self.classes1, use_pretrained=False, num_layers=101,
                            class_agnostic=self.class_agnostic, use_share_regress=self.use_share_regress,
                            use_progress=self.use_progress, att_type=self.att_type)

        self.fasterRCNN.create_architecture()

        print("loading checkpoint %s..." % (load_name))
        if self.cuda > 0:
            checkpoint = torch.load(load_name)
        else:
            checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('succeeded')

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
        with torch.no_grad():
            self.im_data = Variable(self.im_data)
            self.im_info = Variable(self.im_info)
            self.num_boxes = Variable(self.num_boxes)
            self.gt_boxes = Variable(self.gt_boxes)

        if self.cuda > 0:
            cfg.CUDA = True

        if self.cuda > 0:
            self.fasterRCNN.cuda()

        self.fasterRCNN.eval()

        print('- models loaded from {}'.format(path))

    def get_overlap_ratio_meal(self, food_bbox, dish_bbox):
        a_xmax = max(food_bbox[0], food_bbox[2])
        a_xmin = min(food_bbox[0], food_bbox[2])
        a_ymax = max(food_bbox[1], food_bbox[3])
        a_ymin = min(food_bbox[1], food_bbox[3])

        b_xmax = max(dish_bbox[0], dish_bbox[2])
        b_xmin = min(dish_bbox[0], dish_bbox[2])
        b_ymax = max(dish_bbox[1], dish_bbox[3])
        b_ymin = min(dish_bbox[1], dish_bbox[3])

        # a: food, b: dish
        dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
        dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)

        # dx and dy is width and height of IoU

        if (dx >= 0) and (dy >= 0):
            return float(dx * dy) / float(
                (a_xmax - a_xmin) * (a_ymax - a_ymin))
        else:
            return 0.

    def detect(self, cv_img, is_rgb=True):
        # - image shape is (height,width,no_channels)
        # print('- input image shape: {}'.format(cv_img.shape))
        # - result is a list of [x1,y1,x2,y2,class_id]
        results = []

        im_in = np.array(cv_img)

        if is_rgb:
            im = im_in[:, :, ::-1]      # rgb -> bgr
        else:
            im = im_in

        blobs, im_scales = self._get_image_blob(im)     # prep_type = 'caffe' is applied
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

        rois0, cls_prob0, bbox_pred0, _, _, _, _, _, share_pred0, _, progress_pred0, _ = \
            self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes, flow_id=0)
        rois1, cls_prob1, bbox_pred1, _, _, _, _, _, share_pred1, _, _, _ = \
            self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes, flow_id=1)


        # rois0 # [1, 300, 5]
        rois = torch.cat((rois0, rois1), dim=1)

        # share_pred0 # [1, 300, 1]
        aaaa = torch.ones((share_pred0.shape[0], rois0.shape[1], 1)).cuda() * share_pred1
        share_pred = torch.cat((share_pred0, aaaa), dim=1)

        progress_pred = progress_pred0

        # cls_prob0 # [1, 300, 3]
        # bbox_pred0 # [1, 300, 12 (3x4)

        cls_prob = torch.zeros((cls_prob0.shape[0],
                                cls_prob0.shape[1]+cls_prob1.shape[1],
                                len(self.classes_total))).cuda()
        bbox_pred = torch.zeros((bbox_pred0.shape[0],
                                 bbox_pred0.shape[1] + bbox_pred1.shape[1],
                                 4*len(self.classes_total))).cuda()

        for j, j_name in enumerate(self.classes_total):
            if j_name in self.classes0:
                j_idx = (j_name == self.classes0).nonzero()[0][0]
                num_batch0 = cls_prob0.shape[1]
                cls_prob[:, :num_batch0, j] = cls_prob0[:, :, j_idx]
                bbox_pred[:, :num_batch0, j * 4:(j + 1) * 4] = bbox_pred0[:, :, j_idx * 4:(j_idx + 1) * 4]

            if j_name in self.classes1:
                j_idx = (j_name == self.classes1).nonzero()[0][0]
                num_batch1 = cls_prob1.shape[1]
                cls_prob[:, num_batch0:num_batch0+num_batch1, j] = cls_prob1[:, :, j_idx]
                bbox_pred[:, num_batch0:num_batch0+num_batch1, j * 4:(j + 1) * 4] = bbox_pred1[:, :, j_idx * 4:(j_idx + 1) * 4]

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
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.classes_total))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        if self.use_share_regress:
            share_pred = share_pred.squeeze()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if self.vis or self.save_result:
            h, w = im.shape[:2]
            if max(h, w) > 800:
                if h > 800:
                    im2show = cv2.resize(im, (int(800 / h * w), 800))
                if w > 800:
                    im2show = cv2.resize(im, (800, int(800 / w * h)))

                h_display = im2show.shape[0]
                im_scale = h_display / h
            else:
                im2show = np.copy(im)
                im_scale = 1.0

        im_pil = torchvision.transforms.ToPILImage(mode=None)(im[:,:,::-1])
        im_width, im_height = im_pil.size

        # for j in range(1, len(self.classes_total)):
        #     inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)    # find index with scores > threshold in j-class
        #     # if there is det
        #     if inds.numel() > 0:
        #         cls_scores = scores[:, j][inds]
        #         if self.use_share_regress:
        #             share_pred_inds = share_pred[inds]
        #
        #         _, order = torch.sort(cls_scores, 0, True)
        #         if self.class_agnostic:
        #             cls_boxes = pred_boxes[inds, :]
        #         else:
        #             cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
        #
        #         if self.use_share_regress:
        #             cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), share_pred_inds.unsqueeze(1)), 1)
        #         else:
        #             cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        #         # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        #         cls_dets = cls_dets[order]
        #         keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        #         cls_dets = cls_dets[keep.view(-1).long()]
        #
        #         # im: original image, (768, 1024, 3)
        #         # im_data: blob image, (1, 3, 600, 800)
        #         # cls_dets: x1, y1, x2, y2, score
        #
        #         # crop and feed to classifier
        #         # im_pil.save(osp.join(pathOutputSaveImages, 'debug_input.png'))
        #         if self.classes_total[j] == 'food':
        #             for k in range(cls_dets.shape[0]):
        #                 crop_margin_ratio = 0.1
        #
        #                 x1 = int(cls_dets[k, 0])
        #                 y1 = int(cls_dets[k, 1])
        #                 x2 = int(cls_dets[k, 2])
        #                 y2 = int(cls_dets[k, 3])
        #
        #                 crop_h_margin = (y2 - y1) * crop_margin_ratio/2.
        #                 crop_w_margin = (x2 - x1) * crop_margin_ratio/2.
        #
        #                 x1 = x1 - crop_w_margin
        #                 y1 = y1 - crop_h_margin
        #                 x2 = x2 + crop_w_margin
        #                 y2 = y2 + crop_h_margin
        #
        #                 if x1 < 0: x1 = 0
        #                 if y1 < 0: y1 = 0
        #                 if x2 > im_width-1: x2 = im_width-1
        #                 if y2 > im_height-1: y2 = im_height-1
        #
        #                 im_crop = im_pil.crop((x1, y1, x2, y2))
        #                 # im_crop.save(osp.join(pathOutputSaveImages, 'debug_crop.png'))
        #
        #                 im_crop = self.food_classifier.test_transform(im_crop)
        #                 im_crop = torch.unsqueeze(im_crop, dim=0)
        #
        #                 if self.food_classifier.eval_crop_type == 'TenCrop':
        #                     bs, ncrops, c, h, w = im_crop.size()
        #                     im_crop = im_crop.view(-1, c, h, w)
        #
        #                 food_output = self.food_classifier.classify(im_crop)
        #
        #                 if self.food_classifier.eval_crop_type == 'TenCrop':
        #                     food_output = food_output.view(bs, ncrops, -1).mean(1)  # avg over crops
        #
        #                 topk_score, topk_index = torch.topk(food_output, 5, dim=1)
        #
        #                 food_class = [self.food_classifier.idx_to_class[topk_index[0][l].item()] for l in range(5)]
        #                 food_score = torch.nn.functional.softmax(topk_score[0], dim=0)
        #
        #                 if self.vis or self.save_result:
        #                     bbox_draw = cls_dets.detach().cpu().numpy()[k:k + 1, :]
        #                     bbox_draw[:, :4] = bbox_draw[:, :4] * im_scale
        #
        #                     # - result is a list of [x1,y1,x2,y2,class_id]
        #                     results.append([int(bbox_draw[0][0]), int(bbox_draw[0][1]), int(bbox_draw[0][2]), int(bbox_draw[0][3]),
        #                                     self.classes_total[j],
        #                                     topk_index[0][0].item(),
        #                                     food_class[0],
        #                                     bbox_draw[0][5].item()])
        #
        #                     # class_name_w_food = '%s (%s: %.2f)'%(pascal_classes[j], food_class[0], food_score[0].item())
        #                     class_name_w_food = '%s (%s)'%(self.classes_total[j], food_class[0])
        #                     im2show = vis_detections_korean_ext2_wShare(im2show, class_name_w_food, bbox_draw,
        #                                                          box_color=self.list_box_color[j], text_color=(255, 255, 255),
        #                                                          text_bg_color=self.list_box_color[j], fontsize=20, thresh=self.vis_th,
        #                                                          draw_score=False, draw_text_out_of_box=True)
        #         else:
        #             if self.vis or self.save_result:
        #                 bbox_draw = cls_dets.detach().cpu().numpy()
        #                 bbox_draw[:, :4] = bbox_draw[:, :4] * im_scale
        #
        #                 results.append([int(bbox_draw[0][0]), int(bbox_draw[0][1]), int(bbox_draw[0][2]), int(bbox_draw[0][3]),
        #                                 self.classes_total[j],
        #                                 0,
        #                                 0,
        #                                 0])
        #
        #                 im2show = vis_detections_korean_ext2(im2show, self.classes_total[j], bbox_draw,
        #                                                      box_color=self.list_box_color[j], text_color=(255, 255, 255),
        #                                                      text_bg_color=self.list_box_color[j], fontsize=20, thresh=self.vis_th,
        #                                                      draw_score=False, draw_text_out_of_box=True)

        for j in range(1, len(self.classes_total)):
            inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)    # find index with scores > threshold in j-class
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                if self.use_share_regress:
                    share_pred_inds = share_pred[inds]

                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                if self.use_share_regress:
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), share_pred_inds.unsqueeze(1)), 1)
                else:
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
                if self.classes_total[j] == 'food':
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

                        bbox_draw = cls_dets.detach().cpu().numpy()[k:k + 1, :]
                        bbox_draw[:, :4] = bbox_draw[:, :4] * im_scale

                        if bbox_draw[0, 4] >= self.vis_th:
                            # - result is a list of [x1,y1,x2,y2,class_id]
                            results.append([int(bbox_draw[0][0]), int(bbox_draw[0][1]), int(bbox_draw[0][2]), int(bbox_draw[0][3]),
                                            self.classes_total[j],
                                            topk_index[0][0].item(),        # food_class index
                                            food_class[0],
                                            bbox_draw[0][5].item()])
                else:
                    bbox_draw = cls_dets.detach().cpu().numpy()
                    bbox_draw[:, :4] = bbox_draw[:, :4] * im_scale

                    for k in range(cls_dets.shape[0]):
                        if bbox_draw[k, 4] >= self.vis_th:
                            results.append([int(bbox_draw[k][0]), int(bbox_draw[k][1]), int(bbox_draw[k][2]), int(bbox_draw[k][3]),
                                            self.classes_total[j],
                                            0,
                                            0,
                                            0])

        # dish-food converter
        # every dish find the food and its amount
        # if food is not found, zero amount is assigned.
        print('0.results:', results)
        new_results = []
        for item in results:
            x1, y1, x2, y2, class_name, food_index, food_name, food_amount = item

            if class_name == 'dish':
                new_results.append(item)
        print('1.new_results:', new_results)

        for item in results:
            x1, y1, x2, y2, class_name, food_index, food_name, food_amount = item

            if class_name == 'food':
                is_find_dish = False
                for dish_i, dish_item in enumerate(new_results):
                    d_x1, d_y1, d_x2, d_y2, _, _, _, dish_amount = dish_item

                    # check overlap
                    overlap_ratio = self.get_overlap_ratio_meal(food_bbox=[x1, y1, x2, y2],
                                                                dish_bbox=[d_x1, d_y1, d_x2, d_y2])
                    if overlap_ratio > 0.9:
                        new_results[dish_i][5] = food_index
                        new_results[dish_i][6] = food_name
                        new_results[dish_i][7] += food_amount

                        is_find_dish = True

                if not is_find_dish:
                    new_results.append(item)
        print('2.new_results:', new_results)

        for dish_i, dish_item in enumerate(new_results):
            # x1, y1, x2, y2, class_name, food_index, food_name, food_amount = item
            # new_results[dish_i][4] = 'food'
            # new_results[dish_i][6] = 'food'
            if new_results[dish_i][5] == 94 or new_results[dish_i][5] == 64:
                new_results[dish_i][4] = 'drink'
            else:
                new_results[dish_i][4] = 'food'

            new_amount = new_results[dish_i][7]
            if new_amount > 1.0: new_amount = 1.0
            if new_amount < 0.0: new_amount = 0.0
            new_results[dish_i][7] = int(round(new_amount * 100))

        old_results = copy.copy(results)
        results = copy.copy(new_results)
        # dish-food converter - end

        print('3.results: ', results)

        # if self.save_result:
        #     for item in old_results:
        #         # item = [x1, y1, x2, y2, category, (food_index), (food_name), (amount)]
        #         if item[4] == 'food':
        #             str_name = '%s (%s, %s, %.2f)' % (item[4], item[5], item[6], item[7])
        #         else:
        #             str_name = '%s' % (item[0])
        #
        #         bbox_draw = np.array([[item[0], item[1], item[2], item[3], 1.0]])
        #
        #         color_index = 0
        #         im2show = vis_detections_korean_ext2(im2show, str_name, bbox_draw,
        #                                                     box_color=self.list_box_color[color_index], text_color=(255, 255, 255),
        #                                                     text_bg_color=self.list_box_color[color_index], fontsize=20,
        #                                                     thresh=self.vis_th,
        #                                                     draw_score=False, draw_text_out_of_box=False)

        if self.save_result:
            for item in results:
                # item = [x1, y1, x2, y2, category, (food_name), (amount)]
                if item[4] == 'food':
                    str_name = '%s (%.2f)' % (item[4], item[7])
                else:
                    str_name = '%s' % (item[4])

                bbox_draw = np.array([[item[0], item[1], item[2], item[3], 1.0]])

                color_index = 1
                im2show = vis_detections_korean_ext2(im2show, str_name, bbox_draw,
                                                            box_color=self.list_box_color[color_index], text_color=(255, 255, 255),
                                                            text_bg_color=self.list_box_color[color_index], fontsize=20,
                                                            thresh=self.vis_th,
                                                            draw_score=False, draw_text_out_of_box=True)

        if self.vis:
            cv2.imwrite('debug.png', im2show)
            #print('debug.png is saved.')

        return results, im2show


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
        im_orig -= cfg.PIXEL_MEANS      # BGR order

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


    def get_class_list(self, dataset_name):
        if dataset_name == 'OpenImageSimpleCategory_test' or dataset_name == 'OpenImageSimpleCategory_validation':
            pascal_classes = np.asarray(['__background__',  # always index 0
                                        'food',
                                        'tableware',
                                        'drink'])
        elif dataset_name == 'CloudStatus_val':
            pascal_classes = np.asarray(['__background__',
                                        'food', 'dish', ])
        elif dataset_name == 'CloudTableThings_val':
            pascal_classes = np.asarray(['__background__',  # always index 0
                                        'dish', 'outlery', 'etc',])
        elif dataset_name == 'CloudStatusTableThings_val':
            pascal_classes = np.asarray(['__background__',  # always index 0
                                        'food', 'dish', 'outlery', 'etc', ])
        elif dataset_name == 'CloudTableThingsFineClass_val':
            pascal_classes = np.asarray(['__background__',  # always index 0
                                        'dish', 'chopsticks', 'spoon', 'fork', 'knife',
                                        'wallet', 'mobile_phone', 'tissue_box', 'wet_tissue_pack', 'trash',])
        elif dataset_name == 'CloudStatusTableThingsFineClass_val':
            pascal_classes = np.asarray(['__background__',  # always index 0
                                        'food', 'dish', 'chopsticks', 'spoon', 'fork', 'knife',
                                        'wallet', 'mobile_phone', 'tissue_box', 'wet_tissue_pack', 'trash',])
        else:
            raise AssertionError('%s is unsupported!' % (dataset_name))

        return pascal_classes


class FoodDetectionRequestHandler(object):
    def __init__(self, model_path):
        self.food_detector = FoodDetector(model_path)

    def process_inference_request(self, img):
        # 1. Read an image and convert it to a numpy array
        pixels = np.array(img)

        # 2. Perform food detection
        # - result is a list of [x1,y1,x2,y2,class_id]
        # - ex) result = [(100,100,200,200,154), (200,300,200,300,12)]
        results, vis_img = self.food_detector.detect(pixels, is_rgb=False)

        return results, vis_img

    def process_inference_request_imgurl(self, img_url):
        # 1. Read an image and convert it to a numpy array
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))     # read as rgb
        pixels = np.array(img)

        # 2. Perform food detection
        # - result is a list of [x1,y1,x2,y2,class_id]
        # - ex) result = [(100,100,200,200,154), (200,300,200,300,12)]
        results = self.food_detector.detect(pixels, is_rgb=False)

        return results


if __name__ == '__main__':
    image_url = 'https://ppss.kr/wp-content/uploads/2019/08/03-62-540x362.jpg'
    # TODO: set model_path
    model_path = './output'
    handler = FoodDetectionRequestHandler(model_path)   # init
    print('FoodDetectionRequestHandler is initialized!')
    results, im2show = handler.process_inference_request_imgurl(image_url)        # request

    # 3. Print the result
    print("Detection Result: {}".format(json.dumps(results)))
    for result in results:
        print("  BBox(x1={},y1={},x2={},y2={}) => {}".format(result[0],result[1],result[2],result[3],result[4]))

    print('FoodDetectionRequestHandler request is processed!')