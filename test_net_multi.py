# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
# from model.roi_layers import nms
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections_korean_ext2_wShare, vis_detections_korean_ext2
from model.utils.parser_func import set_dataset_args

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--imdb_name1', dest='imdb_name1',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--imdb_name2', dest='imdb_name2',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    # parser.add_argument('--load_dir', dest='load_dir',
    #                     help='directory to load models', default="models",
    #                     type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--ss', dest='small_scale',
                        help='whether use small imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--path_load_model', dest='path_load_model',
                        default='', type=str)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')

    parser.add_argument('--use_FPN', dest='use_FPN', action='store_true')
    parser.add_argument('--save_res_img', dest='save_res_img', help='save result images', action='store_true')
    parser.add_argument('--vis_th', dest='vis_th', default=0.7, type=float)
    parser.add_argument('--anchors4', dest='anchors4', action='store_true')
    parser.add_argument('--ratios5', dest='ratios5', action='store_true')
    parser.add_argument('--vis_classes', dest='vis_classes', nargs='+', default=[''], type=str)

    # for table model
    parser.add_argument('--use_share_regress', dest='use_share_regress', action='store_true')
    parser.add_argument('--use_progress', dest='use_progress', action='store_true')

    parser.add_argument('--prep_type', dest='prep_type', default='caffe', type=str)

    parser.add_argument('--att_type', dest='att_type', help='None, BAM, CBAM', default='None', type=str)

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', num_class=1):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.comp_mat = np.zeros((num_class, num_class), np.int)
        self.num_gt_class = np.zeros((num_class), np.int)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, est=0, gt=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if n == 1:
            self.comp_mat[gt, est] += 1
            self.num_gt_class[gt] += 1

        # print(self.comp_mat)
        # print(self.num_gt_class)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def get_value(self):
        return self.avg

    def get_comp_mat(self):
        ret_comp_mat_avg = np.zeros(self.comp_mat.shape, np.float)
        for ith, item in enumerate(self.num_gt_class):
            # self.comp_mat[ith, :] = np.true_divide(self.comp_mat[ith, :], item)
            ret_comp_mat_avg[ith, :] = self.comp_mat[ith, :] / item

        return ret_comp_mat_avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)      # pred = [batch_size, maxk]
        pred = pred.t()        # transpose
        correct = pred.eq(target.view(1, -1).expand_as(pred))       # [maxk, batch_size] = True/False about correct

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)        # correct[:k] == correct[:k, :]
            res.append(correct_k.mul_(100.0 / batch_size))                          # accracy of minibatch in percentage

        return res


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    args.dataset_t = ''  # assign dummy naming
    args = set_dataset_args(args, test=True)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    imdb1, _, _, _ = combined_roidb(args.imdb_name1, False)
    imdb2, _, _, _ = combined_roidb(args.imdb_name2, False)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))

    # for loading share gt
    imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(args.imdbval_name, False)
    imdb_s.competition_mode(on=True)

    classes0 = np.asarray(imdb1.classes)
    classes1 = np.asarray(imdb2.classes)
    classes_total = np.asarray(imdb.classes)

    print('classes0: ', classes0)
    print('classes1: ', classes1)
    print('classes_total: ', classes_total)

    if len(args.vis_classes[0]) == 0:
        args.vis_classes = classes_total[1:]    # except for bg

    if not os.path.exists(args.path_load_model):
        raise Exception('There is no loading model for loading network from ' + args.path_load_model)
    load_name = args.path_load_model

    # initilize the network here.
    if args.use_FPN:
        from model.fpn.resnet_multi_CBAM import resnet

        if args.net == 'resnet101':
            fasterRCNN = resnet(classes0, classes1, use_pretrained=False, num_layers=101,
                                class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress,
                                use_progress=args.use_progress, att_type=args.att_type)
        elif args.net == 'resnet50':
            fasterRCNN = resnet(classes0, classes1, use_pretrained=False, num_layers=50,
                                class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress,
                                use_progress=args.use_progress, att_type=args.att_type)
        else:
            print("network is not defined")
            pdb.set_trace()
    else:
        from model.faster_rcnn.resnet_multi import resnet

        if args.net == 'resnet101':
            fasterRCNN = resnet(classes0, classes1, use_pretrained=False, num_layers=101,
                                class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress,
                                use_progress=args.use_progress)
        elif args.net == 'resnet50':
            fasterRCNN = resnet(classes0, classes1, use_pretrained=False, num_layers=50,
                                class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress,
                                use_progress=args.use_progress)
        else:
            print("network is not defined")
            pdb.set_trace()


    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    gt_progress = torch.LongTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        gt_progress = gt_progress.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    gt_progress = Variable(gt_progress)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = os.path.splitext(load_name)[0]
    output_dir = os.path.join(output_dir, args.imdbval_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path_to_result_images = os.path.join(output_dir, 'result_images')
    if args.save_res_img:
        if not os.path.exists(path_to_result_images):
            os.makedirs(path_to_result_images)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False, prep_type=args.prep_type,
                             share_return=True, progress_return=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)
    data_iter = iter(dataloader)


    # share
    dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, 1, \
                             imdb_s.num_classes, training=True, normalize=False, prep_type=args.prep_type,
                             share_return=True, progress_return=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)
    data_iter_s = iter(dataloader_s)


    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    # top1 = AverageMeter('Acc@1', ':6.2f', num_class=len(imdb._progress_classes))

    prog_m5 = AverageMeter('Acc@1', ':6.2f', num_class=2)   # progress classification w/ margin 5 percent
    prog_m10 = AverageMeter('Acc@1', ':6.2f', num_class=2)  # progress classification w/ margin 10 percent
    prog_m15 = AverageMeter('Acc@1', ':6.2f', num_class=2)  # progress classification w/ margin 15 percent
    prog_m20 = AverageMeter('Acc@1', ':6.2f', num_class=2)  # progress classification w/ margin 20 percent
    prog_m30 = AverageMeter('Acc@1', ':6.2f', num_class=2)  # progress classification w/ margin 30 percent

    sum_share_sqr_error = 0.
    sum_num_boxes = 0

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = next(data_iter)
        with torch.no_grad():
            # data_pt[0]: image [1, 3, 600/H, 1200/W]
            # data_pt[1]: W, H, resized_ratio
            # data_pt[2]: bbox [1, 20, 5], 5 has x1, y1, x2, y2, class_index
            # data_pt[3]: num_bboxes
            # data_pt[4]: [0] path_to_images, data_pt[4][0].split('/')[-1]
            # data_pt[5]: share_gt
            # data_pt[6]: progress_index
            # im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            gt_progress.resize_(data[6].size()).copy_(data[6])

            im_b, im_c, im_h, im_w = data[0].shape
            im_data.resize_([im_b, im_c, im_h - int(im_h / 2), im_w - 2 * int(im_w / 8)]).copy_(
                data[0][:, :, int(im_h / 2):, int(im_w / 8):-int(im_w / 8)])

            im_info[0, 0] = im_data.shape[2]
            im_info[0, 1] = im_data.shape[3]

        det_tic = time.time()
        rois0, cls_prob0, bbox_pred0, _, _, _, _, _, share_pred0, _, progress_pred0, _ = \
            fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=0)
        rois1, cls_prob1, bbox_pred1, _, _, _, _, _, share_pred1, _, _, _ = \
            fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=1)

        # merge two outputs into one
        # rois0/1 # [1, 300, 5]
        rois = torch.cat((rois0, rois1), dim=1)

        # share_pred0 # [1, 300, 1], share_pred1 # 0 (no food in outputs)
        aaaa = torch.ones((share_pred0.shape[0], rois0.shape[1], 1)).cuda() * share_pred1
        share_pred = torch.cat((share_pred0, aaaa), dim=1)

        progress_pred = progress_pred0

        # cls_prob0 # [1, 300, 3]
        # bbox_pred0 # [1, 300, 12 (3x4)
        cls_prob = torch.zeros((cls_prob0.shape[0],
                                cls_prob0.shape[1] + cls_prob1.shape[1],
                                len(classes_total))).cuda()
        bbox_pred = torch.zeros((bbox_pred0.shape[0],
                                 bbox_pred0.shape[1] + bbox_pred1.shape[1],
                                 4 * len(classes_total))).cuda()

        for j, j_name in enumerate(classes_total):
            if j_name in classes0:
                j_idx = (j_name == classes0).nonzero()[0][0]
                num_batch0 = cls_prob0.shape[1]
                cls_prob[:, :num_batch0, j] = cls_prob0[:, :, j_idx]
                bbox_pred[:, :num_batch0, j * 4:(j + 1) * 4] = bbox_pred0[:, :, j_idx * 4:(j_idx + 1) * 4]

            if j_name in classes1:
                j_idx = (j_name == classes1).nonzero()[0][0]
                num_batch1 = cls_prob1.shape[1]
                cls_prob[:, num_batch0:num_batch0+num_batch1, j] = cls_prob1[:, :, j_idx]
                bbox_pred[:, num_batch0:num_batch0+num_batch1, j * 4:(j + 1) * 4] = bbox_pred1[:, :, j_idx * 4:(j_idx + 1) * 4]


        # progress classification - no gt
        # print('gt_progress vs progress_pred: ', gt_progress, progress_pred)
        # progress_pred = torch.zeros(3)
        # acc1 = accuracy(progress_pred, gt_progress)
        # pred_index_progress = torch.argmax(progress_pred[0])
        # top1.update(acc1[0].item(), data[0].shape[0], est=pred_index_progress.item(), gt=gt_progress[0].item())

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        if args.use_share_regress:
            share_pred = share_pred.squeeze()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis or args.save_res_img:
            im = cv2.imread(imdb.image_path_at(i))
            im_h, im_w = im.shape[:2]
            im = im[int(im_h/2):, int(im_w/8):-int(im_w/8), :]
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

        for j in reversed(xrange(1, imdb.num_classes)):
            # print(j)
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]

                if args.use_share_regress:
                    share_pred_inds = share_pred[inds]

                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                object_name = imdb.classes[j]
                if (vis or args.save_res_img) and object_name in args.vis_classes:
                    if args.use_share_regress:
                        cls_dets_scaled = torch.cat((cls_boxes, cls_scores.unsqueeze(1), share_pred_inds.unsqueeze(1)), 1)
                    else:
                        cls_dets_scaled = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                    cls_dets_scaled = cls_dets_scaled[order]
                    cls_dets_scaled = cls_dets_scaled[keep.view(-1).long()]
                    cls_dets_scaled = cls_dets_scaled.detach()
                    cls_dets_scaled[:, :4] = cls_dets_scaled[:, :4] * im_scale

                    if object_name == 'food':
                        box_color = (200, 0, 0)
                        text_bg_color = (220, 0, 0)
                        text_color = (255, 255, 255)
                    elif object_name == 'dish':
                        box_color = (0, 200, 0)
                        text_bg_color = (0, 220, 0)
                        text_color = (255, 255, 255)
                    elif object_name == 'outlery':
                        box_color = (0, 200, 200)
                        text_bg_color = (0, 220, 220)
                        text_color = (0, 0, 0)
                    elif object_name == 'etc':
                        box_color = (200, 0, 200)
                        text_bg_color = (220, 0, 220)
                        text_color = (0, 0, 0)

                    if object_name == 'food' and args.use_share_regress:
                        im2show = vis_detections_korean_ext2_wShare(im2show, imdb.classes[j], cls_dets_scaled.cpu().numpy(), thresh=args.vis_th, draw_score=False,
                                                                    box_color=box_color, text_bg_color=text_bg_color, text_color=text_color)
                    else:
                        im2show = vis_detections_korean_ext2(im2show, imdb.classes[j], cls_dets_scaled.cpu().numpy(), thresh=args.vis_th, draw_score=False,
                                                                    box_color=box_color, text_bg_color=text_bg_color, text_color=text_color)

                all_boxes[j][i] = cls_dets.cpu().numpy()

            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))

        # # TODO: cause an errors
        # # share regression
        # # Assume bbox is correct
        # # _s : _for_share, not source
        # data_s = next(data_iter_s)
        # with torch.no_grad():
        #     # data_pt[0]: image [1, 3, 600/H, 1200/W]
        #     # data_pt[1]: W, H, resized_ratio
        #     # data_pt[2]: bbox [1, 20, 5], 5 has x1, y1, x2, y2, class_index
        #     # data_pt[3]: num_bboxes
        #     # data_pt[4]: [0] path_to_images, data_pt[4][0].split('/')[-1]
        #     # data_pt[5]: share_gt
        #     # data_pt[6]: progress_index
        #     # im_data.resize_(data_s[0].size()).copy_(data_s[0])
        #     im_info.resize_(data_s[1].size()).copy_(data_s[1])
        #     gt_boxes.resize_(data_s[2].size()).copy_(data_s[2])
        #     num_boxes.resize_(data_s[3].size()).copy_(data_s[3])
        #     gt_progress.resize_(data_s[6].size()).copy_(data_s[6])
        #
        #     im_b, im_c, im_h, im_w = data_s[0].shape
        #     im_data.resize_([im_b, im_c, im_h - int(im_h / 2), im_w - 2 * int(im_w / 8)]).copy_(
        #         data_s[0][:, :, int(im_h / 2):, int(im_w / 8):-int(im_w / 8)])
        #
        #     im_info[0, 0] = im_data.shape[2]
        #     im_info[0, 1] = im_data.shape[3]
        #
        #     pdb.set_trace()
        #     frcn_output2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=0, use_gt_bbox_in_rpn=True)
        #     share_pred0 = frcn_output2[8]
        #
        #     frcn_output3 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=1, use_gt_bbox_in_rpn=True)
        #     share_pred1 = frcn_output3[8]
        #
        # # share_pred0 # [1, 300, 1], share_pred1 # 0 (no food in outputs)
        # aaaa = torch.ones((share_pred0.shape[0], rois0.shape[1], 1)).cuda() * share_pred1
        # share_pred = torch.cat((share_pred0, aaaa), dim=1)
        #
        # # print(share_pred[0, :, 0])
        # # print(gt_boxes[0, :num_boxes.item(), 5])
        #
        # food_index = imdb_s._class_to_ind['food']
        # bbox_index = (gt_boxes[0, :num_boxes.item(), 4] == food_index).nonzero(as_tuple=True)[0]
        #
        # # share_pred[0, bbox_index, 0], gt_boxes[0, bbox_index, 5]
        # # print(share_pred[0, bbox_index, 0])
        # # print(gt_boxes[0, bbox_index, 5])
        #
        # num_food_boxes = len(bbox_index)
        #
        # if num_food_boxes > 0:
        #     # torch.sum(torch.pow(share_pred[0, bbox_index, 0] - gt_boxes[0, bbox_index, 5] * 0.01, 2)).detach()
        #     sum_share_sqr_error += torch.sum(torch.pow(share_pred[0, bbox_index, 0] - gt_boxes[0, bbox_index, 5] * 0.01, 2)).detach()
        #     sum_num_boxes += num_food_boxes
        #
        #     # share classification
        #     share_diff = torch.abs(share_pred[0, bbox_index, 0] - gt_boxes[0, bbox_index, 5] * 0.01)
        #     share_diff = share_diff.cpu().detach().numpy()
        #
        #     share_diff_05 = share_diff < 0.05
        #     share_diff_10 = share_diff < 0.10
        #     share_diff_15 = share_diff < 0.15
        #     share_diff_20 = share_diff < 0.20
        #     share_diff_30 = share_diff < 0.30
        #
        #     prog_m5.update(sum(share_diff_05) / len(share_diff_05), len(share_diff_05))
        #     prog_m10.update(sum(share_diff_10) / len(share_diff_10), len(share_diff_10))
        #     prog_m15.update(sum(share_diff_15) / len(share_diff_15), len(share_diff_15))
        #     prog_m20.update(sum(share_diff_20) / len(share_diff_20), len(share_diff_20))
        #     prog_m30.update(sum(share_diff_30) / len(share_diff_30), len(share_diff_30))
        #
        # if vis or args.save_res_img:
        #     if args.use_progress:
        #         progress_max_index = torch.argmax(progress_pred[0].detach().cpu())
        #         cv2.putText(im2show, '%d-%.1f' % (progress_max_index.item(), progress_pred[0][progress_max_index].detach().cpu().item()),
        #                     (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

        if args.save_res_img:
            path_base, path_filename = os.path.split(imdb.image_path_at(i))
            cv2.imwrite(os.path.join(path_to_result_images, path_filename), im2show)
            print('result image is saved!')

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    # print('progress accuracy: ', top1.get_value())
    # print(top1.get_comp_mat())

    # fid_progress = open(os.path.join(output_dir, 'progress_result.txt'), 'w')
    # # top1_float = top1.get_value()
    # # fid_progress.write('progress accuracy: %.4f\n' % top1_float)
    # # np.savetxt(fid_progress, top1.get_comp_mat(), fmt='%.4f')
    #
    # print('prog classification <0.05: ', prog_m5.get_value())
    # print('prog classification <0.10: ', prog_m10.get_value())
    # print('prog classification <0.15: ', prog_m15.get_value())
    # print('prog classification <0.20: ', prog_m20.get_value())
    # print('prog classification <0.30: ', prog_m30.get_value())
    #
    # fid_progress.write('prog classification <0.05: %.4f\n' % prog_m5.get_value())
    # fid_progress.write('prog classification <0.10: %.4f\n' % prog_m10.get_value())
    # fid_progress.write('prog classification <0.15: %.4f\n' % prog_m15.get_value())
    # fid_progress.write('prog classification <0.20: %.4f\n' % prog_m20.get_value())
    # fid_progress.write('prog classification <0.30: %.4f\n' % prog_m30.get_value())
    #
    # avg_share_sqr_error = sum_share_sqr_error / sum_num_boxes
    # print('share MSE: ', avg_share_sqr_error)
    # print('share MSE sqrt: ', torch.sqrt(avg_share_sqr_error))
    # fid_progress.write('share MSE: %.4f\n' % avg_share_sqr_error)
    # fid_progress.write('share MSE: %.4f\n' % torch.sqrt(avg_share_sqr_error))
    #
    # fid_progress.close()

    end = time.time()
    print("test time: %0.4fs" % (end - start))
