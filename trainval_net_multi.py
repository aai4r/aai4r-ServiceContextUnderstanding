# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.utils.parser_func import set_dataset_args

import cv2

TENSOR_BGR_MEANS = torch.from_numpy(cfg.PIXEL_MEANS).float().cuda().permute(2, 0, 1)

def convert_2images(image_output, add_pixel_mean=False, renom_multiply_255=False):
    save_image = image_output.squeeze()       # (3, H, W), BGR order

    if renom_multiply_255:
        # -1.0 ~ 1.0 is input range
        save_image = (save_image * 0.5) + 0.5       # 0.0 ~ 1.0
        save_image = save_image * 255.              # 0.0 ~ 255.0

    if add_pixel_mean:
        save_image = save_image + TENSOR_BGR_MEANS

    save_image_np = save_image.permute(1, 2, 0).cpu().detach().numpy()
    save_image_np = save_image_np.astype(np.uint8)

    return save_image_np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--imdb_name2', dest='imdb_name2',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path and filename for loading the model', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
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
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    parser.add_argument('--use_FPN', dest='use_FPN', action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--manual_seed', dest='manual_seed', default=0, type=int)

    parser.add_argument('--anchors4', dest='anchors4', action='store_true')
    parser.add_argument('--ratios5', dest='ratios5', action='store_true')
    parser.add_argument('--use_share_regress', dest='use_share_regress', action='store_true')
    parser.add_argument('--use_progress', dest='use_progress', action='store_true')

    parser.add_argument('--pretrained_path', dest='pretrained_path',
                        help='path to load pretrained models', default="",
                        type=str)
    parser.add_argument('--prep_type', dest='prep_type', default='caffe', type=str)

    parser.add_argument('--att_type', dest='att_type', help='None, BAM, CBAM', default='None', type=str)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    args.dataset_t = ''   # assign dummy naming
    args = set_dataset_args(args)
    print(args)

    if args.att_type == 'None':
        args.att_type = None

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if args.manual_seed:
        print('use_manual_seed: ', args.manual_seed)
        # set manual seed
        args.manual_seed = np.uint32(args.manual_seed)
        np.random.seed(args.manual_seed)
        # random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # logger.info('Random Seed: {}'.format(int(args.manual_seed)))
        args.random_seed = args.manual_seed  # save seed into args
    else:
        print('use_default_seed')
        np.random.seed(cfg.RNG_SEED)
        # random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        # if you are suing GPU
        # torch.cuda.manual_seed(cfg.RNG_SEED)
        # torch.cuda.manual_seed_all(cfg.RNG_SEED)

        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    # dataset-1
    print('args.imdb_name: ', args.imdb_name)
    imdb1, roidb1, ratio_list1, ratio_index1 = combined_roidb(args.imdb_name)
    train_size1 = len(roidb1)     # this is not firmed.
    sampler_batch1 = sampler(train_size1, args.batch_size)
    dataset1 = roibatchLoader(roidb1, ratio_list1, ratio_index1, args.batch_size, \
                             imdb1.num_classes, training=True, prep_type=args.prep_type,
                             share_return=True, progress_return=True
                             )
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size,
                                             sampler=sampler_batch1, num_workers=args.num_workers)
    print('{:d} roidb1 entries'.format(len(roidb1)))

    # dataset-2
    print('args.imdb_name2: ', args.imdb_name2)
    imdb2, roidb2, ratio_list2, ratio_index2 = combined_roidb(args.imdb_name2)
    train_size2 = len(roidb2)  # this is not firmed.
    sampler_batch2 = sampler(train_size2, args.batch_size)
    dataset2 = roibatchLoader(roidb2, ratio_list2, ratio_index2, args.batch_size, \
                             imdb2.num_classes, training=True, prep_type=args.prep_type,
                             share_return=False
                             )
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size,
                                              sampler=sampler_batch2, num_workers=args.num_workers)
    print('{:d} roidb2 entries'.format(len(roidb2)))

    # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    output_dir = args.save_dir + "/" + args.net
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'images_det')):
        os.makedirs(os.path.join(output_dir, 'images_det'))

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

    # initilize the network here.
    # freeze_base, pretrained_path, num_layers=101
    if args.use_FPN:
        from model.fpn.resnet_multi_CBAM import resnet
    else:
        from model.faster_rcnn.resnet_multi import resnet
    if args.net == 'resnet101':
        fasterRCNN = resnet(imdb1.classes, imdb2.classes, use_pretrained=True,
                            pretrained_path=args.pretrained_path, num_layers=101,
                            class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress, use_progress=args.use_progress, att_type=args.att_type)
    elif args.net == 'resnet50':
        fasterRCNN = resnet(imdb1.classes, imdb2.classes, use_pretrained=True,
                            pretrained_path=args.pretrained_path, num_layers=50,
                            class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress, use_progress=args.use_progress, att_type=args.att_type)
    elif args.net == 'resnet152':
        fasterRCNN = resnet(imdb1.classes, imdb2.classes, use_pretrained=True,
                            pretrained_path=args.pretrained_path, num_layers=152,
                            class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress, use_progress=args.use_progress, att_type=args.att_type)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        # load_name = os.path.join(output_dir,
        #                          'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        load_name = args.load_name
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    # iters_per_epoch = int(train_size / args.batch_size)       # original faster-rcnn-pytorch-1.0
    iters_per_epoch = int(10000 / args.batch_size)          # modified in DA_Detection
    # iters_per_epoch = int(300 / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter(os.path.join(output_dir, 'logs'))
        print('log is saved in %s'%os.path.join(output_dir, 'logs'))

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter1 = iter(dataloader1)
        data_iter2 = iter(dataloader2)
        for step in range(iters_per_epoch):
            # dataset1
            try:
                data = next(data_iter1)
            except:
                data_iter1 = iter(dataloader1)
                data = next(data_iter1)

            # data[0] [1, C3, H600, W800]
            # data[1] scaling info
            # data[2] bbox
            # data[3] num_bbox
            # data[4] gt_share
            # data[5] gt_progress

            with torch.no_grad():
                # data_pt[0]: image [1, 3, 600/H, 1200/W]
                # data_pt[1]: W, H, resized_ratio
                # data_pt[2]: bbox [1, 20, 5], 5 has x1, y1, x2, y2, class_index
                # data_pt[3]: num_bboxes
                # data_pt[4]: [0] path_to_images, data_pt[4][0].split('/')[-1]
                # data_pt[5]: share_gt
                # data_pt[6]: progress_index
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
                gt_progress.resize_(data[6].size()).copy_(data[6])

            if step < 20:
                result_path = os.path.join(output_dir, 'images_det', "debug_dataset1_e%d_s%05d.jpg" % (epoch, step))
                cv_im_data_swap = convert_2images(im_data, add_pixel_mean=True)

                for kkk in range(num_boxes):
                    cv2.rectangle(cv_im_data_swap, (int(gt_boxes[0, kkk, 0]), int(gt_boxes[0, kkk, 1])),
                                                   (int(gt_boxes[0, kkk, 2]), int(gt_boxes[0, kkk, 3])), (0,0,255))
                    class_name = imdb1.classes[int(gt_boxes[0, kkk, 4])]
                    amount = float(gt_boxes[0, kkk, 5])
                    cv2.putText(cv_im_data_swap, '%s-%.1f' % (class_name, amount),
                                (int(gt_boxes[0, kkk, 0]), int(gt_boxes[0, kkk, 1]) - 15), cv2.FONT_HERSHEY_PLAIN,
                                2.0, (0, 0, 255), thickness=2)

                cv2.putText(cv_im_data_swap, '%s' % (gt_progress.item()),
                            (1, 22), cv2.FONT_HERSHEY_PLAIN,
                            2.0, (0, 0, 255), thickness=2)

                cv2.imwrite(result_path, cv_im_data_swap)
                # pdb.set_trace()     # check input image and bbox
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, \
            share_pred, share_loss, \
            progress_pred, progress_loss = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=0,
                                                      gt_progress=gt_progress)
            # print('gt_progress: ', gt_progress)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                   + 10. * share_loss.mean() + progress_loss.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 5.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    loss_share = share_loss.mean().item()
                    loss_progress = progress_loss.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    loss_share = share_loss.item()
                    loss_progress = progress_loss.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[dataset0]")
                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                print("\t\t\tloss_share: %.4f, loss_progress: %.4f" \
                      % (10. * loss_share, loss_progress))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                        'loss_share': 10. * loss_share,
                        'loss_progress': loss_progress,
                    }
                    logger.add_scalars("losses_dataset0", info, (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()


            #dataset2
            try:
                data = next(data_iter2)
            except:
                data_iter2 = iter(dataloader2)
                data = next(data_iter2)

            # data[0] [1, C3, H600, W800]
            # data[1] scaling info
            # data[2] bbox
            # data[3] num_bbox
            # data[4] gt_share

            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
                # gt_share.resize_(data[4].size()).copy_(data[4])

            if step < 20:
                result_path = os.path.join(output_dir, 'images_det', "debug_dataset2_e%d_s%05d.jpg" % (epoch, step))
                cv_im_data_swap = convert_2images(im_data, add_pixel_mean=True)

                for kkk in range(num_boxes):
                    cv2.rectangle(cv_im_data_swap, (int(gt_boxes[0, kkk, 0]), int(gt_boxes[0, kkk, 1])),
                                                   (int(gt_boxes[0, kkk, 2]), int(gt_boxes[0, kkk, 3])), (0,0,255))
                    class_name = imdb2.classes[int(gt_boxes[0, kkk, 4])]
                    cv2.putText(cv_im_data_swap, '%s' % (class_name), (int(gt_boxes[0, kkk, 0]), int(gt_boxes[0, kkk, 1]) - 15), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 255), thickness=1)

                cv2.imwrite(result_path, cv_im_data_swap)
                # pdb.set_trace()     # check input image and bbox

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, \
            _, _, \
            _, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=1)
            # print(share_pred_dummy, share_loss_dummy)       # always 0

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 5.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[dataset1]")
                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                    }
                    logger.add_scalars("losses_dataset1", info, (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
