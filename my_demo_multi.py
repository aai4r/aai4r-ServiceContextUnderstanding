# --------------------------------------------------------
# Tensorflow Faster R-CNN
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

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread   # scipy.__version__ < 1.2.0
from imageio import imread  # new
from roi_data_layer.roidb import combined_roidb
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections_korean_ext2_wShare, vis_detections_korean_ext2
from model.utils.blob import im_list_to_blob
from model.utils.parser_func import set_dataset_args
import pdb

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
    parser.add_argument('--imdb_name2', dest='imdb_name2',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--total_imdb_name', dest='total_imdb_name',
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
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="/srv/share/jyang375/models")
    parser.add_argument('--load_name', dest='load_name',
                        help='directory to load models',
                        default="/srv/share/jyang375/models")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--ss', dest='small_scale',
                        help='whether use large imag scale',
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
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--num_save_images', dest='num_save_images', default=-1, type=int)
    parser.add_argument('--shuffle_db', dest='shuffle_db', default=False, type=bool)
    parser.add_argument('--anchors4', dest='anchors4', action='store_true')
    parser.add_argument('--ratios5', dest='ratios5', action='store_true')
    parser.add_argument('--use_share_regress', dest='use_share_regress', action='store_true')

    # for SimCLR
    parser.add_argument('--prep_type', dest='prep_type', default='caffe', type=str)

    parser.add_argument('--image_dir', dest='image_dir', default="")
    parser.add_argument('--webcam_num', dest='webcam_num', default=-1, type=int)
    parser.add_argument('--youtube_url', dest='youtube_url', default="")
    parser.add_argument('--youtube_fpm', dest='youtube_fpm', default=1, type=int)

    parser.add_argument('--save_result', dest='save_result', action='store_true')
    parser.add_argument('--vis', dest='vis', action='store_true')

    parser.add_argument('--vis_th', dest='vis_th', default=0.8, type=float)

    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def get_class_list(dataset_name):
    if dataset_name == 'clipart' or dataset_name == 'pascal_voc_0712':
        pascal_classes = np.asarray(['__background__',
                                     'aeroplane', 'bicycle', 'bird', 'boat',
                                     'bottle', 'bus', 'car', 'cat', 'chair',
                                     'cow', 'diningtable', 'dog', 'horse',
                                     'motorbike', 'person', 'pottedplant',
                                     'sheep', 'sofa', 'train', 'tvmonitor'])
    elif dataset_name == 'cityscape' or dataset_name == 'foggy_cityscape':
        pascal_classes = np.asarray(['__background__',  # always index 0
                                     'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck'])
    elif dataset_name == 'cityscape_car' or dataset_name == 'kitti_car':
        pascal_classes = np.asarray(['__background__',  # always index 0
                                     'car'])
    elif dataset_name == 'pascal_voc_water' or dataset_name == 'water':
        pascal_classes = np.asarray(['__background__',  # always index 0
                                     'bicycle', 'bird', 'car', 'cat', 'dog', 'person'])
    elif dataset_name == 'bdd100k_night' or dataset_name == 'bdd100k_daytime':
        pascal_classes = np.asarray(["__background__",  # always index 0
                                     "bike", "bus", "car", "motor", "person", "rider", "traffic light", "traffic sign",
                                     "train", "truck"])
    elif dataset_name == 'OpenImageSimpleCategory_test' or dataset_name == 'OpenImageSimpleCategory_validation':
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

def _get_image_blob(im):
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


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    args.dataset_t = ''  # assign dummy naming
    args = set_dataset_args(args, test=True)
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_name
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = args.load_name

    classes0 = get_class_list(args.dataset)
    classes1 = get_class_list(args.imdb_name2)
    classes_total = get_class_list(args.total_imdb_name)

    # initilize the network here.
    if args.net == 'vgg16':
        from model.faster_rcnn.vgg16 import vgg16
        fasterRCNN = vgg16(pascal_classes, use_pretrained=False, class_agnostic=args.class_agnostic)
    elif 'resnet' in args.net:
        from model.faster_rcnn.resnet_multi import resnet
        if args.net == 'resnet101':
            fasterRCNN = resnet(classes0, classes1, num_layers=101, use_pretrained=False, class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress)
        elif args.net == 'resnet50':
            fasterRCNN = resnet(classes0, classes1, num_layers=50, use_pretrained=False, class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress)
        elif args.net == 'resnet152':
            fasterRCNN = resnet(classes0, classes1, num_layers=152, use_pretrained=False, class_agnostic=args.class_agnostic, use_share_regress=args.use_share_regress)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05

    total_num_images = 0

    pathOutputSave = load_name
    pathOutputSave = '/'.join(os.path.split(pathOutputSave)[:-1])
    # pathOutputSave = pathOutputSave + '/' + args.dataset

    if args.image_dir != '':
        imglist = os.listdir(args.image_dir)
        total_num_images = len(imglist)

        pathOutputSaveImages = os.path.join(pathOutputSave, 'result', 'images', args.image_dir.split()[-1])

        if args.shuffle_db:
            import random as rnd

            rnd.seed(17)
            rnd.shuffle(imglist)

        if args.num_save_images == -1:
            print('args.num_save_images(=roidb_s): ', total_num_images)
        else:
            total_num_images = args.num_save_images
            print('args.num_save_images: ', total_num_images)

    elif args.webcam_num >= 0:
        cap = cv2.VideoCapture(args.webcam_num)
        total_num_images = np.inf

        pathOutputSaveImages = os.path.join(pathOutputSave, 'result', 'webcam')

    elif args.youtube_url != '':
        import pafy

        url = args.youtube_url
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)
        total_num_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS)
        pathOutputSaveImages = os.path.join(pathOutputSave, 'result', url.split('/')[-1])

    else:
        raise AssertionError('at least one input should be selected!')


    if not os.path.exists(pathOutputSaveImages):
        print('make a directory for save images: %s' % pathOutputSaveImages)
        os.makedirs(pathOutputSaveImages)

    print('Loaded Photo: {} images.'.format(total_num_images))

    for cur_index_image in range(total_num_images):
        total_tic = time.time()
        if cur_index_image % 100 == 0:
            print('%d/%d - images remained' % (cur_index_image, total_num_images))

        # Get image from the webcam
        if args.webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()     # read bgr (default)
            im_in = np.array(frame)
        elif args.youtube_url != '':
            if not cap.isOpened():
                raise RuntimeError("Youtube could not open. Please check connection.")

            skip_seconds = int(60 / args.youtube_fpm)

            # cap.set(cv2.CAP_PROP_POS_FRAMES, int(cur_index_image * fps * skip_seconds))       # cannot get correct frame because of high compressed video and key frame tech.
            cap.set(cv2.CAP_PROP_POS_MSEC, int(cur_index_image * skip_seconds * 1000))

            ret, frame = cap.read()     # read bgr (default)

            if ret is False:
                print('Youtube is ended!')
                break

            if frame is None:
                print('frame %d (index %d) is None, skip it' % (int(cur_index_image * skip_seconds * 1000), cur_index_image))
                continue


            im_in = np.array(frame)     # [H, W, C]

            cur_index_image_from_cap = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[cur_index_image])
            # im = cv2.imread(im_file)
            im_in = np.array(imread(im_file))   # read rgb

            # rgb -> bgr
            im_in = im_in[:, :, ::-1]



        if len(im_in.shape) == 2:
            raise AssertionError('Input has two dimension, not three!')
            # im_in = im_in[:, :, np.newaxis]
            # im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        im = im_in  # bgr

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        rois0, cls_prob0, bbox_pred0, _, _, _, _, _, share_pred0, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=0)
        rois1, cls_prob1, bbox_pred1, _, _, _, _, _, share_pred1, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=1)


        # rois0 # [1, 300, 5]
        rois = torch.cat((rois0, rois1), dim=1)

        # share_pred0 # [1, 300, 1]
        aaaa = torch.ones((share_pred0.shape[0], rois0.shape[1], 1)).cuda() * share_pred1
        share_pred = torch.cat((share_pred0, aaaa), dim=1)

        # cls_prob0 # [1, 300, 3]
        # bbox_pred0 # [1, 300, 12 (3x4)

        cls_prob = torch.zeros((cls_prob0.shape[0],
                                cls_prob0.shape[1]+cls_prob1.shape[1],
                                len(classes_total))).cuda()
        bbox_pred = torch.zeros((bbox_pred0.shape[0],
                                 bbox_pred0.shape[1] + bbox_pred1.shape[1],
                                 4*len(classes_total))).cuda()

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

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(classes_total))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        if args.use_share_regress:
            share_pred = share_pred.squeeze()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if args.vis or args.save_result:
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
        for j in xrange(1, len(classes_total)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)    # find index with scores > threshold in j-class
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

                if args.use_share_regress:
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), share_pred_inds.unsqueeze(1)), 1)
                else:
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if args.vis or args.save_result:
                    cls_dets[:, :4] = cls_dets[:, :4] * im_scale
                    if args.use_share_regress and classes_total[j] == 'food':
                        im2show = vis_detections_korean_ext2_wShare(im2show, classes_total[j],
                                                                    cls_dets.cpu().detach().numpy(), draw_score=False, thresh=args.vis_th)
                    else:
                        im2show = vis_detections_korean_ext2(im2show, classes_total[j],
                                                                    cls_dets.cpu().detach().numpy(), draw_score=False, thresh=args.vis_th)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if args.webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(cur_index_image, total_num_images, detect_time, nms_time))
            sys.stdout.flush()

        total_toc = time.time()
        total_time = total_toc - total_tic
        frame_rate = 1 / total_time
        print('Frame rate:', frame_rate)

        if args.save_result:
            if args.image_dir != '':
                result_path = os.path.join(pathOutputSaveImages, imglist[cur_index_image][:-4] + "_det.jpg")
            else:
                result_path = os.path.join(pathOutputSaveImages, "%05d_det.jpg" % cur_index_image)
            cv2.imwrite(result_path, im2show)

        if args.vis:
            # im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            im2showRGB = im2show
            cv2.imshow("frame", im2showRGB)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()
