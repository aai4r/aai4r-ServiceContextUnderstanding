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
import os.path as osp
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable

# from scipy.misc import imread   # scipy.__version__ < 1.2.0
from imageio import imread  # new
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

import pdb


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

# detection: 'Food', 'Tableware', 'Drink'
# classification: KFood or others

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
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
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--num_save_images', dest='num_save_images', default=-1, type=int)
    parser.add_argument('--shuffle_db', dest='shuffle_db', default=False, type=bool)
    parser.add_argument('--anchors4', dest='anchors4', action='store_true')
    parser.add_argument('--ratios5', dest='ratios5', action='store_true')

    parser.add_argument('--prep_type', dest='prep_type', default='caffe', type=str)

    parser.add_argument('--image_dir', dest='image_dir', default="sample_images")

    args = parser.parse_args()
    return args


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


class FoodClassifier:
    # eval_crop_type: 'TenCrop' or 'CenterCrop'
    def __init__(self, net, dbname, eval_crop_type, ck_file):
        self.eval_crop_type = eval_crop_type
        # load class info
        path_class_to_idx = 'output/class_info_%s.pkl' % dbname
        if os.path.exists(path_class_to_idx):
            fid = open(path_class_to_idx, 'rb')
            self.class_to_idx = pickle.load(fid)
            fid.close()
        else:
            if dbname == 'Food101':
                ts_db_path = 'data/Cloud_PublicDB/[Food]Food101/food-101/organized/test'  # 250 / food
            elif dbname == 'FoodX251':
                ts_db_path = 'data/Cloud_PublicDB/[Food]FoodX251/organized/val_set'
            elif dbname == 'Kfood':
                ts_db_path = 'data/Cloud_PublicDB/[Food]Kfood/organized/test'
            else:
                raise AssertionError('%s dbname is not supported' % dbname)

            temp = torchvision.datasets.ImageFolder(root=ts_db_path)
            self.class_to_idx = temp.class_to_idx

            fid = open(path_class_to_idx, 'wb')
            pickle.dump(self.class_to_idx, fid)
            fid.close()

        self.idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())

        # create model
        self.model = network.pret_torch_nets(model_name=net, pretrained=True, class_num=len(self.class_to_idx))
        self.test_transform = utils.TransformImage(self.model.model, crop_type=eval_crop_type,
                                          rescale_input_size=1.0)
        checkpoint = torch.load(ck_file)
        self.model.load_state_dict(checkpoint['state_dict'])

    def classify(self, image):
        self.model.eval()
        output = self.model(image)

        return output


if __name__ == '__main__':
    args = parse_args()

    # # for yochin
    # path_model_detector = 'output/frcn-OpenImageSimpleCategory/resnet50/resnet50/faster_rcnn_1_7_9999.pth'
    # path_to_model_classifier = 'output/baseline-Kfood-torchZR/senet154/senet154/model_best.pth.tar'

    # for github
    path_model_detector = 'output/faster_rcnn_1_7_9999.pth'
    path_to_model_classifier = 'output/model_best.pth.tar'

    args.dataset = 'OpenImageSimpleCategory_test'
    args.load_name = path_model_detector

    args.net = 'resnet50'
    args.prep_type = 'caffe'
    args.num_save_images = -1
    args.shuffle_db = True
    # args.image_dir = 'images/OpenImageSimpleCategory'
    # args.image_dir = 'images/MyFoodImages_Kfood'
    # args.image_dir = 'images/MyFoodImages_Food101'
    args.cuda = True
    args.dataset_t = ''  # assign dummy naming

    SAVE_IMAGE_AS_RESIZED = True

    # Load food classifier
    # possible dbname='FoodX251', 'Food101', 'Kfood'
    # possible eval_crop_type='CenterCrop', 'TenCrop'
    food_classifier = FoodClassifier(net='senet154', dbname='Kfood', eval_crop_type='CenterCrop',
                                     ck_file=path_to_model_classifier)

    print('Called with args:')

    # args = set_dataset_args(args, test=True)
    args.imdb_name = args.imdbval_name = "OpenImageSimpleCategory_test"
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

    if args.large_scale:
        args.cfg_file = "cfgs/{}_ls.yml".format(args.net)
    elif args.small_scale:
        args.cfg_file = "cfgs/{}_ss.yml".format(args.net)
    else:
        args.cfg_file = "cfgs/{}.yml".format(args.net)

    if args.anchors4 == True and args.ratios5 == True:
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25,0.5,1,2,4]', 'MAX_NUM_GT_BOXES', '30']
    elif args.anchors4 == True and args.ratios5 == False:
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.anchors4 == False and args.ratios5 == True:
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.25,0.5,1,2,4]', 'MAX_NUM_GT_BOXES', '30']
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

    if args.dataset == 'OpenImageSimpleCategory_test' or args.dataset == 'OpenImageSimpleCategory_validation':
        pascal_classes = np.asarray(['__background__',  # always index 0
                        '음식',
                        '식기',
                        '음료'])
        thresh = [0., 0.9, 0.5, 0.7]
        vis = True

        list_box_color = [(0, 0, 0),
                          (0, 0, 200),
                          (0, 200, 0),
                          (200, 0, 0)]
    else:
        raise AssertionError('check args.dataset')

    if args.image_dir == '':
        args.image_dir = os.path.join('images', args.dataset, 'JPEGImages')
        print(args.image_dir + ' dataset is loaded for analysis')

    pathOutputSave = load_name
    pathOutputSave = '/'.join(os.path.split(pathOutputSave)[:-1])
    pathOutputSave = pathOutputSave + '/' + args.dataset
    pathOutputSaveImages = os.path.join(pathOutputSave, 'det_cls_images')
    if not os.path.exists(pathOutputSaveImages):
        print('make a directory for save images: %s' % pathOutputSaveImages)
        os.makedirs(pathOutputSaveImages)

    # initilize the network here.
    from model.faster_rcnn.vgg16 import vgg16
    from model.faster_rcnn.resnet import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, class_agnostic=args.class_agnostic)
    elif args.net == 'resnet101':
        fasterRCNN = resnet(pascal_classes, use_pretrained=False, num_layers=101, class_agnostic=args.class_agnostic)
    elif args.net == 'resnet50':
        fasterRCNN = resnet(pascal_classes, use_pretrained=False, num_layers=50, class_agnostic=args.class_agnostic)
    elif args.net == 'resnet152':
        fasterRCNN = resnet(pascal_classes, use_pretrained=False, num_layers=152, class_agnostic=args.class_agnostic)
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
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100

    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    else:
        imglist = os.listdir(args.image_dir)
        num_images = len(imglist)

        if args.shuffle_db == True:
            import random as rnd

            rnd.seed(17)
            rnd.shuffle(imglist)

        if args.num_save_images == -1:
            print('args.num_save_images(=roidb_s): ', num_images)
        else:
            num_images = args.num_save_images
            print('args.num_save_images: ', num_images)

    print('Loaded Photo: {} images.'.format(num_images))

    total_num_images = num_images

    while (num_images >= 0):
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1

        if num_images % 100 == 0:
            print('%d/%d - images remained' % (num_images, total_num_images))

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[num_images])
            try:
                im_in = np.array(imread(im_file))       # imread is reading in 'rgb' order
            except:
                raise AssertionError('%s is corrupted' % im_file)

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

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

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

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
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)

            if SAVE_IMAGE_AS_RESIZED:
                im2show = cv2.resize(im2show, (int(im2show.shape[1] * im_scales[0]), int(im2show.shape[0] * im_scales[0])))

        im_pil = torchvision.transforms.ToPILImage(mode=None)(im[:,:,::-1])
        im_width, im_height = im_pil.size

        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh[j]).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
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

                        im_crop = food_classifier.test_transform(im_crop)
                        im_crop = torch.unsqueeze(im_crop, dim=0)

                        if food_classifier.eval_crop_type == 'TenCrop':
                            bs, ncrops, c, h, w = im_crop.size()
                            im_crop = im_crop.view(-1, c, h, w)

                        food_output = food_classifier.classify(im_crop)

                        if food_classifier.eval_crop_type == 'TenCrop':
                            food_output = food_output.view(bs, ncrops, -1).mean(1)  # avg over crops

                        topk_score, topk_index = torch.topk(food_output, 5, dim=1)

                        food_class = [food_classifier.idx_to_class[topk_index[0][l].item()] for l in range(5)]
                        food_score = torch.nn.functional.softmax(topk_score[0], dim=0)

                        # print(food_class, food_score)

                        if vis:
                            bbox_draw = cls_dets.cpu().numpy()[k:k + 1, :]

                            if SAVE_IMAGE_AS_RESIZED:
                                bbox_draw[:, :4] = bbox_draw[:, :4] * im_scales[0]

                            # class_name_w_food = '%s (%s: %.2f)'%(pascal_classes[j], food_class[0], food_score[0].item())
                            class_name_w_food = '%s (%s)'%(pascal_classes[j], food_class[0])
                            im2show = vis_detections_korean_ext2(im2show, class_name_w_food, bbox_draw,
                                                            box_color=list_box_color[j], text_color=(255, 255, 255),
                                                                 text_bg_color=list_box_color[j], fontsize=20, thresh=thresh[j],
                                                                 draw_score=False, draw_text_out_of_box=True)
                else:
                    if vis:
                        bbox_draw = cls_dets.cpu().numpy()

                        if SAVE_IMAGE_AS_RESIZED:
                            bbox_draw[:, :4] = bbox_draw[:, :4] * im_scales[0]

                        im2show = vis_detections_korean_ext2(im2show, pascal_classes[j], bbox_draw,
                                                            box_color=list_box_color[j], text_color=(255, 255, 255),
                                                             text_bg_color=list_box_color[j], fontsize=20, thresh=thresh[j],
                                                             draw_score=False, draw_text_out_of_box=True)



        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()

        if vis and webcam_num == -1:
            result_path = osp.join(pathOutputSaveImages, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        else:
            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()

    print('Results are stored in %s' % pathOutputSaveImages)