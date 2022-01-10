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
import pdb
import copy
import xml.etree.ElementTree as ET

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

# detection: 'food', 'dish', 'outlery', 'etc'
# classification: classes in KFood
# food amount: 0.0 ~ 1.0

def save_file_from_list(list_of_list, filename):
    # create xml
    anno = ET.Element('annotation')

    with open(filename, 'w') as fid:
        for item in list_of_list:
            if len(item) == 5:
                classname, x1, y1, x2, y2 = item
            elif len(item) == 7:
                classname, x1, y1, x2, y2, foodname, amount = item

            obj = ET.SubElement(anno, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = classname

            bndbox = ET.SubElement(obj, 'bndbox')

            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(x1)

            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(y1)

            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(x2)

            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(y2)

            if len(item) == 7:
                foodname_tag = ET.SubElement(obj, 'foodname')
                foodname_tag.text = str(foodname)

                amount_tag = ET.SubElement(obj, 'amount')
                amount_tag.text = str(amount)

    ET.ElementTree(anno).write(filename)


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
    parser.add_argument('--use_FPN', dest='use_FPN', action='store_true')
    parser.add_argument('--num_save_images', dest='num_save_images', default=-1, type=int)
    parser.add_argument('--shuffle_db', dest='shuffle_db', default=False, type=bool)
    parser.add_argument('--anchors4', dest='anchors4', action='store_true')
    parser.add_argument('--ratios5', dest='ratios5', action='store_true')
    parser.add_argument('--use_share_regress', dest='use_share_regress', action='store_true')
    parser.add_argument('--use_progress', dest='use_progress', action='store_true')

    parser.add_argument('--prep_type', dest='prep_type', default='caffe', type=str)
    parser.add_argument('--att_type', dest='att_type', help='None, BAM, CBAM', default='None', type=str)

    parser.add_argument('--image_dir', dest='image_dir', default="")
    parser.add_argument('--video_path', dest='video_path', default="")
    parser.add_argument('--webcam_num', dest='webcam_num', default=-1, type=int)
    parser.add_argument('--youtube_url', dest='youtube_url', default="")
    parser.add_argument('--youtube_video_fpm', dest='youtube_video_fpm', default=1, type=int)

    parser.add_argument('--save_result', dest='save_result', action='store_true')
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--result_resized_to_800', dest='result_resized_to_800', action='store_true')

    parser.add_argument('--vis_th', dest='vis_th', default=0.8, type=float)

    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def get_overlap_ratio_meal(food_bbox, dish_bbox):
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
            checkpoint = torch.load(os.path.join(ck_file_folder, 'model_best_{}_{}.pth.tar'.format(net, dbname)),  map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(os.path.join(ck_file_folder, 'model_best_{}_{}.pth.tar'.format(net, dbname)))
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
    path_model_detector = 'output/fpn101_1_10_9999.pth'
    path_to_model_classifier = 'output'

    # fixed, to get class name
    args.dataset = 'CloudStatus_val'
    args.imdb_name2 = 'CloudTableThings_val'
    args.total_imdb_name = 'CloudStatusTableThings_val'
    args.load_name = path_model_detector

    args.use_share_regress = True
    args.use_progress = True

    args.net = 'resnet101'
    args.prep_type = 'caffe'
    args.large_scale = True
    args.shuffle_db = True
    args.cuda = True
    args.dataset_t = ''  # assign dummy naming
    args.save_result = True
    args.use_FPN = True
    args.result_resized_to_800 = True

    # args.image_dir = 'images/OpenImageSimpleCategory'
    # args.image_dir = 'images/MyFoodImages_Kfood'
    # args.image_dir = 'images/MyFoodImages_Food101'
    # args.image_dir = 'images/YM/JPEGImages/45'

    # Load food classifier
    # possible dbname='FoodX251', 'Food101', 'Kfood'
    # possible eval_crop_type='CenterCrop', 'TenCrop'
    food_classifier = FoodClassifier(net='senet154', dbname='Kfood', eval_crop_type='CenterCrop',
                                     ck_file_folder=path_to_model_classifier,
                                     use_cuda=args.cuda, pretrained=False)

    print('Called with args:')

    # args = set_dataset_args(args, test=True)
    # args.imdb_name = args.imdbval_name = "OpenImageSimpleCategory_test"
    # args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

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

    list_box_color = [(0, 0, 0),
                      (0, 0, 200),
                      (0, 200, 0),
                      (200, 200, 0),
                      (200, 0, 200),
                      (0, 200, 200),
                      (200, 0, 200)]

    classes0 = get_class_list(args.dataset)
    classes1 = get_class_list(args.imdb_name2)
    classes_total = get_class_list(args.total_imdb_name)

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
    elif args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
        total_num_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS)
        pathOutputSaveImages = os.path.join(pathOutputSave, 'result', args.video_path.split('/')[-1])

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

    print('\n Result images are stored in %s' % pathOutputSaveImages)

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
        elif args.youtube_url != '' or args.video_path != '':
            if not cap.isOpened():
                raise RuntimeError("Youtube could not open. Please check connection.")

            skip_seconds = int(60 / args.youtube_video_fpm)

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
        # im = im[1080:, 384:-384, :]


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

        rois0, cls_prob0, bbox_pred0, _, _, _, _, _, share_pred0, _, progress_pred0, _ = \
            fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=0)
        rois1, cls_prob1, bbox_pred1, _, _, _, _, _, share_pred1, _, _, _ = \
            fasterRCNN(im_data, im_info, gt_boxes, num_boxes, flow_id=1)

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

        results = []

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

        h, w = im.shape[:2]
        if max(h, w) > 800 and args.result_resized_to_800:
            if h > 800:
                im2show = cv2.resize(im, (int(800 / h * w), 800))
            if w > 800:
                im2show = cv2.resize(im, (800, int(800 / w * h)))

            h_display = im2show.shape[0]
            im_scale = h_display / h
        else:
            im2show = np.copy(im)
            im_scale = 1.0

        im2show_wobbox = copy.copy(im2show)

        im_pil = torchvision.transforms.ToPILImage(mode=None)(im[:,:,::-1])
        im_width, im_height = im_pil.size

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

                # im: original image, (768, 1024, 3)
                # im_data: blob image, (1, 3, 600, 800)
                # cls_dets: x1, y1, x2, y2, score

                # crop and feed to classifier
                # im_pil.save(osp.join(pathOutputSaveImages, 'debug_input.png'))
                if classes_total[j] == 'food':
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

                        bbox_draw = cls_dets.detach().cpu().numpy()[k:k + 1, :]
                        bbox_draw[:, :4] = bbox_draw[:, :4] * im_scale

                        if bbox_draw[0, 4] >= args.vis_th:
                            results.append([classes_total[j],
                                            int(bbox_draw[0, 0]), int(bbox_draw[0, 1]), int(bbox_draw[0, 2]), int(bbox_draw[0, 3]),
                                            food_class[0], bbox_draw[0, 5]])
                else:
                    bbox_draw = cls_dets.detach().cpu().numpy()
                    bbox_draw[:, :4] = bbox_draw[:, :4] * im_scale

                    for k in range(cls_dets.shape[0]):
                        if bbox_draw[k, 4] >= args.vis_th:
                            results.append(
                                [classes_total[j],
                                 int(bbox_draw[k, 0]), int(bbox_draw[k, 1]), int(bbox_draw[k, 2]), int(bbox_draw[k, 3]),
                                 0, 0])

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

        # # dish-food converter
        # # every dish find the food and its amount
        # # if food is not found, zero amount is assigned.
        # new_results = []
        # for item in results:
        #     class_name, x1, y1, x2, y2, food_name, food_amount = item
        #
        #     if class_name == 'dish':
        #         new_results.append(item)
        #
        # for item in results:
        #     class_name, x1, y1, x2, y2, food_name, food_amount = item
        #
        #     if class_name == 'food':
        #         for dish_i, dish_item in enumerate(new_results):
        #             _, d_x1, d_y1, d_x2, d_y2, _, dish_amount = dish_item
        #
        #             # check overlap
        #             overlap_ratio = get_overlap_ratio_meal(food_bbox=[x1, y1, x2, y2],
        #                                                    dish_bbox=[d_x1, d_y1, d_x2, d_y2])
        #             if overlap_ratio > 0.9:
        #                 new_results[dish_i][5] = food_name
        #                 new_results[dish_i][6] += food_amount
        #
        # for dish_i, dish_item in enumerate(new_results):
        #     # class_name, x1, y1, x2, y2, food_name, food_amount = item
        #     new_results[dish_i][0] = 'food'
        #     new_results[dish_i][5] = 'food'
        #
        #     new_amount = new_results[dish_i][6]
        #     if new_amount > 1.0: new_amount = 1.0
        #     if new_amount < 0.0: new_amount = 0.0
        #     new_results[dish_i][6] = int(round(new_amount * 100))
        #
        # results = new_results
        # # dish-food converter - end

        # print('---- ---- ----')
        # print(results)

        if args.vis or args.save_result:
            for item in results:
                # item = [category, x1, y1, x2, y2, (food_name), (amount)]
                if item[0] == 'food':
                    str_name = '%s (%s, %.2f)' % (item[0], item[5], item[6])
                else:
                    str_name = '%s' % (item[0])

                bbox_draw = np.array([[item[1], item[2], item[3], item[4], 1.0]])

                color_index = np.where(classes_total == item[0])[0][0]
                im2show = vis_detections_korean_ext2(im2show, str_name, bbox_draw,
                                                            box_color=list_box_color[color_index], text_color=(255, 255, 255),
                                                            text_bg_color=list_box_color[color_index], fontsize=20,
                                                            thresh=args.vis_th,
                                                            draw_score=False, draw_text_out_of_box=True)

        if args.save_result:
            if args.image_dir != '':
                result_path = os.path.join(pathOutputSaveImages, imglist[cur_index_image][:-4] + "_det.jpg")
                result_path_wobbox = os.path.join(pathOutputSaveImages, imglist[cur_index_image][:-4] + "_det_wobbox.jpg")
                result_path_txt = os.path.join(pathOutputSaveImages,
                                                  imglist[cur_index_image][:-4] + "_det_wobbox.xml")
            else:
                result_path = os.path.join(pathOutputSaveImages, "%05d_det.jpg" % cur_index_image)
                result_path_wobbox = os.path.join(pathOutputSaveImages, "%05d_det_wobbox.jpg" % cur_index_image)
                result_path_txt = os.path.join(pathOutputSaveImages, "%05d_det_wobbox.xml" % cur_index_image)

            cv2.imwrite(result_path, im2show)
            cv2.imwrite(result_path_wobbox, im2show_wobbox)

            save_file_from_list(results, result_path_txt)

        if args.vis:
            # im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            im2showRGB = im2show
            cv2.imshow("frame", im2showRGB)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()

    print('Results are stored in %s' % pathOutputSaveImages)


