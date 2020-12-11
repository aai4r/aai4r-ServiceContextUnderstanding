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
from model.utils.net_utils import save_net, load_net, vis_detections_korean_ext2
from model.utils.blob import im_list_to_blob
from model.utils.parser_func import set_dataset_args
import pdb
import os.path as osp

# for classification
import network
import pretrained_utils_v2 as utils
import torchvision
import pickle

# for socket
import socket
import concurrent.futures
import threading

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


# Received from Moon
import socket
import cv2
import numpy as np

normal = 0
abnormal = 0

# HOST = '192.168.0.2'
HOST = '129.254.82.234'
PORT = 9999


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

    # for SimCLR
    parser.add_argument('--prep_type', dest='prep_type', default='caffe', type=str)

    parser.add_argument('--image_dir', dest='image_dir',
                        default="")

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

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
        self.model.cuda()

        self.model.eval()

    def classify(self, image):
        output = self.model(image)

        return output


if __name__ == '__main__':
    args = parse_args()

    # detection: 'Food', 'Tableware', 'Drink'
    # classification: KFood or FoodX251, Food101
    # eval_crop_type: 'TenCrop' or 'CenterCrop'

    args.dataset = 'OpenImageSimpleCategory_test'
    args.load_name = 'output/frcn-OpenImageSimpleCategory/resnet50/resnet50/faster_rcnn_1_7_9999.pth'
    args.net = 'resnet50'
    args.prep_type = 'caffe'
    args.cuda = True
    args.dataset_t = ''  # assign dummy naming

    # Load food classifier
    # possible dbname='FoodX251', 'Food101', 'Kfood'
    # possible eval_crop_type='CenterCrop', 'TenCrop'
    net_clf = 'senet154' # 'senet154'
    dbname = 'Kfood'
    food_classifier = FoodClassifier(net=net_clf, dbname=dbname, eval_crop_type='TenCrop',
                                     ck_file='output/baseline-%s-torchZR/%s/%s/model_best.pth.tar' % (dbname, net_clf, net_clf))
    pascal_classes = np.asarray(['__background__',  # always index 0
                    '음식',
                    '식기',
                    '음료'])
    thresh = [0., 0.9, 0.5, 0.7]    # good for image
    vis = True

    list_box_color = [(0, 0, 0),
                      (0, 0, 200),
                      (0, 200, 0),
                      (200, 0, 0)]
    SAVE_IMAGE_AS_RESIZED = False

    print('Called with args:')

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
    if not os.path.exists(args.load_name):
        raise Exception('There is no input directory for loading network from ' + args.load_name)

    # initilize the network here.
    from model.faster_rcnn.vgg16 import vgg16
    from model.faster_rcnn.resnet import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, use_pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'resnet101':
        fasterRCNN = resnet(pascal_classes, num_layers=101, use_pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'resnet50':
        fasterRCNN = resnet(pascal_classes, num_layers=50, use_pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'resnet152':
        fasterRCNN = resnet(pascal_classes, num_layers=152, use_pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (args.load_name))
    if args.cuda > 0:
        checkpoint = torch.load(args.load_name)
    else:
        checkpoint = torch.load(args.load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print("load model from checkpoint %s successfully!" % (args.load_name))

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

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    # TCP
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))

    # 서버가 클라이언트의 접속을 허용하도록 합니다.
    server_socket.listen()

    # accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴합니다.
    client_socket, addr = server_socket.accept()

    # 접속한 클라이언트의 주소입니다.
    print('Connected by', addr)

    # opencv: bgr is assumed

    while True:
        # Get image from the webcam
        print('in main: start')
        cap_tic = time.time()
        # CheckBox 상태 수신
        dataLengthStr = client_socket.recv(8).decode()
        dataLengthInt = int(dataLengthStr.rstrip('\x00'))
        dataByte = client_socket.recv(dataLengthInt)
        dataByteString = dataByte.decode('utf-8')
        print(dataByteString)

        # JPEG 이미지 크기를 수신합니다.
        dataLengthStr = client_socket.recv(8).decode()
        dataLengthInt = int(dataLengthStr.rstrip('\x00'))
        dataLengthIntRead = 0
        dataByteSum = b""

        # Flir의 JPEG 이미지 크기만큼 data를 수신합니다.
        # JPEG 이미지 크기만큼의 data가 들어오리라는 보장이 없기 때문에 체크가 필요해 while 루프를 돌게 됩니다.
        while True:
            dataByte = client_socket.recv(dataLengthInt - dataLengthIntRead)
            dataByteSum += dataByte
            print('read data : ', len(dataByte))
            print('dataByteSum : ', len(dataByteSum))
            # 빈 문자열을 수신하면 루프를 중지합니다.
            if not dataByte:
                break

            # byte 수 만큼 다 읽었으나 알 수 없는 이유로 JPEG이미지를 다 수신받지 못함. 따라서 추가적인 Read 필요
            if len(dataByteSum) >= dataLengthInt and dataByteSum[-2:] != b'\xff\xd9':
                print('image byte curruption', dataByteSum[-2:])
                while True:
                    dataByte = client_socket.recv(1)
                    print(dataByte)
                    dataByteSum += dataByte
                    if dataByteSum[-2:] == b'\xff\xd9':
                        break

            # byte 수 만큼 다 읽지 못함. 더 Read할 byte가 남았음
            elif len(dataByteSum) != dataLengthInt and dataByteSum[-2:] != b'\xff\xd9':
                dataLengthIntRead = len(dataByte)
                continue

            # 제대로 수신이 완료되었다면 앞의 케이스를 다 뛰어넘음
            # print(dataByteSum)
            dataByteDecoded = cv2.imdecode(np.frombuffer(dataByteSum, np.uint8), -1)
            if dataByteString[-1] == '0':
                print(dataByteString[-1])
                dataByteProcessed = cv2.rotate(dataByteDecoded, cv2.ROTATE_180)
            else:
                print(dataByteString[-1])
                dataByteProcessed = cv2.flip(dataByteDecoded, cv2.ROTATE_180)

            client_socket.send(("Image Transfer Complete" + str(normal) + "\n").encode())
            normal += 1
            break

        cap_toc = time.time()
        '''
        부가적인 코드 들어갈 공간, img = dataByteDecoded // 640 x 480
        '''
        # dataByteProcessed
        # cv2.imshow('dataByte imshow', dataByteProcessed)
        # cv2.waitKey(1)
        client_cmd = dataByteString

        if client_cmd is not None:
            proc_tic = time.time()

            # bgr
            # cv2.imwrite('./yochin_debug_server_image.jpg', im_bgr)
            im = dataByteProcessed

            # im2show = np.copy(im)
            im2show = im

            if SAVE_IMAGE_AS_RESIZED:
                im2show = cv2.resize(im2show,
                                     (int(im2show.shape[1] * im_scales[0]), int(im2show.shape[0] * im_scales[0])))

            if client_cmd[0] == '1':
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

                if client_cmd[1] == '1':  # food
                    im_pil = torchvision.transforms.ToPILImage(mode=None)(im[:,:,::-1])     # bgr to rgb
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
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]

                        # crop and feed to classifier
                        # im_pil.save(osp.join(pathOutputSaveImages, 'debug_input.png'))
                        if j == 1 and client_cmd[1] == '1':  # food
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
                                # im_crop.save(osp.join('.', 'debug_crop.png'))       # rgb

                                # im_crop.show()

                                with torch.no_grad():
                                    im_crop = food_classifier.test_transform(im_crop)

                                im_crop = torch.unsqueeze(im_crop, dim=0)

                                if food_classifier.eval_crop_type == 'TenCrop':
                                    bs, ncrops, c, h, w = im_crop.size()
                                    im_crop = im_crop.view(-1, c, h, w)

                                food_output = food_classifier.classify(im_crop.cuda())
                                food_output = food_output.cpu()

                                if food_classifier.eval_crop_type == 'TenCrop':
                                    food_output = food_output.view(bs, ncrops, -1).mean(1)  # avg over crops

                                topk_score, topk_index = torch.topk(food_output, 5, dim=1)

                                food_class = [food_classifier.idx_to_class[topk_index[0][l].item()] for l in range(5)]
                                food_score = torch.nn.functional.softmax(topk_score[0], dim=0)

                                # food_class = ['밥']
                                # food_score = [0.9]

                                if vis:
                                    bbox_draw = cls_dets.cpu().numpy()[k:k + 1, :]

                                    if SAVE_IMAGE_AS_RESIZED:
                                        bbox_draw[:, :4] = bbox_draw[:, :4] * im_scales[0]

                                    # class_name_w_food = '%s (%s: %.2f)'%(pascal_classes[j], food_class[0], food_score[0].item())
                                    class_name_w_food = '%s (%s, %s, %s)'%(pascal_classes[j], food_class[0], food_class[1], food_class[2])
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


            # visualize
            cv2.imshow("frame", im2show)    # bgr
            proc_toc = time.time()
            cap_time = cap_toc - cap_tic
            proc_time = proc_toc - proc_tic
            total_time = cap_time + proc_time
            frame_rate = 1.0 / total_time
            print('Frame rate:', frame_rate)
            print('\t\t\t capturing: ', cap_time)
            print('\t\t\t proc: ', proc_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    # 소켓을 닫고 화면을 제거합니다.
    client_socket.close()
    server_socket.close()
