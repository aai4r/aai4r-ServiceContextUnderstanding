from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from munch import munchify
import torch.nn.functional as F

# copy and modified from https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/utils.py

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, opts, scale=0.875, crop_type='RandomCrop',
                 random_hflip=False, random_vflip=False, colorjitter=False,
                 preserve_aspect_ratio=True, rescale_input_size=1.0):
        if type(opts) == dict:
            opts = munchify(opts)

        self.input_size = [opts.input_size[0],
                           int(math.floor(opts.input_size[1] * rescale_input_size)),
                           int(math.floor(opts.input_size[2] * rescale_input_size))]
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        print('input_size: ', self.input_size)

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = crop_type
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width)))

        if crop_type == 'RandomCrop':
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        elif crop_type == 'CenterCrop':
            tfs.append(transforms.CenterCrop(max(self.input_size)))
        elif crop_type == 'TenCrop':
            tfs.append(transforms.TenCrop(max(self.input_size)))
        elif crop_type == 'RandomResizedCrop':
            tfs.append(transforms.RandomResizedCrop(max(self.input_size), scale=(0.8, 1.2)))
        else:
            raise AssertionError('%s is not supported crop_type' % crop_type)

        if colorjitter:
            tfs.append(transforms.ColorJitter(brightness=0.25, contrast=0.25))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        if crop_type == 'TenCrop':
            tfs.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))   # returns a 4D tensor
            tfs.append(transforms.Lambda(lambda crops: torch.stack([ToSpaceBGR(self.input_space == 'BGR')(crop) for crop in crops])))
            tfs.append(transforms.Lambda(lambda crops: torch.stack([ToRange255(max(self.input_range) == 255)(crop) for crop in crops])))
            tfs.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=self.mean, std=self.std)(crop) for crop in crops])))
        else:
            tfs.append(transforms.ToTensor())
            tfs.append(ToSpaceBGR(self.input_space == 'BGR'))
            tfs.append(ToRange255(max(self.input_range) == 255))
            tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        # tfs.append(transforms.Lambda(
        #     lambda crops: print(crops.shape)))  # returns a 4D tensor



        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class LoadTransformImage(object):

    def __init__(self, model, scale=0.875):
        self.load = LoadImage()
        self.tf = TransformImage(model, scale=scale)

    def __call__(self, path_img):
        img = self.load(path_img)
        tensor = self.tf(img)
        return tensor


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# https://github.com/CoinCheung/pytorch-loss/blob/master/label_smooth.py
# version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


# # https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
# class LabelSmoothingLoss(nn.Module):
#     """
#     With label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     """
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
#         # assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         super(LabelSmoothingLoss, self).__init__()
#
#         smoothing_value = label_smoothing / (tgt_vocab_size - 2)
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value)
#         one_hot[self.ignore_index] = 0
#         self.register_buffer('one_hot', one_hot.unsqueeze(0))
#
#         self.confidence = 1.0 - label_smoothing
#
#     def forward(self, output, target):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         model_prob = self.one_hot.repeat(target.size(0), 1)
#         model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
#         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
#
#         return F.kl_div(output, model_prob, reduction='sum')
