# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from datasets.OpenImageSimpleCategory import OpenImageSimpleCategory
from datasets.CloudTableThings import CloudTableThings
from datasets.CloudTableThingsFineClass import CloudTableThingsFineClass
from datasets.CloudStatus import CloudStatus
from datasets.CloudStatus_al import CloudStatus_al
from datasets.plabel import plabel
from datasets.YMTestBed import YMTestBed
from datasets.YMTestBed_al import YMTestBed_al

import numpy as np

for split in ['train', 'validation', 'test']:
    name = 'OpenImageSimpleCategory_{}'.format(split)
    __sets[name] = (lambda split=split: OpenImageSimpleCategory(split))

for split in ['train', 'val', 'trainval']:
    name = 'CloudTableThings_{}'.format(split)
    __sets[name] = (lambda split=split: CloudTableThings(split))

for split in ['train', 'val', 'trainval']:
    name = 'CloudTableThingsFineClass_{}'.format(split)
    __sets[name] = (lambda split=split: CloudTableThingsFineClass(split))

for split in ['train', 'val', 'trainval', 'test']:
    for subsplit in ['90', '45', '10', 'inner', 'outer', 'window']:
        name = 'CloudStatus_{}_{}_None_None'.format(subsplit, split)
        __sets[name] = (lambda split=split, subsplit=subsplit: CloudStatus(split, subsplit, None, None))

for split in ['train', 'val', 'trainval', 'test']:
    for angle in ['90', '45', '10']:
        for loc in ['inner', 'outer', 'window']:
            name = 'CloudStatus_None_{}_{}_{}'.format(loc, angle, split)
            __sets[name] = (lambda split=split, loc=loc, angle=angle: CloudStatus(split, None, loc, angle))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'CloudStatus_None_None_None_{}'.format(split)
    __sets[name] = (lambda split=split: CloudStatus(split, None, None, None))

for split in ['test', 'list', 'listwoErr', 'listwoErrPart1', 'listwoErrPart1Short', 'listwoErrPart2', 'listwoErrPart2Short']:
    for angle in ['90', '45', '10', 'all']:
        name = 'YMTestBed_{}_{}'.format(split, angle)
        __sets[name] = (lambda split=split, angle=angle: YMTestBed(split, angle))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if 'plabel' in name:
        # decode name
        # name = 'plabel_[cityscape_train_normal-foggy_cityscape_train_foggy]_base0_e1_oneSADAk1_0.5'
        db_part, infos = name.split(']')

        _, db_pairs = db_part.split('[')
        db_src, db_tgt = db_pairs.split('-')

        items = infos.split('_')
        baseNet = '_'.join(items[1:-3])
        epoch = items[-3]
        mosaic_type = items[-2]
        th = items[-1]

        print('#### check plabel_{}_{}_{}_{}_{}_{} ####'.format(db_src, db_tgt, baseNet, epoch, mosaic_type, th))

        return plabel(db_src, db_tgt, baseNet, epoch, mosaic_type, th)
    elif 'CloudStatus_al' in name:
        print('#### check {} list ####'.format(name))

        return CloudStatus_al(name)
    elif 'YMTestBed_al' in name:
        print('#### check {} list ####'.format(name))

        return YMTestBed_al(name)
    else:
        if name not in __sets:
            raise KeyError('Unknown dataset: {}'.format(name))
        return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
