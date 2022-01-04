# this is a code to train a multi-output model predicting a progress_index
import os
from yo_utils.yo_my_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pathOutput = '/home/yochin/Desktop/DA_Detection_py1_CloudProj/output'

dbSrc = 'CloudStatus_None_inner_90'
dbTgt = 'CloudStatus_None_inner_10'
dbVal = 'CloudStatus_None_inner_90_val'
dbTest = 'CloudStatus_None_inner_10_test'
dbTest2 = 'CloudStatus_None_inner_90_test'

netConvFtr = 'resnet50'

from time import sleep
# hour_delay = 1
# min_delay = 0
# sleep(60*60*hour_delay + 60*min_delay)

# source-trained model
PRETRAINED_PTH = os.path.join(pathOutput, 'frcn-progress-%s' % dbSrc, netConvFtr, 'faster_rcnn_1_8_9999.pth')

prjName = os.path.join(pathOutput, 'frcn-3output-plabel05-%s-mosaic' % dbSrc)
cmd_args = [
    '--dataset', dbSrc,
    '--dataset_t', dbTgt,
    '--net', netConvFtr,
    '--epochs',  '10',
    '--lr_decay_step', '5',
    '--lr', '0.0001',
    '--lr_decay_gamma', '0.1',
    '--save_dir', prjName,
    '--use_pretrained', 'True',
    '--pretrained_path', PRETRAINED_PTH,
    '--prep_type', 'caffe',
    '--use_share_regress',
    '--use_progress',

    # '--ls',
    # '--anchors4',
    # '--ratios5',
    '--manual_seed', '100',


    #
    # '--freeze_base', 'False',
    # '--freeze_base_layers', '0',        # 0~5
    # '--freeze_top_layers', '0',         # 0~2
    # # '--freeze_RPN_layers',            # True or False
    # # '--freeze_final_RPN_layers',        # True or False
    # # '--freeze_final_layers',            # True or False
    # # '--freeze_finalScore_layers',
    # # '--freeze_finalBbox_layers'
    # # '--freeze_btlneck_layers',
    #
    # '--reg_l1_loss', '0.0',
    # '--reg_l1_loss_rpn', '0.0',
    #
    # # '--dim_bottleneck', dim_bottleneck,
    # # '--dim_bottleneck_rpn', dim_bottleneck_rpn,
    #
    # # '--use_labels',
    # # '--use_labels_rpn',
    #
    '--plabel_conf', '0.5',  # trainval_net_v4_plabel_v6.py
    # '--selectedMI',
    # '--w_ent_loss', '0.0',
    # # '--use_div_loss',
    #
    # # '--add_bn_fc8',
    # # '--add_wn_final',
    # # '--add_wn_rpn',
    # # '--apply_branch',
    #
    # '--selection_type', 'all',
    # # '--selection_type', 'fgbg_th',
    # # '--selection_th', '0.9',
    #
    '--use_repository_dataset_at_1epoch',     # use exist annot, jpegimages
    '--use_first_plabel',                       # use the first repository (xml)
    # # '--use_cur_plabel',
    # # '--use_cur_plabel_wo_bbox',
    #
    # # '--use_SADA_plabel_generator_mixture',    # trainval_net_v4_plabel_v7.py
    # # '--use_SADA_plabel_generator',              # trainval_net_v4_plabel_v6 / v8.py
    # # '--n_clusters', '1',
    #
    # # '--ignore_one',   # not in trainval_net_v4_plabel_v5.py
    # '--apply_original',
    #
    '--apply_mosaic',
    '--mosaic_type', 'mosaic',
    # '--mosaic_scale_min', '0.3',
    # '--mosaic_scale_max', '1.0',
    #
    # # '--margin_conf', '0.4',
    # # '--nineMosaic_invert',
    # '--use_rnd_split_pts',
    #
    # # '--apply_mixup',
    # # '--mixup_type', 'bal0mixup',  # 'bal0mosaic', 'bal1mosaic', 'bal2mosaic', 'mosaic'
    # '--beta_param', '1.5',
    #
    # # '--apply_cutout',
    # # '--cutout_type', 'bal0cutout',  # 'bal0mosaic', 'bal1mosaic', 'bal2mosaic', 'mosaic'
    # # '--max_overlap_th', '0.5',
    # # '--ratio_cutout_bbox', '0.5',
    '--use_num_plabel_dataset_insteadof_10000',
    #
    # '--save1e_and_reuseOthers_dataset', # trainval_net_v4_plabel_v5_2.py
    #
    # # '--r', 'True',
    # # '--load_name', os.path.join(prjName, netConvFtr, 'faster_rcnn_1_7_end.pth'),
    '--cuda', '--use_tfb'
]
str_cmd_args = ' '.join(cmd_args)

if not os.path.exists(os.path.join(prjName, netConvFtr)):
    os.makedirs(os.path.join(prjName, netConvFtr))

os.system('%s -W ignore trainval_net_v4_plabel_v5_2_xml.py %s | tee -a "%s/%s/logfile.txt"' % (PATH_PYTHON, str_cmd_args, prjName, netConvFtr))

# dbVal
for i_epoch in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:  #
    pth_filename = os.path.join(prjName, netConvFtr,
                                'faster_rcnn_1_%s_end.pth' % i_epoch)
    cmd_val_args = [
        '--dataset', dbVal,
        '--net', netConvFtr,
        '--path_load_model', pth_filename,
        # '--ls',
        # '--anchors4',
        # '--ratios5',
        # '--dim_bottleneck', dim_bottleneck,
        # '--dim_bottleneck_rpn', dim_bottleneck_rpn,

        # '--add_bn_fc8',
        # '--add_wn_final',
        # '--add_wn_rpn',
        '--prep_type', 'caffe',
        '--use_share_regress',
        '--use_progress',
        # '--save_res_img',
        '--cuda'
    ]
    str_cmd_val_args = ' '.join(cmd_val_args)

    os.system('%s test_net_v4.py %s' % (PATH_PYTHON, str_cmd_val_args))

# test1
for i_epoch in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:  #
    pth_filename = os.path.join(prjName, netConvFtr,
                                'faster_rcnn_1_%s_end.pth' % i_epoch)
    cmd_val_args = [
        '--dataset', dbTest,
        '--net', netConvFtr,
        '--path_load_model', pth_filename,
        # '--ls',
        # '--anchors4',
        # '--ratios5',
        # '--dim_bottleneck', dim_bottleneck,
        # '--dim_bottleneck_rpn', dim_bottleneck_rpn,

        # '--add_bn_fc8',
        # '--add_wn_final',
        # '--add_wn_rpn',
        '--prep_type', 'caffe',
        '--use_share_regress',
        '--use_progress',
        # '--save_res_img',
        '--cuda'
    ]
    str_cmd_val_args = ' '.join(cmd_val_args)

    os.system('%s test_net_v4.py %s' % (PATH_PYTHON, str_cmd_val_args))

# test2
for i_epoch in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:  #
    pth_filename = os.path.join(prjName, netConvFtr,
                                'faster_rcnn_1_%s_end.pth' % i_epoch)
    cmd_val_args = [
        '--dataset', dbTest2,
        '--net', netConvFtr,
        '--path_load_model', pth_filename,
        # '--ls',
        # '--anchors4',
        # '--ratios5',
        # '--dim_bottleneck', dim_bottleneck,
        # '--dim_bottleneck_rpn', dim_bottleneck_rpn,

        # '--add_bn_fc8',
        # '--add_wn_final',
        # '--add_wn_rpn',
        '--prep_type', 'caffe',
        '--use_share_regress',
        '--use_progress',
        # '--save_res_img',
        '--cuda'
    ]
    str_cmd_val_args = ' '.join(cmd_val_args)

    os.system('%s test_net_v4.py %s' % (PATH_PYTHON, str_cmd_val_args))