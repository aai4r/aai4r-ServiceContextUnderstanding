import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
# print('add_path: ', lib_path)

coco_path = osp.join(this_dir, 'data', 'coco', 'PythonAPI')
add_path(coco_path)
# print('add_path: ', coco_path)

# sys.path.remove('/home/yochin/Desktop/DA_Detection_py1/lib')

# print('in sys.path:', )
# for p in sys.path:
#     print(p)

