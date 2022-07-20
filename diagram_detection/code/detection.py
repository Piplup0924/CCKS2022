import wandb

import sys
sys.path.insert(0, '../mmdetection')


import torch, torchvision
import mmdet
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

import random
import numpy as np
from pathlib import Path
import copy
import json
from pycocotools.coco import COCO
import os

wandb.login(key="1dd5638ff742d2a11893d11a3b62e9522efd629c")

seed = 42

"""Sets the random seeds."""
set_random_seed(seed, deterministic=False)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

random.seed(seed)

def create_subset(c, cats, valid_n = 3000,test_n=3000):
    new_coco = {}
    cat_ids = c.getCatIds(cats)
    train_img_ids = set()
    valid_img_ids = set()
    test_img_ids = set()
    for cat in cat_ids[::-1]:                                    # 按照检测种类均匀分配图片
        img_ids = copy.copy(c.getImgIds(catIds=[cat]))
        random.shuffle(img_ids)
        tn = max(int(len(img_ids) / 10), 1)
        vn = max(int(len(img_ids) / 10), 1)
        new_test = set(img_ids[:tn])
        new_valid = set(img_ids[tn:tn + vn])
        exist_test_ids = new_test.intersection(train_img_ids)    # 找到交集（有些图片在其他物品特征处理的时候已经分到了train中）
        exist_test_ids_in_valid = new_test.intersection(new_valid)
        a = new_test.intersection(valid_img_ids)
        b = new_test.intersection(test_img_ids)
        exist_valid_ids = new_valid.intersection(train_img_ids)
        d = new_valid.intersection(valid_img_ids)
        e = new_valid.intersection(test_img_ids)
        f = new_valid.intersection(new_test)
        test_ids = new_test.difference(new_valid).difference(train_img_ids).difference(valid_img_ids).difference(test_img_ids) # 挑出new_test中不是交集的部分
        valid_ids = new_valid.difference(new_test).difference(train_img_ids).difference(valid_img_ids).difference(test_img_ids)
        train_ids = set(img_ids).difference(new_valid).difference(new_test).difference(train_img_ids).difference(valid_img_ids).difference(test_img_ids)
#         print(tn, vn, len(img_ids), len(new_test), len(test_ids), len(new_valid), len(valid_ids),len(train_ids))
        print(len(train_ids), len(valid_ids), len(test_ids))
        train_img_ids.update(train_ids)
        test_img_ids.update(test_ids)
        valid_img_ids.update(valid_ids)
#         print(len(test_img_ids))

    # prune duplicates
    dup_test = train_img_ids.intersection(test_img_ids)
    dup_valid = train_img_ids.intersection(valid_img_ids)
    train_img_ids = train_img_ids - dup_test - valid_img_ids
    
    train_anno_ids = set()
    test_anno_ids = set()
    valid_anno_ids = set()
    for cat in cat_ids:
        train_anno_ids.update(c.getAnnIds(imgIds=list(train_img_ids), catIds=[cat]))
        test_anno_ids.update(c.getAnnIds(imgIds=list(test_img_ids), catIds=[cat]))
        valid_anno_ids.update(c.getAnnIds(imgIds=list(valid_img_ids), catIds=[cat]))

    assert len(train_img_ids.intersection(test_img_ids)) == 0, 'img id conflicts with test, {} '.format(train_img_ids.intersection(test_img_ids))
    assert len(train_anno_ids.intersection(test_anno_ids)) == 0, 'anno id conflicts with test'
    assert len(train_img_ids.intersection(valid_img_ids)) == 0, 'img id conflicts with valid, {} '.format(train_img_ids.intersection(valid_img_ids))
    assert len(train_anno_ids.intersection(valid_anno_ids)) == 0, 'anno id conflicts with valid'
    print('train img ids #:', len(train_img_ids), 'train anno #:', len(train_anno_ids))
    print('valid img ids #:', len(valid_img_ids), 'valid anno #:', len(valid_anno_ids))
    print('test img ids #:', len(test_img_ids), 'test anno #:', len(test_anno_ids))
    new_coco_test = copy.deepcopy(new_coco)
    new_coco_valid = copy.deepcopy(new_coco)

    new_coco["images"] = c.loadImgs(list(train_img_ids))
    new_coco["annotations"] = c.loadAnns(list(train_anno_ids))
    
    for ann in new_coco["annotations"]:
        ann.pop('segmentation', None)
    new_coco["categories"] = c.loadCats(cat_ids)
    
    new_coco_valid['images'] = c.loadImgs(list(valid_img_ids))
    new_coco_valid['annotations'] = c.loadAnns(list(valid_anno_ids))
    
    new_coco_test["images"] = c.loadImgs(list(test_img_ids))
    new_coco_test["annotations"] = c.loadAnns(list(test_anno_ids))
    for ann in new_coco_test["annotations"]:
        ann.pop('segmentation', None)
    for ann in new_coco_valid["annotations"]:
        ann.pop('segmentation', None)
    new_coco_test["categories"] = c.loadCats(cat_ids)
    new_coco_valid['categories'] = c.loadCats(cat_ids)
    print('new train split, images:', len(new_coco["images"]), 'annos:', len(new_coco["annotations"]))
    print('new valid split, images:', len(new_coco_valid["images"]), 'annos:', len(new_coco_valid["annotations"]))
    print('new test split, images:', len(new_coco_test["images"]), 'annos:', len(new_coco_test["annotations"]))
    return new_coco, new_coco_valid ,new_coco_test

coco = COCO('./dataset/bbox_coco.json')
nc, nc_valid, nc_test = create_subset(coco, [str(i) for i in range(-100, 101)])

with open('./new_anno/new_train.json', 'w') as f:
    json.dump(nc, f)
with open('./new_anno/new_valid.json', 'w') as f:
    json.dump(nc_valid, f)
with open('./new_anno/new_test.json', 'w') as f:
    json.dump(nc_test, f)

from mmcv import Config

baseline_cfg_path = "./mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py"

cfg = Config.fromfile(baseline_cfg_path)
cfg_focal = Config.fromfile(baseline_cfg_path)


model_name = 'cascade_rcnn_r50_fpn_1x_1'
job = 8

# Folder to store model logs and weight files
job_folder = f'./job{job}_{model_name}'
cfg.work_dir = job_folder

# Change the wandb username and project name below
wnb_username = 'sunstar0708'
wnb_project_name = 'ccks'


# Set seed thus the results are more reproducible
cfg.seed = seed

# You should change this if you use different model
cfg.load_from = ''
if not os.path.exists(job_folder):
    os.makedirs(job_folder)

print("Job folder:", job_folder)

# Set the number of classes
for i in cfg.model.roi_head.bbox_head:
    i.num_classes = 201

# cfg.model.train_cfg.rpn.allowed_border=-1

cfg.gpu_ids = [1]

cfg.runner.max_epochs = 20 # Epochs for the runner that runs the workflow 
cfg.total_epochs = 20

# Learning rate of optimizers. The LR is divided by 8 since the config file is originally for 8 GPUs
cfg.optimizer.lr = 0.02

## Learning rate scheduler config used to register LrUpdater hook
cfg.lr_config = dict(
    policy='CosineAnnealing', # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    by_epoch=False,
    warmup='linear', # The warmup policy, also support `exp` and `constant`.
    warmup_iters=1000, # The number of iterations for warmup
    warmup_ratio=0.001, # The ratio of the starting learning rate used for warmup
    min_lr=1e-07)

# config to register logger hook
cfg.log_config.interval = 40 # Interval to print the log

# Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
cfg.checkpoint_config.interval = 1 # The save interval is 1


cfg.dataset_type = 'CocoDataset' # Dataset type, this will be used to define the dataset
cfg.classes = [str(i) for i in range(-100, 101)]

data_images = './dataset/image'

cfg.data.train.img_prefix = data_images
cfg.data.train.classes = cfg.classes
cfg.data.train.ann_file = './new_anno/new_train.json'
cfg.data.train.type='CocoDataset'

cfg.data.val.img_prefix = data_images
cfg.data.val.classes = cfg.classes
cfg.data.val.ann_file = './new_anno/new_valid.json'
cfg.data.val.type='CocoDataset'

cfg.data.test.img_prefix = data_images
cfg.data.test.classes = cfg.classes
cfg.data.test.ann_file = './new_anno/new_test.json'
cfg.data.test.type='CocoDataset'

cfg.data.samples_per_gpu = 8  # Batch size of a single GPU used in training
cfg.data.workers_per_gpu = 4  # Worker to pre-fetch data for each single GPU
# cfg.data.train_dataloader = dict(samples_per_gpu = 8, workers_per_gpu = 4)
cfg.data.val_dataloader = dict(samples_per_gpu=8, workers_per_gpu=4)
cfg.data.test_dataloader = dict(samples_per_gpu=8, workers_per_gpu=4)

# The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
cfg.evaluation.metric = 'bbox' # Metrics used during evaluation

# Set the epoch intervel to perform evaluation
cfg.evaluation.interval = 1

cfg.evaluation.save_best='bbox_mAP'


cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                        dict(type='WandbLoggerHook',
                             init_kwargs=dict(project=wnb_project_name,
                                              name=f'exp-{model_name}-job{job}',
                                              entity=wnb_username))
                       ]

cfg_path = f'{job_folder}/job{job}_{Path(baseline_cfg_path).name}'
print(cfg_path)

# Save config file for inference later
cfg.dump(cfg_path)
print(f'Config:\n{cfg.pretty_text}')

model = build_detector(cfg.model,
                       train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('valid_cfg'))
# model.init_weights()

datasets = [build_dataset(cfg.data.train)]

train_detector(model, datasets[0], cfg, distributed = False, validate = True)