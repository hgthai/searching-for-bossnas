import copy
_base_ = 'base.py'
# model settings
model = dict(
    type='SiameseSupernetsHyTraPP',
    pretrained=None,
    base_momentum=0.99,
    masking_ratio=0.3,
    patch_size=16,
    num_sampled_subnets=2,
    pre_conv=True,
    backbone=dict(
        type='SupernetHyTra',
    ),
    start_block=0,
    num_block=4,
    neck=dict(
        type='NonLinearNeckSimCLRProject',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        sync_bn=False,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckSimCLR',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, num_layers=2, sync_bn=False,
                             with_bias=True, with_last_bn=False, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client',
    return_label=False)
data_train_list = '../data/imagenet/meta/train.txt'
data_train_root = '../data/imagenet/train'
train_dataset_type = 'MultiAugBYOLDataset'
test_dataset_type = 'StoragedBYOLDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline1[4]['p'] = 0.1 # gaussian blur
train_pipeline2[5]['p'] = 0.2 # solarization

data_test_list = '../data/imagenet/meta/val.txt'
data_test_root = '../data/imagenet/val'

test_pipeline1 = copy.deepcopy(train_pipeline1)
test_pipeline2 = copy.deepcopy(train_pipeline2)

data = dict(
    imgs_per_gpu=16,  # total 16*1(gpu)*8(interval)=128
    workers_per_gpu=0,
    train=dict(
        type=train_dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
        num_pairs=2),
    val=dict(
        type=test_dataset_type,
        data_source=dict(list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline1=test_pipeline1, pipeline2=test_pipeline2, prefetch=prefetch,
    )
)
# optimizer
optimizer = dict(type='LARS', lr=0.15, weight_decay=0.000001, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})
# apex
use_fp16 = False
# interval for accumulate gradient
update_interval = 8
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=6, save_optimizer=False)
# runtime settings
total_epochs = 4
load_from = '../checkpoint.pth'
resume_from = None
# additional hooks
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval),
    dict(type='RandomPathHook'),
    dict(
        type='ValBestPathHookPP',
        dataset=data['val'],
        bn_dataset=data['train'],
        initial=True,
        interval=1,
        optimizer_cfg=optimizer,
        lr_cfg=lr_config,
        imgs_per_gpu=64,
        workers_per_gpu=0,
        epoch_per_stage=1,
        resume_best_path='',  # e.g. 'path_rank/bestpath_2.yml'
        madds_constraint=3.4e9,
        initial_lambda=0.1,
        soft_margin_beta=2.0,
        soft_margin_alpha=1e-9,
        init_delta=0.0,
        soft_margin_lr=1e-3,
        num_generations=50,
        topk_update=10,
        topk_log=3,
        masking_ratio=0.3,
        patch_size=16,
        supernet_checkpoint='../checkpoint.pth')
]
# resume_from = 'checkpoints/stage3_epoch3.pth'
# resume_optimizer = False
cudnn_benchmark = True
