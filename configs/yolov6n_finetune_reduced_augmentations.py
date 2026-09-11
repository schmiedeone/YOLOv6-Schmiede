# YOLOv6s model
model = dict(
    type='YOLOv6n',
    pretrained='weights/yolov6n.pt',
    depth_multiple=0.33,
    width_multiple=0.25,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,
        cspsppf=True,
    ),
    neck=dict(
        type='RepBiFPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
    ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=3,
        anchors_init=[[10, 13, 19, 19, 33, 23],
                      [30, 61, 59, 59, 59, 119],
                      [116, 90, 185, 185, 373, 326]],
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,  # set to True if you want to further train with distillation
        reg_max=0,  # set to 16 if you want to further train with distillation
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.0032,
    lrf=0.12,
    momentum=0.843,
    weight_decay=0.00036,
    warmup_epochs=2.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.05
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.1,
    degrees=30.0,
    translate=0.1,
    scale=0.4,
    shear=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.25,
    mixup=0.0,
)
