# YOLOv6n model with a P2/4 detection head
model = dict(
    type='YOLOv6n_P2',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.25,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True, # required: RepBiFPANNeck_P2 consumes the stride-4 feature
        cspsppf=True,
        ),
    neck=dict(
        type='RepBiFPANNeck_P2',
        # channels_list idx: 5    6    7    8    9   10   11   12   13   14
        out_channels=[      256, 128,  64, 128,  64, 128, 128, 256, 256, 512],
        # Rep_p4, Rep_p3, Rep_p2, Rep_n2, Rep_n3, Rep_n4
        num_repeats=[12, 12, 12, 12, 12, 12],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[64, 128, 256, 512],
        num_layers=4,
        anchors=1,
        strides=[4, 8, 16, 32],
        chx=[8, 10, 12, 14],
        atss_warmup_epoch=0,
        iou_type='siou',
        use_dfl=False,
        reg_max=0 #if use_dfl is False, please set reg_max to 0
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.02,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)
