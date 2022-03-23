from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes


from tops.config import LazyCall as L
# Inherit from the ssd300 configuration
from .a_ssd300 import (
    train,
    #anchors, # We do not want to inherit the anchors, we want new ones:)
    #backbone,
    loss_objective,
    model,
    optimizer,
    schedulers,
    data_train,
    data_val,
    label_map
)


backbone = L(backbones.BasicModel)(
    output_channels=[256, 512, 256, 256, 128, 128],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

# Low AP for the number 1. I guess because the aspect of the anchors are not tall enogh
anchors = L(AnchorBoxes)(
    feature_sizes=[[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[8, 8], [16, 16], [32, 32], [64, 64], [100, 100], [300, 300]],
    min_sizes=[[20, 20], [60, 60], [111, 111], [162, 162], [213, 213], [264, 264], [315, 315]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2,4,6,8], [2, 3,5,7,9], [2, 3,5,7,9], [2, 3,5,7,9], [2,3,5,7,9], [2,4,6,8]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)