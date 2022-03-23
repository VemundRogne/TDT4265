from tops.config import LazyCall as L
# The line belows inherits the configuration set for the tdt4265 dataset
from .tdt4265 import (
    train,
    anchors,
    backbone,
    loss_objective,
    model,
    optimizer,
    schedulers,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

anchors.min_sizes = [[15, 15], [60, 60], [111, 111], [162, 162], [213, 213], [264, 264], [315, 315]],
