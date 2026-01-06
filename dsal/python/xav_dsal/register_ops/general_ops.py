import torch
from ..registry import register_op

try:
    get_indice_pairs_2d = torch.ops.xav_dsal.get_indice_pairs_2d
    get_indice_pairs_3d = torch.ops.xav_dsal.get_indice_pairs_3d
    get_indice_pairs_4d = torch.ops.xav_dsal.get_indice_pairs_4d
    register_op("get_indice_pairs_2d", get_indice_pairs_2d, for_mmcv=False)
    register_op("get_indice_pairs_3d", get_indice_pairs_3d, for_mmcv=False)
    register_op("get_indice_pairs_4d", get_indice_pairs_4d, for_mmcv=False)

    # radius
    radius = torch.ops.xav_dsal.radius
    register_op("radius", radius, for_mmcv=True)

    # knn
    knn = torch.ops.xav_dsal.knn
    register_op("knn", knn, for_mmcv=True)

    # torch scatter
    scatter_sum = torch.ops.xav_dsal.scatter_sum
    scatter_mul = torch.ops.xav_dsal.scatter_mul
    scatter_mean = torch.ops.xav_dsal.scatter_mean
    scatter_max = torch.ops.xav_dsal.scatter_max
    scatter_min = torch.ops.xav_dsal.scatter_min
    scatter = torch.ops.xav_dsal.scatter
    register_op("scatter_sum", scatter_sum, for_mmcv=False)
    register_op("scatter_mul", scatter_mul, for_mmcv=False)
    register_op("scatter_mean", scatter_mean, for_mmcv=False)
    register_op("scatter_max", scatter_max, for_mmcv=False)
    register_op("scatter_min", scatter_min, for_mmcv=False)
    register_op("scatter", scatter, for_mmcv=False)

    softmax_focal_loss_forward = torch.ops.xav_dsal.softmax_focal_loss_forward
    softmax_focal_loss_backward = torch.ops.xav_dsal.softmax_focal_loss_backward
    register_op("softmax_focal_loss_forward", softmax_focal_loss_forward, for_mmcv=True)
    register_op("softmax_focal_loss_backward", softmax_focal_loss_backward, for_mmcv=True)

    #
    sigmoid_focal_loss_forward = torch.ops.xav_dsal.sigmoid_focal_loss_forward
    sigmoid_focal_loss_backward = torch.ops.xav_dsal.sigmoid_focal_loss_backward
    register_op("sigmoid_focal_loss_forward", sigmoid_focal_loss_forward, for_mmcv=True)
    register_op("sigmoid_focal_loss_backward", sigmoid_focal_loss_backward, for_mmcv=True)

    # dynamic_scatter
    dynamic_scatter_forward = torch.ops.xav_dsal.dynamic_scatter_forward
    register_op("dynamic_scatter_forward", dynamic_scatter_forward, for_mmcv=False)
    dynamic_scatter_backward = torch.ops.xav_dsal.dynamic_scatter_backward
    register_op("dynamic_scatter_backward", dynamic_scatter_backward, for_mmcv=False)

    # unique_dim
    unique_dim = torch.ops.xav_dsal.unique_dim
    register_op("unique_dim", unique_dim, for_mmcv=False)

except Exception as e:
    raise RuntimeError(f"Failed to load op: {e}")
