#include <torch/library.h>

#include "pybind_ops.h"

PyMODINIT_FUNC PyInit__ext_xpu(void) {
    // No need to do anything.
    return NULL;
}

#if 0
TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("box_iou_rotated", &box_iou_rotated);
    m.impl("hard_voxelize", &hard_voxelize);
    m.impl("dcnv4_forward", &dcnv4_forward);
    m.impl("dcnv4_backward", &dcnv4_backward);
    m.impl("deformable_aggregation_forward", &deformable_aggregation_forward);
    m.impl("deformable_aggregation_backward", &deformable_aggregation_backward);
    m.impl("softmax_focal_loss_forward", &softmax_focal_loss_forward);
    m.impl("softmax_focal_loss_backward", &softmax_focal_loss_backward);
    m.impl("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward);
    m.impl("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward);
    m.impl("bev_pool_v2_forward", &bev_pool_v2_forward);
    m.impl("bev_pool_v2_backward", &bev_pool_v2_backward);
    m.impl("geometric_kernel_attn_cuda_forward", &geometric_kernel_attn_forward);
    m.impl("geometric_kernel_attn_cuda_backward", &geometric_kernel_attn_backward);
    m.impl("ms_deform_attn_forward", &ms_deform_attn_forward);
    m.impl("ms_deform_attn_backward", &ms_deform_attn_backward);
    m.impl("modulated_deform_conv_forward", &modulated_deform_conv_forward);
    m.impl("modulated_deform_conv_backward", &modulated_deform_conv_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def("box_iou_rotated", &box_iou_rotated);
    m.def("hard_voxelize", &hard_voxelize);
    m.def("dcnv4_forward", &dcnv4_forward);
    m.def("dcnv4_backward", &dcnv4_backward);
    m.def("deformable_aggregation_forward", &deformable_aggregation_forward);
    m.def("deformable_aggregation_backward", &deformable_aggregation_backward);
    m.def("softmax_focal_loss_forward", &softmax_focal_loss_forward);
    m.def("softmax_focal_loss_backward", &softmax_focal_loss_backward);
    m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward);
    m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward);
    m.def("bev_pool_v2_forward", &bev_pool_v2_forward);
    m.def("bev_pool_v2_backward", &bev_pool_v2_backward);
    m.def("geometric_kernel_attn_cuda_forward", &geometric_kernel_attn_forward);
    m.def("geometric_kernel_attn_cuda_backward", &geometric_kernel_attn_backward);
    m.def("ms_deform_attn_forward", &ms_deform_attn_forward);
    m.def("ms_deform_attn_backward", &ms_deform_attn_backward);
    m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward);
    m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward);
}
#endif
