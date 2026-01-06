#include "xpytorch.hpp"

void box_iou_rotated(
        const at::Tensor& boxes1,
        const at::Tensor& boxes2,
        at::Tensor& ious,
        const int64_t mode,
        const bool aligned) {
    auto num_boxes1 = boxes1.size(0);
    auto num_boxes2 = boxes2.size(0);
    AT_ASSERTM(boxes1.size(1) == 5, "boxes1 number of columns should be 5");
    AT_ASSERTM(mode == 0 || mode == 1, "mode should be either 0 or 1");
    AT_ASSERTM(num_boxes1 < 2048 && num_boxes1 >= 0, "the data size of num_boxes1 is not currently supported.");
    AT_ASSERTM(num_boxes2 < 2048 && num_boxes2 >= 0, "the data size of num_boxes2 is not currently supported.");
    AT_ASSERTM(!aligned || (num_boxes1 == num_boxes2), "nBoxANum and nBoxBNum mismatch in aligned mode.");

    auto ctx = xmlir_rt::getXpuKernelContext();
    auto kernel = xav::cpu::box_iou_rotated<float>;
    if (boxes1.device().is_cuda()) {
        kernel = xav::xpu::box_iou_rotated<float>;
    }

    kernel(ctx,
           num_boxes1,
           num_boxes2,
           boxes1.data_ptr<float>(),
           boxes2.data_ptr<float>(),
           ious.data_ptr<float>(),
           mode,
           aligned);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("box_iou_rotated", &box_iou_rotated);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("box_iou_rotated", &box_iou_rotated);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "box_iou_rotated(Tensor boxes1, Tensor boxes2, Tensor(a!) ious, int mode, bool aligned) -> ()"));
}
