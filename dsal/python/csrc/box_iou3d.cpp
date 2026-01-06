#include "xpytorch.hpp"
#include <ATen/ATen.h>

void iou3d_boxes_overlap_bev_forward(const at::Tensor& boxes_a, const at::Tensor& boxes_b, at::Tensor& overlaps_bev)
{
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM(boxes_a.dim() == 2, "boxes_a must be 2 dim");
    AT_ASSERTM(boxes_b.dim() == 2, "boxes_b must be 2 dim");
    AT_ASSERTM(overlaps_bev.dim() == 2, "overlaps_bev must be 2 dim");

    AT_ASSERTM(boxes_a.size(1) == 7, "boxes_a.size(1) must = 7, which is [x, y, z, dx, dy, dz, heading]");
    AT_ASSERTM(boxes_b.size(1) == 7, "boxes_b.size(1) must = 7, which is [x, y, z, dx, dy, dz, heading]");

    AT_ASSERTM(boxes_a.size(0) == overlaps_bev.size(0), "boxes_a.size(0) must == overlaps_bev.size(0)");
    AT_ASSERTM(boxes_b.size(0) == overlaps_bev.size(1), "boxes_b.size(0) must == overlaps_bev.size(1)");

    int N = boxes_a.size(0);
    int M = boxes_b.size(0);

    int ret = 0;
    if (boxes_a.device().is_cuda()) {
        ret = xav::xpu::iou3d_boxes_overlap_bev_forward(
            ctx, 
            N, 
            boxes_a.data_ptr<float>(), 
            M, 
            boxes_b.data_ptr<float>(), 
            overlaps_bev.data_ptr<float>()
        );
    } else {
        ret = xav::cpu::iou3d_boxes_overlap_bev_forward(
            ctx, 
            N, 
            boxes_a.data_ptr<float>(), 
            M, 
            boxes_b.data_ptr<float>(), 
            overlaps_bev.data_ptr<float>()
        );
    }
    assert(ret == 0);
}

int gather_keep_from_mask_cpu(
    const int boxes_num, 
    const uint32_t *mask,
    const int BIT_LEN,
    int *keep_data)
{
    int num_to_keep = 0;
    const int col_blocks = (boxes_num + BIT_LEN - 1) / BIT_LEN;
    uint32_t removed[col_blocks];
    memset(removed, 0, col_blocks * sizeof(uint32_t));
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / BIT_LEN;
        int inblock = i % BIT_LEN;
        if (!(removed[nblock] & (1u << inblock))) {
            keep_data[num_to_keep++] = i;
            const uint32_t *p = mask + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                removed[j] |= p[j];
            }
        }
    }
    return num_to_keep;
}

void iou3d_nms3d_forward(
    const at::Tensor& boxes, 
    at::Tensor& keep, 
    at::Tensor& keep_num, 
    double nms_overlap_thresh)
{
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM(boxes.dim() == 2, "boxes must be 2 dim");
    AT_ASSERTM(boxes.size(1) == 7, "boxes.size(1) must be 7");
    AT_ASSERTM(boxes.size(0) == keep.size(0), "boxes.size(0) must == keep.size(0)");
    TORCH_CHECK(boxes.is_contiguous(), " must be contiguous");
    TORCH_CHECK(keep.is_contiguous(), " must be contiguous");
    int boxes_num = boxes.size(0);
    AT_ASSERTM(boxes_num < 2097152, "boxes_num mask must be full load in SM");
    const int BIT_LEN = 32;
    const int col_blocks = (boxes_num + BIT_LEN - 1) / BIT_LEN;
    at::Tensor mask = at::zeros({boxes_num, col_blocks}, boxes.options().dtype(at::kInt));

    int ret = 0;
    if (boxes.device().is_cuda()) {
        ret = xav::xpu::iou3d_nms3d_forward(
            ctx,
            boxes_num,
            nms_overlap_thresh,
            boxes.data_ptr<float>(),
            (uint32_t*)mask.data_ptr<int32_t>()
        );

        at::Tensor keep_t = at::zeros({boxes_num}, boxes.options().dtype(at::kBool));
        ret = xav::xpu::gather_keep_from_mask(
            ctx,
            boxes_num,
            (uint32_t*)mask.data_ptr<int32_t>(),
            keep_t.data_ptr<bool>()
        );
        auto keep_data = keep_t.nonzero().index({at::indexing::Slice(), 0});
        keep_num.fill_(at::Scalar(keep_data.size(0)));
        keep.index_put_({at::indexing::Slice(0, keep_data.size(0))}, keep_data);
    } else {
        ret = xav::cpu::iou3d_nms3d_forward(
            ctx,
            boxes_num,
            nms_overlap_thresh,
            boxes.data_ptr<float>(),
            (uint32_t*)mask.data_ptr<int32_t>()
        );

        int64_t* keep_data = keep.data_ptr<int64_t>();
        uint32_t* mask_data = (uint32_t*)mask.data_ptr<int32_t>();
        uint32_t removed[col_blocks];
        memset(removed, 0, col_blocks * sizeof(uint32_t));
        int64_t num_to_keep = 0;
        for (int i = 0; i < boxes_num; i++) {
            int nblock = i / BIT_LEN;
            int inblock = i % BIT_LEN;
            if (!(removed[nblock] & (1u << inblock))) {
                keep_data[num_to_keep++] = i;
                const uint32_t *p = mask_data + i * col_blocks;
                for (int j = nblock; j < col_blocks; j++) {
                    removed[j] |= p[j];
                }
            }
        } 
        keep_num.fill_(at::Scalar(num_to_keep));
    }
}

void iou3d_nms3d_normal_forward(
    const at::Tensor& boxes, 
    at::Tensor& keep, 
    at::Tensor& keep_num, 
    double nms_overlap_thresh)
{
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM(boxes.dim() == 2, "boxes must be 2 dim");
    AT_ASSERTM(boxes.size(1) == 7, "boxes.size(1) must be 7");
    AT_ASSERTM(boxes.size(0) == keep.size(0), "boxes.size(0) must == keep.size(0)");
    TORCH_CHECK(boxes.is_contiguous(), " must be contiguous");
    TORCH_CHECK(keep.is_contiguous(), " must be contiguous");
    int boxes_num = boxes.size(0);
    AT_ASSERTM(boxes_num < 2097152, "boxes_num mask must be full load in SM");
    const int BIT_LEN = 32;
    const int col_blocks = (boxes_num + BIT_LEN - 1) / BIT_LEN;
    at::Tensor mask = at::zeros({boxes_num, col_blocks}, boxes.options().dtype(at::kInt));

    int ret = 0;
    if (boxes.device().is_cuda()) {
        ret = xav::xpu::iou3d_nms3d_normal_forward(
            ctx,
            boxes_num,
            nms_overlap_thresh,
            boxes.data_ptr<float>(),
            (uint32_t*)mask.data_ptr<int32_t>()
        );

        at::Tensor keep_t = at::zeros({boxes_num}, boxes.options().dtype(at::kBool));
        ret = xav::xpu::gather_keep_from_mask(
            ctx,
            boxes_num,
            (uint32_t*)mask.data_ptr<int32_t>(),
            keep_t.data_ptr<bool>()
        );

        auto keep_data = keep_t.nonzero().index({at::indexing::Slice(), 0});
        keep_num.fill_(at::Scalar(keep_data.size(0)));
        keep.index_put_({at::indexing::Slice(0, keep_data.size(0))}, keep_data);
    } else {
        ret = xav::cpu::iou3d_nms3d_normal_forward(
            ctx,
            boxes_num,
            nms_overlap_thresh,
            boxes.data_ptr<float>(),
            (uint32_t*)mask.data_ptr<int32_t>()
        );

        int64_t* keep_data = keep.data_ptr<int64_t>();
        uint32_t* mask_data = (uint32_t*)mask.data_ptr<int32_t>();
        uint32_t removed[col_blocks];
        memset(removed, 0, col_blocks * sizeof(uint32_t));
        int64_t num_to_keep = 0;
        for (int i = 0; i < boxes_num; i++) {
            int nblock = i / BIT_LEN;
            int inblock = i % BIT_LEN;
            if (!(removed[nblock] & (1u << inblock))) {
                keep_data[num_to_keep++] = i;
                const uint32_t *p = mask_data + i * col_blocks;
                for (int j = nblock; j < col_blocks; j++) {
                    removed[j] |= p[j];
                }
            }
        } 
        keep_num.fill_(at::Scalar(num_to_keep));
    }
}

void boxes_iou_bev_gpu(const at::Tensor& boxes_a, const at::Tensor& boxes_b, at::Tensor& overlaps_bev)
{
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM(boxes_a.dim() == 2, "boxes_a must be 2 dim");
    AT_ASSERTM(boxes_b.dim() == 2, "boxes_b must be 2 dim");
    AT_ASSERTM(overlaps_bev.dim() == 2, "overlaps_bev must be 2 dim");

    AT_ASSERTM(boxes_a.size(1) == 7, "boxes_a.size(1) must = 7, which is [x, y, z, dx, dy, dz, heading]");
    AT_ASSERTM(boxes_b.size(1) == 7, "boxes_b.size(1) must = 7, which is [x, y, z, dx, dy, dz, heading]");

    AT_ASSERTM(boxes_a.size(0) == overlaps_bev.size(0), "boxes_a.size(0) must == overlaps_bev.size(0)");
    AT_ASSERTM(boxes_b.size(0) == overlaps_bev.size(1), "boxes_b.size(0) must == overlaps_bev.size(1)");

    int N = boxes_a.size(0);
    int M = boxes_b.size(0);

    int ret = 0;
    if (boxes_a.device().is_cuda()) {
        ret = xav::xpu::boxes_iou_bev_kernel(
            ctx, 
            N, 
            boxes_a.data_ptr<float>(), 
            M, 
            boxes_b.data_ptr<float>(), 
            overlaps_bev.data_ptr<float>()
        );
    } else {
        ret = xav::cpu::boxes_iou_bev_kernel(
            ctx, 
            N, 
            boxes_a.data_ptr<float>(), 
            M, 
            boxes_b.data_ptr<float>(), 
            overlaps_bev.data_ptr<float>()
        );
    }
    assert(ret == 0);
}

void paired_boxes_overlap_bev_gpu(const at::Tensor& boxes_a, const at::Tensor& boxes_b, at::Tensor& overlaps_bev) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM(boxes_a.dim() == 2, "boxes_a must be 2 dim");
    AT_ASSERTM(boxes_b.dim() == 2, "boxes_b must be 2 dim");
    AT_ASSERTM(overlaps_bev.dim() == 2, "overlaps_bev must be 2 dim");

    AT_ASSERTM(boxes_a.size(1) == 7, "boxes_a.size(1) must = 7, which is [x, y, z, dx, dy, dz, heading]");
    AT_ASSERTM(boxes_b.size(1) == 7, "boxes_b.size(1) must = 7, which is [x, y, z, dx, dy, dz, heading]");

    AT_ASSERTM(boxes_a.size(0) == overlaps_bev.size(0), "boxes_a.size(0) must == overlaps_bev.size(0)");
    AT_ASSERTM(boxes_b.size(0) == overlaps_bev.size(0), "boxes_b.size(0) must == overlaps_bev.size(1)");

    int N = boxes_a.size(0);
    int M = boxes_b.size(0);
    AT_ASSERTM(N == M, "For paired_boxes_overlap, boxes_a.size(0) must equal boxes_b.size(0)");

    int ret = 0;
    if (boxes_a.device().is_cuda()) {
        ret = xav::xpu::paired_boxes_overlap_kernel(
            ctx, 
            N, 
            boxes_a.data_ptr<float>(), 
            M, 
            boxes_b.data_ptr<float>(), 
            overlaps_bev.data_ptr<float>()
        );
    } else {
        ret = xav::cpu::paired_boxes_overlap_kernel(
            ctx, 
            N, 
            boxes_a.data_ptr<float>(), 
            M, 
            boxes_b.data_ptr<float>(), 
            overlaps_bev.data_ptr<float>()
        );
    }
    assert(ret == 0);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("iou3d_boxes_overlap_bev_forward", &iou3d_boxes_overlap_bev_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("iou3d_nms3d_forward", &iou3d_nms3d_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("iou3d_nms3d_normal_forward", &iou3d_nms3d_normal_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("boxes_iou_bev_gpu", &boxes_iou_bev_gpu);
}
TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("paired_boxes_overlap_bev_gpu", &paired_boxes_overlap_bev_gpu);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("iou3d_boxes_overlap_bev_forward", &iou3d_boxes_overlap_bev_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("iou3d_nms3d_forward", &iou3d_nms3d_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("iou3d_nms3d_normal_forward", &iou3d_nms3d_normal_forward);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("boxes_iou_bev_gpu", &boxes_iou_bev_gpu);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("paired_boxes_overlap_bev_gpu", &paired_boxes_overlap_bev_gpu);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("iou3d_boxes_overlap_bev_forward(Tensor boxes_a, Tensor boxes_b, "
                                "Tensor(a!) overlaps_bev) -> ()"));
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("iou3d_nms3d_forward(Tensor boxes, Tensor(a!) keep, Tensor(b!) keep_num, "
                                "float nms_overlap_thresh) -> ()"));
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("iou3d_nms3d_normal_forward(Tensor boxes, Tensor(a!) keep, Tensor(b!) keep_num, "
                                "float nms_overlap_thresh) -> ()"));
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("boxes_iou_bev_gpu(Tensor boxes_a, Tensor boxes_b, "
                                "Tensor(a!) overlaps_bev) -> ()"));
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("paired_boxes_overlap_bev_gpu(Tensor boxes_a, Tensor boxes_b, "
                                "Tensor(a!) overlaps_bev) -> ()"));
}