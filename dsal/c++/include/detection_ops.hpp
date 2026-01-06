#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include "xav_dsal_defs.h"

namespace xpytorch {
namespace xpu {
namespace api {
struct Context;
}
}    // namespace xpu
}    // namespace xpytorch

namespace xav {

namespace api = xpytorch::xpu::api;

XAV_FUNC_TMPL_XPU_AND_CPU(
        box_iou_rotated,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const int num_boxes1,
               const int num_boxes2,
               const T* boxes1,
               const T* boxes2,
               T* ious,
               const int mode,
               const bool aligned));
XAV_FUNC_XPU_AND_CPU(
        iou3d_boxes_overlap_bev_forward,
        F_ARGS(api::Context* ctx,
               int N,
               const float* boxes_a,
               int M,
               const float* boxes_b,
               float* overlap));
XAV_FUNC_XPU_AND_CPU(
        iou3d_nms3d_forward,
        F_ARGS(api::Context* ctx,
                int boxes_num, 
                float nms_overlap_thresh, 
                const float *boxes, 
                uint32_t *mask));
XAV_FUNC_XPU(
        gather_keep_from_mask,
        F_ARGS(api::Context* ctx,
                int n_boxes,
                uint32_t *mask,
                bool *keep));
XAV_FUNC_XPU_AND_CPU(
        iou3d_nms3d_normal_forward,
        F_ARGS(api::Context* ctx,
                int boxes_num, 
                float nms_overlap_thresh, 
                const float *boxes, 
                uint32_t *mask));
XAV_FUNC_XPU_AND_CPU(
        boxes_iou_bev_kernel,
        F_ARGS(api::Context* ctx,
                int N,
                const float* boxes_a,
                int M,
                const float* boxes_b,
                float* overlap));
XAV_FUNC_XPU_AND_CPU(
        paired_boxes_overlap_kernel,
        F_ARGS(api::Context* ctx,
                int N,
                const float* boxes_a,
                int M,
                const float* boxes_b,
                float* overlap));
} // namespace xav