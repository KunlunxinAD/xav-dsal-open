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

XAV_FUNC_XPU_AND_CPU(
        smooth_cosine_loss_forward,
        F_ARGS(api::Context* ctx,
               const float* emb_gt,
               const float* offset_feat,
               float* output,
               int batch_size,
               int bev_h,
               int bev_w,
               float h_pix,
               float w_pix,
               int ignore_label));

XAV_FUNC_XPU_AND_CPU(
        smooth_cosine_loss_backward,
        F_ARGS(api::Context* ctx,
               const float* emb_gt,
               const float* offset_feat,
               const float* grad_output,
               float* grad_offset_feat,
               int batch_size,
               int bev_h,
               int bev_w,
               float h_pix,
               float w_pix,
               int ignore_label));

XAV_FUNC_XPU_AND_CPU(
        linear_interpolate,
        F_ARGS(api::Context* ctx, const float* batch_points, float* output, int num_points, int batch_size, int n));
}    // namespace xav