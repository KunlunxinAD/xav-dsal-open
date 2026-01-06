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
        bev_pool_v2,
        F_ARGS(api::Context* ctx,
               int c,
               int n_intervals,
               const float* depth,
               const float* feat,
               const int* ranks_depth,
               const int* ranks_feat,
               const int* ranks_bev,
               const int* interval_starts,
               const int* interval_lengths,
               float* out));

XAV_FUNC_XPU_AND_CPU(
        bev_pool_v2_grad,
        F_ARGS(api::Context* ctx,
               int c,
               int n_intervals,
               const float* out_grad,
               const float* depth,
               const float* feat,
               const int* ranks_depth,
               const int* ranks_feat,
               const int* ranks_bev,
               const int* interval_starts,
               const int* interval_lengths,
               float* depth_grad,
               float* feat_grad));

XAV_FUNC_TMPL_XPU_AND_CPU(
        roiaware_pool3d,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* rois,
               const T* pts,
               const T* pts_feature,
               T* pooled_features,
               int pool_method,
               int* argmax,
               int* pts_idx_of_voxels,
               int boxes_num,
               int pts_num,
               int channels,
               int max_pts_each_voxel,
               int out_x,
               int out_y,
               int out_z));

XAV_FUNC_TMPL_XPU_AND_CPU(
        roiaware_pool3d_grad,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               int boxes_num,
               int out_x,
               int out_y,
               int out_z,
               int channels,
               int max_pts_each_voxel,
               int npoints,
               const int* pts_idx_of_voxels,
               const int* argmax_data,
               const T* grad_out_data,
               T* grad_in_data,
               int pool_method));

XAV_FUNC_TMPL_XPU_AND_CPU(
        roipoint_pool3d_forward,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
                int batch_size,
                int pts_num,
                int box_num,
                int feature_in_len,
                int sampled_pts_num,
                const T *points,
                const T *point_features,
                const T *boxes3d,
                T *pooled_features,
                int *pooled_empty_flag));
} // namespace xav