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
        hard_voxelize,
        TMPL_ARGS(typename T, typename TID),
        F_ARGS(api::Context* ctx,
               const T* points,
               const float coors_x_min,
               const float coors_y_min,
               const float coors_z_min,
               const float voxel_size_x,
               const float voxel_size_y,
               const float voxel_size_z,
               const int grid_size_x,
               const int grid_size_y,
               const int grid_size_z,
               const int64_t num_points,
               const int num_point_dim,
               const int max_points,
               const int max_voxels,
               T* voxels,
               TID* coords,
               TID* num_points_per_voxel,
               TID* grid_idx_to_voxel_idx,
               TID* num_voxels));

XAV_FUNC_TMPL_XPU_AND_CPU(
        hard_voxelize_forward,
        TMPL_ARGS(typename T, typename TID),
        F_ARGS(api::Context* ctx,
               const T* points,
               const float coors_x_min,
               const float coors_y_min,
               const float coors_z_min,
               const float voxel_size_x,
               const float voxel_size_y,
               const float voxel_size_z,
               const int grid_size_x,
               const int grid_size_y,
               const int grid_size_z,
               const int64_t num_points,
               const int num_point_dim,
               const int max_points,
               const int max_voxels,
               T* voxels,
               TID* coords,
               TID* num_points_per_voxel,
               TID* grid_idx_to_voxel_idx,
               TID* num_voxels));

XAV_FUNC_TMPL_XPU_AND_CPU(
        dynamic_voxelize,
        TMPL_ARGS(typename T, typename TID),
        F_ARGS(api::Context* ctx,
                const T* points,
                TID* coors,
                const float coors_x_min,
                const float coors_y_min,
                const float coors_z_min,
                const float voxel_x,
                const float voxel_y,
                const float voxel_z,
                const int grid_x,
                const int grid_y,
                const int grid_z,
                const int64_t num_points,
                const int num_features));

} // namespace xav

