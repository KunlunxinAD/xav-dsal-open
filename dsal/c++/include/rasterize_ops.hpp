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

// XAV_FUNC_CPU(
//         gaussian_rasterization_forward,
//         F_ARGS(api::Context* ctx,
//                 std::function<char* (size_t)> geometryBuffer,
//                 std::function<char* (size_t)> binningBuffer,
//                 std::function<char* (size_t)> imageBuffer,
//                 const int P, int N,
//                 const float* pts,
//                 const int* points_int,
//                 const float* means3D,
//                 const int* means3D_int,
//                 const float* opacities,
//                 const float* semantics,
//                 const float* cov3D,
//                 const int* radii,
//                 const int H, const int W, const int D,
//                 float* out));

// XAV_FUNC_CPU(
//         gaussian_rasterization_backward,
//         F_ARGS(api::Context* ctx,
//                 const int P, int R, int N,
//                 const int H, int W, int D,
//                 char* geom_buffer,
//                 char* binning_buffer,
//                 char* img_buffer,
//                 const int* points_int,
//                 int* voxel2pts,
//                 const float* pts,
//                 const float* means3D,
//                 const float* cov3D,
//                 const float* opacities,
//                 const float* semantics,
//                 const float* out_grad,
//                 float* means3D_grad,
//                 float* opacity_grad,
//                 float* semantics_grad,
//                 float* cov3D_grad));

XAV_FUNC_TMPL_XPU_AND_CPU(
        forward_rasterize,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* vertices,
               T* rasterized,
               int* contribution_map,
               const int batch_size,
               const int num_vertices,
               const int width,
               const int height,
               const float inv_smoothness,
               const int mode));

XAV_FUNC_TMPL_XPU_AND_CPU(
        backward_rasterize,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* vertices,
               const T* rasterized,
               const int* contribution_map,
               const T* grad_output,
               T* grad_vertices,
               const int batch_size,
               const int num_vertices,
               const int width,
               const int height,
               const float inv_smoothness,
               const int mode));

XAV_FUNC_XPU(
        forward_rasterize_xtrans,
        F_ARGS(api::Context* ctx,
               const int threads,
               const float* __restrict__ vertices,
               int batch_size,
               int number_vertices,
               float* rasterized,
               int* contribution_map,
               int height,
               int width,
               float inv_smoothness,
               int mode));
XAV_FUNC_XPU(
        backward_rasterize_xtrans,
        F_ARGS(api::Context* ctx,
               const int threads,
               const float* __restrict__ vertices,
               const float* __restrict__ rasterized,
               const int* __restrict__ contribution_map,
               const float* __restrict__ grad_output,
               float* grad_vertices,
               int batch_size,
               int number_vertices,
               int width,
               int height,
               float inv_smoothness));
} // namespace xav