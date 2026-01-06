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

// XAV_FUNC_XPU(
//         _getIndicePair3D,
//         F_ARGS(api::Context* ctx,
//                 int* indices,
//                 int64_t batchSize,
//                 std::vector<int64_t> outSpatialShape,
//                 std::vector<int64_t> spatialShape,
//                 std::vector<int64_t> kernelSize,
//                 std::vector<int64_t> stride,
//                 std::vector<int64_t> padding,
//                 std::vector<int64_t> dilation,
//                 std::vector<int64_t> outPadding,
//                 int64_t _subM,
//                 int64_t _transpose));

XAV_FUNC_XPU(
        sigmoid_focal_loss,
        F_ARGS(api::Context* ctx,
               const float* input_ptr,
               const int64_t* target_ptr,
               const float* weight_ptr,
               float* output_ptr,
               float gamma,
               float alpha,
               int64_t input_len,
               int64_t target_len,
               int64_t num_classes));

XAV_FUNC_XPU(
        sigmoid_focal_loss_grad,
        F_ARGS(api::Context* ctx,
               const float* input_ptr,
               const int64_t* target_ptr,
               const float* weight_ptr,
               float* grad_input_ptr,
               float gamma,
               float alpha,
               int64_t input_len,
               int64_t target_len,
               int64_t num_classes));

//
// input tensor:    dim,                        range
// softmax:         [len, num_classes]
// target:          [len],                      [0, num_classes-1]
// weight:          [num_classes]
//
// output tensor:
// output:          [len]
//
XAV_FUNC_XPU_AND_CPU(
        softmax_focal_loss,
        F_ARGS(api::Context* ctx,
               const int len,
               const float* softmax_ptr,
               const int64_t* target_ptr,
               const float* weight_ptr,
               float* output_ptr,
               float gamma,
               float alpha,
               int64_t num_classes));

//
// input tensor:    dim,                        range
// softmax:         [len, num_classes]
// target:          [len],                      [0, num_classes-1]
// weight:          [num_classes]
//
// output tensor:
// grad_input:      [len, num_classes]
//
XAV_FUNC_XPU(
        softmax_focal_loss_grad,
        F_ARGS(api::Context* ctx,
               const int len,
               const float* softmax_ptr,
               const int64_t* target_ptr,
               const float* weight_ptr,
               float* grad_input_ptr,
               float gamma,
               float alpha,
               int64_t num_classes));

XAV_FUNC_CPU(
        softmax_focal_loss_grad,
        F_ARGS(api::Context* ctx,
               const int len,
               const float* softmax_ptr,
               const int64_t* target_ptr,
               const float* weight_ptr,
               float* buff_ptr,
               float* grad_input_ptr,
               float gamma,
               float alpha,
               int64_t num_classes));

//
// is_rstd   false   var
// is_rstd   true    var will be replaced with rstd
//
XAV_FUNC_TMPL_XPU_AND_CPU(
        layer_norm,
        TMPL_ARGS(typename T, typename TW),
        F_ARGS(api::Context* ctx,
               const T* x,
               T* y,
               int64_t m,
               int64_t n,
               float eps,
               const TW* scale,
               const TW* bias,
               float* mean,
               float* var,
               bool is_rstd));

// Real API
// is_rstd   false   var
// is_rstd   true    var input will be  rstd
XAV_FUNC_TMPL_XPU(
        layer_norm_grad,
        TMPL_ARGS(typename T, typename TW),
        F_ARGS(api::Context* ctx,
               const T* x,
               const T* dy,
               T* dx,
               int64_t m,
               int64_t n,
               float eps,
               const TW* scale,
               const float* mean,
               const float* var,
               TW* dscale,
               TW* dbias,
               bool is_rstd));

XAV_FUNC_TMPL_XPU(
        add_tensor,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* x,
               const T* other,
               T* y,
               const std::vector<int64_t>& xshape,
               const std::vector<int64_t>& othershape,
               float alpha));

XAV_FUNCTOR_TMPL_XPU_AND_CPU(
        get_indice_pairs_conv,
        TMPL_ARGS(typename Index, typename IndexGrid, unsigned NDim),
        F_ARGS(api::Context* ctx,
               const Index* indices_in,
               Index* indices_out,
               IndexGrid* grids_out,
               Index* indice_pairs,
               Index* indice_num,
               int64_t* num_act_out,
               const Index num_act_in,
               const Index batch_size,
               const std::vector<Index>& kernel_size,
               const std::vector<Index>& stride,
               const std::vector<Index>& padding,
               const std::vector<Index>& dilation,
               const std::vector<Index>& out_spatial_shape));

XAV_FUNCTOR_TMPL_XPU_AND_CPU(
        get_indice_pairs_subm,
        TMPL_ARGS(typename Index, typename IndexGrid, unsigned NDim),
        F_ARGS(api::Context* ctx,
               const Index* indices_in,
               IndexGrid* grids_out,
               Index* indice_pairs,
               Index* indice_num,
               const Index num_act_in,
               const std::vector<Index>& kernel_size,
               const std::vector<Index>& stride,
               const std::vector<Index>& padding,
               const std::vector<Index>& dilation,
               const std::vector<Index>& out_spatial_shape));

XAV_FUNC_TMPL_CPU(
        radius,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* x,
               const T* y,
               const int* ptrx,
               const int* ptry,
               int* x_res,
               int* y_res,
               int x_num,
               int y_num,
               int batch_size,
               int channel,
               float r,
               int max_num_neighbors,
               int num_workers,
               int ignore_same_index));
XAV_FUNC_TMPL_XPU(
        radius,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* x,
               const T* y,
               const int* ptrx,
               const int* ptry,
               int* x_res,
               int* y_res,
               int* batch_x_size,
               int x_num,
               int y_num,
               int batch_size,
               int channel,
               float r,
               int max_num_neighbors,
               int num_workers,
               int ignore_same_index));
XAV_FUNC_TMPL_CPU(
        knn,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               int b,
               int n,
               int m,
               int k,
               const T* xyz,
               const T* center_xyz,
               int* __restrict__ output_idx,
               T* output_dist));
XAV_FUNC_TMPL_XPU(
        knn,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               int b,
               int n,
               int m,
               int k,
               const T* xyz,
               const T* center_xyz,
               int* __restrict__ output_idx,
               T* output_dist));

XAV_FUNC_TMPL_CPU(
        scatter_reduce,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* src,
               const int* index,
               T* out,
               int* arg_out,
               const std::vector<int>& src_shape,
               const std::vector<int>& index_shape,
               const std::vector<int>& out_shape,
               const std::vector<int>& index_stride,
               const int num_dim,
               const int dim,
               const int64_t reduce,
               T offset,
               int64_t offset_reduce));

XAV_FUNC_TMPL_XPU(
        scatter_reduce,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* src,
               const int* index,
               T* out,
               int* arg_out,
               std::vector<int>& src_shape,
               std::vector<int>& index_shape,
               std::vector<int>& out_shape,
               std::vector<int>& index_stride,
               std::vector<int>& out_stride,
               const int dim,
               const int64_t reduce,
               T offset,
               int64_t offset_reduce));

XAV_FUNC_TMPL_XPU(
        compute_unique,
        TMPL_ARGS(typename T),
        F_ARGS(api::Context* ctx,
               const T* x_sorted,
               const int* sorted_indice,
               int* mark,
               int* inverse_indices,
               int rows,
               int cols));

XAV_FUNC_XPU_AND_CPU(
        dynamic_scatter_bwd,
        F_ARGS(api::Context* ctx,
               float* grad_reduced_feats,
               const float* feats,
               const float* reduced_feats,
               const int* coors_map,
               const int* reduce_count,
               int* reduce_from,
               float* grad_feats,
               int N,
               int C,
               int M,
               int reduce));
}    // namespace xav