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

//
// item    , shape         , info
// ----    , ----          , ----
// dst     , num_dst x len , output
// src     , num_src x len , input
// indices , num_dst       , input
//
// dst[i, :] = src[indices[i], :] for i in [0, num_src)
//
XAV_FUNC_TMPL_XPU_AND_CPU(
        sparse_gather,
        TMPL_ARGS(typename T, typename Index),
        F_ARGS(XAV_HOST api::Context* ctx, T* dst, const T* src, const Index* indices, int num_planes, int num_pairs));

//
// item    , shape         , info
// ----    , ----          , ----
// dst     , num_dst x len , output
// src     , num_src x len , input
// indices , num_dst       , input
//
// dst[indices[i], :] += src[i, :] for i in [0, num_src)
//
XAV_FUNC_TMPL_XPU_AND_CPU(
        sparse_scatter_add,
        TMPL_ARGS(typename T, typename Index),
        F_ARGS(api::Context* ctx,
               T* dst,
               const T* src,
               const Index* indices,
               int out_act_num,
               int num_planes,
               int num_pairs));

} // namespace xav