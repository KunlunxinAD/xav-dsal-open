#include "xpytorch.hpp"
#include "xav_dsal_ops.h"

enum ReductionType { SUM, MEAN, MUL, MIN, MAX, ASS };

const std::map<std::string, ReductionType> reduce2REDUCE
        = {{"sum", SUM}, {"add", SUM}, {"mean", MEAN}, {"mul", MUL}, {"min", MIN}, {"max", MAX}};

std::vector<at::Tensor> dynamic_scatter_forward(at::Tensor feats, at::Tensor coors, std::string reduce_type) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    int ret = 0;
    const int num_input = feats.size(0);
    const int num_feats = feats.size(1);
    auto REDUCE = reduce2REDUCE.at(reduce_type);
    if (REDUCE != SUM && REDUCE != MEAN && REDUCE != MAX) {
        AT_ASSERT(false, "reduce type must be sum, mean or max");
    }

    if (num_input == 0)
        return {feats.clone().detach(),
                coors.clone().detach(),
                coors.new_empty({0}, at::kInt),
                coors.new_empty({0}, at::kInt)};

    at::Tensor out_coors = at::empty({1}, feats.options());
    at::Tensor coors_map;
    at::Tensor reduce_count;
    at::Tensor arg_out;
    int* arg_out_data = nullptr;

    auto coors_clean = coors.masked_fill(coors.lt(0).any(-1, true), -1);
    std::tie(out_coors, coors_map, reduce_count) = unique_dim_xav(coors_clean, 0, true, true, true);
    // std::tie(out_coors, coors_map, reduce_count) = at::unique_dim(coors_clean, 0, true, true, true);

    if (out_coors[0][0].lt(0).item<bool>()) {
        out_coors = out_coors.slice(0, 1);
        reduce_count = reduce_count.slice(0, 1);
        coors_map = coors_map - 1;
    }

    coors_map = coors_map.to(at::kInt);
    reduce_count = reduce_count.to(at::kInt);

    auto reduced_feats = at::empty({out_coors.size(0), num_feats}, feats.options());

    if (REDUCE == MAX) {
        reduced_feats.fill_(-std::numeric_limits<float>::infinity());
        arg_out = at::full_like(reduced_feats, feats.size(0), coors.options().dtype(at::kInt));
        arg_out_data = arg_out.data_ptr<int>();
    } else if (REDUCE == MIN) {
        reduced_feats.fill_(std::numeric_limits<float>::infinity());
        arg_out = at::full_like(reduced_feats, feats.size(0), coors.options().dtype(at::kInt));
        arg_out_data = arg_out.data_ptr<int>();
    } else
        reduced_feats.fill_(static_cast<float>(0));
    auto index_shape_64 = coors_map.sizes().vec();
    auto index_stride_64 = coors_map.strides().vec();
    auto src_shape_64 = feats.sizes().vec();
    auto out_shape_64 = reduced_feats.sizes().vec();
    auto out_stride_64 = reduced_feats.strides().vec();

    std::vector<int> index_shape(index_shape_64.begin(), index_shape_64.end());
    std::vector<int> index_stride(index_stride_64.begin(), index_stride_64.end());
    std::vector<int> src_shape(src_shape_64.begin(), src_shape_64.end());
    std::vector<int> out_shape(out_shape_64.begin(), out_shape_64.end());
    std::vector<int> out_stride(out_stride_64.begin(), out_stride_64.end());

    float offset = 0.0f;
    int64_t offset_reduce = -1;

    if (feats.device().is_cuda()) {
        ret = xav::xpu::scatter_reduce<float>(
                ctx,
                feats.data_ptr<float>(),
                coors_map.data_ptr<int32_t>(),
                reduced_feats.data_ptr<float>(),
                arg_out_data,
                src_shape,
                index_shape,
                out_shape,
                index_stride,
                out_stride,
                0,
                REDUCE,
                offset,
                offset_reduce);
    } else {
        at::Tensor index = coors_map.clone().detach();
        index = xav::utils::broadcast(index, feats, 0);
        auto index_shape_64 = index.sizes().vec();
        auto index_stride_64 = index.strides().vec();
        std::vector<int> index_shape(index_shape_64.begin(), index_shape_64.end());
        std::vector<int> index_stride(index_stride_64.begin(), index_stride_64.end());

        ret = xav::cpu::scatter_reduce<float>(
                ctx,
                feats.data_ptr<float>(),
                index.data_ptr<int>(),
                reduced_feats.data_ptr<float>(),
                arg_out_data,
                src_shape,
                index_shape,
                out_shape,
                index_stride,
                feats.dim(),
                0,
                REDUCE,
                offset,
                offset_reduce);
    }
    assert(ret == 0);
    if (REDUCE == MEAN)
        reduced_feats /= reduce_count.unsqueeze(-1).to(reduced_feats.dtype());

    return {reduced_feats, out_coors, coors_map, reduce_count};
}

void dynamic_scatter_backward(
        at::Tensor grad_feats,
        at::Tensor grad_reduced_feats,
        at::Tensor feats,
        at::Tensor reduced_feats,
        at::Tensor coors_map,
        at::Tensor reduce_count,
        std::string reduce_type) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    int ret = 0;
    const int num_input = feats.size(0);
    const int num_reduced = reduced_feats.size(0);
    const int num_feats = feats.size(1);
    auto REDUCE = reduce2REDUCE.at(reduce_type);

    if (REDUCE != SUM && REDUCE != MEAN && REDUCE != MAX) {
        AT_ASSERT(false, "reduce type must be sum, mean or max");
    }
    if (num_input == 0 || num_reduced == 0)
        return;

    grad_feats.fill_(0);
    auto reduce_from = at::full({num_reduced, num_feats}, num_input, coors_map.options());

    auto kernel = xav::cpu::dynamic_scatter_bwd;
    if (feats.device().is_cuda()) {
        kernel = xav::xpu::dynamic_scatter_bwd;
    }
    ret = kernel(
            ctx,
            grad_reduced_feats.data_ptr<float>(),
            feats.data_ptr<float>(),
            reduced_feats.data_ptr<float>(),
            coors_map.data_ptr<int>(),
            reduce_count.data_ptr<int>(),
            reduce_from.data_ptr<int>(),
            grad_feats.data_ptr<float>(),
            num_input,
            num_feats,
            num_reduced,
            REDUCE);
    assert(ret == 0);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("dynamic_scatter_forward", &dynamic_scatter_forward);
    m.impl("dynamic_scatter_backward", &dynamic_scatter_backward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("dynamic_scatter_forward", &dynamic_scatter_forward);
    m.impl("dynamic_scatter_backward", &dynamic_scatter_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("dynamic_scatter_forward(Tensor feats, Tensor coors, str reduce_type) -> Tensor[]"));
    m.def(TORCH_SELECTIVE_SCHEMA(
            "dynamic_scatter_backward(Tensor grad_feats, Tensor grad_reduced_feats, Tensor feats, Tensor "
            "reduced_feats, Tensor coors_map, Tensor reduce_count, str reduce_type) -> ()"));
}
