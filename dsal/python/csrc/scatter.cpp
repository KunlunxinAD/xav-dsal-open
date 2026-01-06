#include "xpytorch.hpp"
#include <vector>
#include <string>
#include <map>

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

namespace xav {
namespace utils {

at::Tensor broadcast(at::Tensor src, at::Tensor other, int64_t dim) {
    if (src.dim() == 1) {
        for (auto i = 0; i < dim; i++) {
            src = src.unsqueeze(0);
        }
    }
    for (auto i = src.dim(); i < other.dim(); i++) {
        src = src.unsqueeze(-1);
    }

    src = src.expand(other.sizes().vec());
    return src;
}

}    // namespace utils
}    // namespace xav

enum ReductionType { SUM, MEAN, MUL, MIN, MAX, ASS };

const std::map<std::string, ReductionType> reduce2REDUCE
        = {{"sum", SUM}, {"add", SUM}, {"mean", MEAN}, {"mul", MUL}, {"min", MIN}, {"max", MAX}};

template <typename scalar_t>
struct Reducer {
    static inline scalar_t init(ReductionType REDUCE) {
        if (REDUCE == MUL)
            return (scalar_t)1;
        else if (REDUCE == MIN)
            return std::numeric_limits<scalar_t>::max();
        else if (REDUCE == MAX)
            return std::numeric_limits<scalar_t>::lowest();
        else
            return (scalar_t)0;
    }
};
inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
    std::vector<int64_t> result;
    result.reserve(list.size());
    for (size_t i = 0; i < list.size(); i++) {
        result.push_back(list[i]);
    }
    return result;
}

int scatter_fwd(
        at::Tensor src,
        at::Tensor& out,
        at::Tensor index,
        int64_t dim,
        ReductionType REDUCE,
        c10::optional<int64_t> dim_size,
        c10::optional<at::Tensor> optional_out,
        c10::optional<at::Tensor> offset,
        c10::optional<std::string> offset_reduce,
        at::Tensor* arg_out = nullptr,
        int* arg_out_data = nullptr) {
    int ret = 0;
    auto xpu_ctx = xmlir_rt::getXpuKernelContext();
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::kLong, "index must be long type otherwise grad compute will fail");
    at::Tensor index_int32 = index.to(at::kInt);
    if (index_int32.dim() == 1) {
        AT_ASSERTM(src.size(dim) >= index_int32.size(0), "src size larger than index size on dim");
    } else {
        for (int i = 0; i < index_int32.dim(); i++) {
            AT_ASSERTM(src.size(i) >= index_int32.size(i), "src size larger than index size");
        }
    }
    if (optional_out.has_value()) {
        TORCH_CHECK(optional_out->is_contiguous());
        out = optional_out.value().contiguous();
        for (auto i = 0; i < out.dim(); i++) {
            if (i != dim)
                AT_ASSERTM(out.size(i) == src.size(i), "src size must equal out size (except dim)");
        }
    } else {
        auto sizes = src.sizes().vec();
        if (dim_size.has_value()) {
            sizes[dim] = dim_size.value();
        } else if (index_int32.numel() == 0) {
            sizes[dim] = 0;
        } else {
            sizes[dim] = 1 + index_int32.max().cpu().data_ptr<int>()[0];
        }
        out = at::zeros(sizes, src.options());
    }
    if (offset.has_value()) {
        TORCH_CHECK(offset_reduce.has_value());
        AT_ASSERTM(offset.value().dim() == 1, "offset dim must be 1");
        AT_ASSERTM(offset.value().size(0) == 1, "offset dim must be 1 number");
        AT_ASSERTM(offset.value().scalar_type() == src.scalar_type(), "offset and src must be same value type");
        auto OFFSET_REDUCE = reduce2REDUCE.at(offset_reduce.value());
        bool flag = (OFFSET_REDUCE == SUM || OFFSET_REDUCE == MUL);
        AT_ASSERTM(flag, "offset reduce only support SUM or MUL");
    }

    if (REDUCE == MIN || REDUCE == MAX) {
        *arg_out = at::full_like(out, src.size(dim), index_int32.options());
        arg_out_data = arg_out->data_ptr<int>();
    }

    if (src.numel() == 0) {
        if (!optional_out.has_value())
            out.fill_(0);
        return 0;
    }

    AT_DISPATCH_FLOAT_INT_TYPES(src.scalar_type(), "scatter_fwd", [&] {
        scalar_t offset_ = (scalar_t)0;
        int64_t offset_reduce_ = -1;
        if (!optional_out.has_value()) {
            out.fill_(Reducer<scalar_t>::init(REDUCE));
        }
        if (offset.has_value()) {
            offset_ = offset.value().data_ptr<scalar_t>()[0];
            offset_reduce_ = reduce2REDUCE.at(offset_reduce.value());
        }
        auto index_shape_64 = index_int32.sizes().vec();
        auto index_stride_64 = index_int32.strides().vec();
        auto src_shape_64 = src.sizes().vec();
        auto out_shape_64 = out.sizes().vec();
        auto out_stride_64 = out.strides().vec();

        std::vector<int> index_shape(index_shape_64.begin(), index_shape_64.end());
        std::vector<int> index_stride(index_stride_64.begin(), index_stride_64.end());
        std::vector<int> src_shape(src_shape_64.begin(), src_shape_64.end());
        std::vector<int> out_shape(out_shape_64.begin(), out_shape_64.end());
        std::vector<int> out_stride(out_stride_64.begin(), out_stride_64.end());

        auto src_data = src.data_ptr<scalar_t>();
        auto index_data = index_int32.data_ptr<int>();
        auto out_data = out.data_ptr<scalar_t>();

        if (src.device().is_cuda()) {
            ret = xav::xpu::scatter_reduce<scalar_t>(
                    xpu_ctx,
                    src_data,
                    index_data,
                    out_data,
                    arg_out_data,
                    src_shape,
                    index_shape,
                    out_shape,
                    index_stride,
                    out_stride,
                    dim,
                    REDUCE,
                    offset_,
                    offset_reduce_);
        } else {
            index_int32 = xav::utils::broadcast(index_int32, src, dim);
            auto index_shape_64 = index_int32.sizes().vec();
            auto index_stride_64 = index_int32.strides().vec();
            std::vector<int> index_shape(index_shape_64.begin(), index_shape_64.end());
            std::vector<int> index_stride(index_stride_64.begin(), index_stride_64.end());

            ret = xav::cpu::scatter_reduce<scalar_t>(
                    xpu_ctx,
                    src_data,
                    index_data,
                    out_data,
                    arg_out_data,
                    src_shape,
                    index_shape,
                    out_shape,
                    index_stride,
                    src.dim(),
                    dim,
                    REDUCE,
                    offset_,
                    offset_reduce_);
        }
        if (!optional_out.has_value() && (REDUCE == MIN || REDUCE == MAX))
            out.masked_fill_(out == Reducer<scalar_t>::init(REDUCE), 0.0f);
    });
    return ret;
}

class ScatterMax : public torch::autograd::Function<ScatterMax> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            Variable src,
            Variable index,
            int64_t dim,
            c10::optional<Variable> optional_out,
            c10::optional<int64_t> dim_size,
            c10::optional<Variable> offset,
            c10::optional<std::string> offset_reduce) {
        dim = dim < 0 ? src.dim() + dim : dim;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["src_shape"] = src.sizes();

        at::Tensor out;
        at::Tensor arg_out;
        int* arg_out_data = nullptr;
        auto REDUCE = reduce2REDUCE.at("max");

        int ret = scatter_fwd(
                src, out, index, dim, REDUCE, dim_size, optional_out, offset, offset_reduce, &arg_out, arg_out_data);
        assert(ret == 0);
        arg_out = arg_out.to(at::kLong);

        ctx->save_for_backward({index, arg_out});
        ctx->mark_non_differentiable({arg_out});
        if (optional_out.has_value())
            ctx->mark_dirty({optional_out.value()});
        return {out, arg_out};
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto index = saved[0];
        auto arg_out = saved[1];
        auto dim = ctx->saved_data["dim"].toInt();
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        src_shape[dim] += 1;
        auto grad_in = at::zeros(src_shape, grad_out.options());
        grad_in.scatter_(dim, arg_out, grad_out);
        grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);
        return {grad_in, Variable(), Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

class ScatterMin : public torch::autograd::Function<ScatterMin> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            Variable src,
            Variable index,
            int64_t dim,
            c10::optional<Variable> optional_out,
            c10::optional<int64_t> dim_size,
            c10::optional<Variable> offset,
            c10::optional<std::string> offset_reduce) {
        dim = dim < 0 ? src.dim() + dim : dim;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["src_shape"] = src.sizes();

        at::Tensor out;
        at::Tensor arg_out;
        int* arg_out_data = nullptr;
        auto REDUCE = reduce2REDUCE.at("min");

        int ret = scatter_fwd(
                src, out, index, dim, REDUCE, dim_size, optional_out, offset, offset_reduce, &arg_out, arg_out_data);
        assert(ret == 0);
        arg_out = arg_out.to(at::kLong);

        ctx->save_for_backward({index, arg_out});
        ctx->mark_non_differentiable({arg_out});
        if (optional_out.has_value())
            ctx->mark_dirty({optional_out.value()});
        return {out, arg_out};
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto index = saved[0];
        auto arg_out = saved[1];
        auto dim = ctx->saved_data["dim"].toInt();
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        src_shape[dim] += 1;
        auto grad_in = torch::zeros(src_shape, grad_out.options());
        grad_in.scatter_(dim, arg_out, grad_out);
        grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);
        return {grad_in, Variable(), Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

class ScatterSum : public torch::autograd::Function<ScatterSum> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            Variable src,
            Variable index,
            int64_t dim,
            c10::optional<Variable> optional_out,
            c10::optional<int64_t> dim_size,
            c10::optional<Variable> offset,
            c10::optional<std::string> offset_reduce) {
        dim = dim < 0 ? src.dim() + dim : dim;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["src_shape"] = src.sizes();

        at::Tensor out;
        auto REDUCE = reduce2REDUCE.at("sum");

        int ret = scatter_fwd(src, out, index, dim, REDUCE, dim_size, optional_out, offset, offset_reduce);
        assert(ret == 0);
        index = xav::utils::broadcast(index, src, dim);

        ctx->save_for_backward({index});
        if (optional_out.has_value())
            ctx->mark_dirty({optional_out.value()});
        return {out};
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto index = saved[0];
        auto dim = ctx->saved_data["dim"].toInt();
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        auto grad_in = torch::gather(grad_out, dim, index, false);
        return {grad_in, Variable(), Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

class ScatterMul : public torch::autograd::Function<ScatterMul> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            Variable src,
            Variable index,
            int64_t dim,
            c10::optional<Variable> optional_out,
            c10::optional<int64_t> dim_size,
            c10::optional<Variable> offset,
            c10::optional<std::string> offset_reduce) {
        dim = dim < 0 ? src.dim() + dim : dim;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["src_shape"] = src.sizes();

        at::Tensor out;
        auto REDUCE = reduce2REDUCE.at("mul");

        int ret = scatter_fwd(src, out, index, dim, REDUCE, dim_size, optional_out, offset, offset_reduce);
        assert(ret == 0);
        index = xav::utils::broadcast(index, src, dim);

        ctx->save_for_backward({src, index, out});
        if (optional_out.has_value())
            ctx->mark_dirty({optional_out.value()});
        return {out};
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto src = saved[0];
        auto index = saved[1];
        auto out = saved[2];
        auto dim = ctx->saved_data["dim"].toInt();
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        auto grad_in = torch::gather(grad_out * out, dim, index, false).div_(src);
        grad_in.masked_fill_(grad_in.isnan(), 0);
        return {grad_in, Variable(), Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

class ScatterMean : public torch::autograd::Function<ScatterMean> {
public:
    static variable_list forward(
            AutogradContext* ctx,
            Variable src,
            Variable index,
            int64_t dim,
            c10::optional<Variable> optional_out,
            c10::optional<int64_t> dim_size,
            c10::optional<Variable> offset,
            c10::optional<std::string> offset_reduce) {
        dim = dim < 0 ? src.dim() + dim : dim;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["src_shape"] = src.sizes();

        at::Tensor out;
        auto REDUCE = reduce2REDUCE.at("sum");
        auto old_index = index;
        at::Tensor count_out;

        int ret = scatter_fwd(src, out, index, dim, REDUCE, dim_size, optional_out, offset, offset_reduce);
        assert(ret == 0);
        if (dim < index.dim()) {
            auto ones = torch::ones(old_index.sizes(), src.options());
            scatter_fwd(
                    ones,
                    count_out,
                    old_index,
                    old_index.dim() <= dim ? old_index.dim() - 1 : dim,
                    REDUCE,
                    out.size(dim),
                    c10::nullopt,
                    c10::nullopt,
                    c10::nullopt);
            count_out.masked_fill_(count_out < 1, 1);
            count_out = xav::utils::broadcast(count_out, out, dim);
        } else {
            auto ones = torch::ones(src.sizes(), src.options());
            scatter_fwd(
                    ones, count_out, old_index, dim, REDUCE, out.size(dim), c10::nullopt, c10::nullopt, c10::nullopt);
            count_out.masked_fill_(count_out < 1, 1);
        }
        if (out.is_floating_point())
            out.true_divide_(count_out);
        else
            out.div_(count_out, "floor");
        index = xav::utils::broadcast(index, src, dim);
        ctx->save_for_backward({index, count_out});
        if (optional_out.has_value())
            ctx->mark_dirty({optional_out.value()});
        return {out};
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto index = saved[0];
        auto count = saved[1];
        auto dim = ctx->saved_data["dim"].toInt();
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        count = torch::gather(count, dim, index, false);
        auto grad_in = torch::gather(grad_out, dim, index, false);
        grad_in.true_divide_(count);
        return {grad_in, Variable(), Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

std::vector<at::Tensor> scatter(
        at::Tensor src,
        at::Tensor index,
        int64_t dim,
        c10::optional<Variable> optional_out,
        c10::optional<int64_t> dim_size,
        std::string reduce,
        c10::optional<at::Tensor> offset,
        c10::optional<std::string> offset_reduce) {
    auto REDUCE = reduce2REDUCE.at(reduce);
    if (REDUCE == SUM) {
        return ScatterSum::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce);
    } else if (REDUCE == MUL) {
        return ScatterMul::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce);
    } else if (REDUCE == MAX) {
        return ScatterMax::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce);
    } else if (REDUCE == MIN) {
        return ScatterMin::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce);
    } else if (REDUCE == MEAN) {
        return ScatterMean::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce);
    } else {
        AT_ASSERTM(false, "Not supported reduce type");
    }
}

at::Tensor scatter_sum(
        at::Tensor src,
        at::Tensor index,
        int64_t dim,
        c10::optional<at::Tensor> optional_out,
        c10::optional<int64_t> dim_size,
        c10::optional<at::Tensor> offset,
        c10::optional<std::string> offset_reduce) {
    return ScatterSum::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce)[0];
}
at::Tensor scatter_mul(
        at::Tensor src,
        at::Tensor index,
        int64_t dim,
        c10::optional<at::Tensor> optional_out,
        c10::optional<int64_t> dim_size,
        c10::optional<at::Tensor> offset,
        c10::optional<std::string> offset_reduce) {
    return ScatterMul::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce)[0];
}
at::Tensor scatter_mean(
        at::Tensor src,
        at::Tensor index,
        int64_t dim,
        c10::optional<at::Tensor> optional_out,
        c10::optional<int64_t> dim_size,
        c10::optional<at::Tensor> offset,
        c10::optional<std::string> offset_reduce) {
    return ScatterMean::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce)[0];
}
std::tuple<at::Tensor, at::Tensor> scatter_min(
        at::Tensor src,
        at::Tensor index,
        int64_t dim,
        c10::optional<at::Tensor> optional_out,
        c10::optional<int64_t> dim_size,
        c10::optional<at::Tensor> offset,
        c10::optional<std::string> offset_reduce) {
    auto result = ScatterMin::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce);
    return std::make_tuple(result[0], result[1]);
}

std::tuple<at::Tensor, at::Tensor> scatter_max(
        at::Tensor src,
        at::Tensor index,
        int64_t dim,
        c10::optional<at::Tensor> optional_out,
        c10::optional<int64_t> dim_size,
        c10::optional<at::Tensor> offset,
        c10::optional<std::string> offset_reduce) {
    auto result = ScatterMax::apply(src, index, dim, optional_out, dim_size, offset, offset_reduce);
    return std::make_tuple(result[0], result[1]);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("scatter_sum", &scatter_sum);
    m.impl("scatter_mul", &scatter_mul);
    m.impl("scatter_mean", &scatter_mean);
    m.impl("scatter_min", &scatter_min);
    m.impl("scatter_max", &scatter_max);
    m.impl("scatter", &scatter);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("scatter_sum", &scatter_sum);
    m.impl("scatter_mul", &scatter_mul);
    m.impl("scatter_mean", &scatter_mean);
    m.impl("scatter_min", &scatter_min);
    m.impl("scatter_max", &scatter_max);
    m.impl("scatter", &scatter);
}

TORCH_LIBRARY_IMPL(xav_dsal, Autograd, m) {
    m.impl("scatter_sum", &scatter_sum);
    m.impl("scatter_mul", &scatter_mul);
    m.impl("scatter_mean", &scatter_mean);
    m.impl("scatter_min", &scatter_min);
    m.impl("scatter_max", &scatter_max);
    m.impl("scatter", &scatter);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("scatter_sum(Tensor src, Tensor index, int dim, Tensor? optional_out, int? dim_size, "
                                 "Tensor? offset, str? offset_reduce) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA("scatter_mul(Tensor src, Tensor index, int dim, Tensor? optional_out, int? dim_size, "
                                 "Tensor? offset, str? offset_reduce) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA("scatter_mean(Tensor src, Tensor index, int dim, Tensor? optional_out, int? dim_size, "
                                 "Tensor? offset, str? offset_reduce) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA("scatter_min(Tensor src, Tensor index, int dim, Tensor? optional_out, int? dim_size, "
                                 "Tensor? offset, str? offset_reduce) -> (Tensor, Tensor)"));
    m.def(TORCH_SELECTIVE_SCHEMA("scatter_max(Tensor src, Tensor index, int dim, Tensor? optional_out, int? dim_size, "
                                 "Tensor? offset, str? offset_reduce) -> (Tensor, Tensor)"));
    m.def(TORCH_SELECTIVE_SCHEMA("scatter(Tensor src, Tensor index, int dim, Tensor? optional_out, int? dim_size, str "
                                 "reduce, Tensor? offset, str? offset_reduce) -> Tensor[]"));
}