#include "xpytorch.hpp"
#include "xav_dsal_ops.h"

namespace torch::ops::custom_ops {
TORCH_API std::tuple<Tensor, Tensor> sort_2d_stable(
        const Tensor& self,
        bool stable,
        int64_t dim,
        bool descending,
        Tensor& output,
        Tensor& indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_xav(
        at::Tensor self,
        int64_t dim,
        bool sorted,
        bool return_inverse,
        bool return_counts) {
    auto origin_sizes = self.sizes().vec();
    int rows = origin_sizes[dim];
    auto num_zero_dims = std::count(origin_sizes.begin(), origin_sizes.end(), 0);

    if (self.size(dim) == 0) {
        TORCH_CHECK(
                num_zero_dims == 1, "Number of zero sized dimensions is more than one, so unique cannot be applied");
        at::Tensor output = at::empty(origin_sizes, self.options());
        at::Tensor inverse_indices = at::empty({0}, self.options().dtype(at::kInt));
        at::Tensor counts = at::empty({0}, self.options().dtype(at::kInt));

        return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, inverse_indices, counts);
    }
    TORCH_CHECK(
            num_zero_dims == 0, "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

    if (dim < 0) {
        dim = self.dim() + dim;
    }
    at::Tensor output;
    at::Tensor sorted_indices = at::empty({rows}, self.options().dtype(at::kInt));
    at::Tensor inverse_indices = at::zeros({rows}, self.options().dtype(at::kInt));
    at::Tensor counts = at::empty({0}, self.options().dtype(at::kInt));
    at::Tensor undefined_inverse_indices = at::empty({0}, self.options().dtype(at::kInt));
    at::Tensor undefined_counts = at::empty({0}, self.options().dtype(at::kInt));
    at::Tensor mark = at::empty({rows}, self.options().dtype(at::kInt));

    if (self.device().is_cuda()) {
        int ret = 0;
        auto xpu_ctx = xmlir_rt::getXpuKernelContext();

        self = self.moveaxis(dim, 0).contiguous().view({rows, -1});
        auto trans_sizes = self.sizes().vec();
        output = at::empty_like(self);
        at::Tensor sorted_self = at::empty_like(self);
        int new_rows = rows;
        int cols = trans_sizes[1];

#if 0
// version2 normal stable_sort + index_select
        for (int i = cols - 1; i >= 0; i--) {
            at::Tensor col = self.select(1, i);
            at::Tensor temp_indices = at::argsort(col, true, 0, false);
            sorted_indices = (i == cols - 1) ? temp_indices : sorted_indices.index_select(0, temp_indices);
            self = self.index_select(0, temp_indices);
        }
        sorted_indices = sorted_indices.to(at::kInt);
#endif
        static auto op = torch::Dispatcher::singleton()
                                 .findSchemaOrThrow("custom_ops::sort_2d_stable", "")
                                 .typed<std::tuple<at::Tensor&, at::Tensor&>(
                                         at::Tensor const&, bool, long, bool, at::Tensor&, at::Tensor&)>();
        op.call(self, true, 0, false, sorted_self, sorted_indices);

        AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "unique_dim", [&] {
            ret = xav::xpu::compute_unique<scalar_t>(
                    xpu_ctx,
                    sorted_self.data_ptr<scalar_t>(),
                    sorted_indices.data_ptr<int>(),
                    mark.data_ptr<int>(),
                    inverse_indices.data_ptr<int>(),
                    rows,
                    cols);
        });

        at::Tensor indexes = at::nonzero(mark).squeeze(1);
        new_rows = indexes.numel();
        output = sorted_self.index_select(0, indexes);

        at::Tensor count_mid = indexes.slice(0, 1) - indexes.slice(0, 0, -1);
        at::Tensor count_last = (rows - indexes.index({-1})).unsqueeze(0);
        counts = at::cat({count_mid, count_last}, 0);

        std::vector<int64_t> reverse_sizes;
        reverse_sizes.push_back(new_rows);
        for (int i = 0; i < origin_sizes.size(); i++) {
            if (i != dim)
                reverse_sizes.push_back(origin_sizes[i]);
        }
        output = output.view(reverse_sizes).moveaxis(0, dim);

        if (!return_inverse) {
            inverse_indices = undefined_inverse_indices;
        }
        if (!return_counts) {
            counts = undefined_counts;
        }

    } else {
        std::tie(output, inverse_indices, counts) = at::unique_dim(self, dim, true, true, true);
    }

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            output, inverse_indices.toType(at::kLong), counts.toType(at::kLong));
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("unique_dim", &unique_dim_xav);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("unique_dim", &unique_dim_xav);
}
TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("unique_dim(Tensor self, int dim, bool sorted, bool return_inverse, bool "
                                 "return_counts) -> (Tensor, Tensor, Tensor)"));
}
