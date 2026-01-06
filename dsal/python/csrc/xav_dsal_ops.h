#pragma once
#include <tuple>
#include "xpytorch.hpp"

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_xav(
        at::Tensor self,
        int64_t dim,
        bool sorted,
        bool return_inverse,
        bool return_counts);