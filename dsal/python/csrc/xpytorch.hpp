#pragma once

#include "xdnn_pytorch/xdnn_pytorch.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_check.h"
#include "xdnn_pytorch/xdnn_pytorch_wrapper_dump.h"

#include <runtime/xpu_context.h>

#include <xav_dsal.hpp>

#include <torch/torch.h>

#include <stdint.h>

#define AT_DISPATCH_FLOAT_INT_TYPES(scalar_type, name, ...)                          \
    [&] {                                                                            \
        using namespace at;                                                          \
        if (scalar_type == ScalarType::Float) {                                      \
            using scalar_t = float;                                                  \
            return __VA_ARGS__();                                                    \
        } else if (scalar_type == ScalarType::Int) {                                 \
            using scalar_t = int;                                                    \
            return __VA_ARGS__();                                                    \
        } else if (scalar_type == ScalarType::Long) {                                \
            using scalar_t = int64_t;                                                \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            AT_ERROR(#name " does not support scalar type ", toString(scalar_type)); \
        }                                                                            \
    }()

#define AT_DISPATCH_FLOAT_TWO_TYPES(scalar_type, name, ...)                          \
    [&] {                                                                            \
        using namespace at;                                                          \
        if (scalar_type == ScalarType::Float) {                                      \
            using scalar_t = float;                                                  \
            using scalar_ptr_t = float;                                              \
            using gemm_t = int;                                                      \
            return __VA_ARGS__();                                                    \
        } else if (scalar_type == ScalarType::Half) {                                \
            using scalar_t = float16;                                                \
            using scalar_ptr_t = at::Half;                                           \
            using gemm_t = int16_t;                                                  \
            return __VA_ARGS__();                                                    \
        } else {                                                                     \
            AT_ERROR(#name " does not support scalar type ", toString(scalar_type)); \
        }                                                                            \
    }()

inline void print_tensor_shape(at::Tensor t, std::string name) {
    auto shape = t.sizes();
    printf("tensor %s (dtype %s) shape: ", name.c_str(), at::toString(t.scalar_type()));
    for (auto s : shape) {
        printf("%ld, ", s);
    }
    printf("\n");
}

namespace xav {

namespace utils {

template <typename T>
struct map_torch_type {
    using type = T;
};

template <>
struct map_torch_type<at::Half> {
    using type = float16;
};

at::Tensor broadcast(at::Tensor src, at::Tensor other, int64_t dim);

}    // namespace utils

}    // namespace xav
