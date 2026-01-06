#include "xpytorch.hpp"

at::Tensor radius(
        at::Tensor x,
        at::Tensor y,
        c10::optional<at::Tensor> ptr_x,
        c10::optional<at::Tensor> ptr_y,
        double r_,
        int64_t max_num_neighbors,
        int64_t num_workers,
        bool ignore_same_index = false) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    TORCH_CHECK(x.is_contiguous(), " must be contiguous");
    TORCH_CHECK(y.is_contiguous(), " must be contiguous");

    AT_ASSERTM(x.dim() == 2, "x must be 2 dim");
    AT_ASSERTM(y.dim() == 2, "y must be 2 dim");
    AT_ASSERTM(x.size(1) == y.size(1), "x.size(1) must = y.size(1), which is channel");

    at::Tensor ptr_x_data, ptr_y_data;
    int* ptrx = nullptr;
    int* ptry = nullptr;
    int batch_size = 1;
    std::vector<int> batch_x_size;

    if (ptr_x.has_value()) {
        ptr_x_data = ptr_x->to(at::kInt);
        TORCH_CHECK(ptr_x_data.is_contiguous(), " must be contiguous");
        AT_ASSERTM(ptr_x_data.dim() == 1, "ptr_x must be 1 dim");
        ptrx = ptr_x_data.data_ptr<int>();
        batch_size = ptr_x_data.size(0) - 1;

        auto x_cpu = ptr_x_data.cpu().contiguous();
        const int* x_cpu_data = x_cpu.data_ptr<int>();
        for (int i = 0; i < batch_size; i++) {
            batch_x_size.push_back((x_cpu_data[i + 1] - x_cpu_data[i]) * x.size(1));
        }
    } else {
        std::vector<int> x_vec = {0, static_cast<int>(x.size(0))};
        ptr_x_data = at::tensor(x_vec, at::kInt);
        ptr_x_data = x.device().is_cuda() ? ptr_x_data.to(at::kCUDA) : ptr_x_data.to(at::kCPU);
        ptrx = ptr_x_data.data_ptr<int>();
        batch_x_size.push_back(x.size(0) * x.size(1));
    }

    if (ptr_y.has_value()) {
        ptr_y_data = ptr_y->to(at::kInt);
        TORCH_CHECK(ptr_y_data.is_contiguous(), " must be contiguous");
        AT_ASSERTM(ptr_y_data.dim() == 1, "ptr_y must be 1 dim");
        ptry = ptr_y_data.data_ptr<int>();
    } else {
        std::vector<int> y_vec = {0, static_cast<int>(y.size(0))};
        ptr_y_data = at::tensor(y_vec, at::kInt);
        ptr_y_data = x.device().is_cuda() ? ptr_y_data.to(at::kCUDA) : ptr_y_data.to(at::kCPU);
        ptry = ptr_y_data.data_ptr<int>();
    }
    float r = static_cast<float>(r_);
    auto x_res = at::full({y.size(0) * max_num_neighbors}, -1, at::TensorOptions().dtype(at::kInt).device(y.device()));
    auto y_res = at::full({y.size(0) * max_num_neighbors}, -1, at::TensorOptions().dtype(at::kInt).device(y.device()));

    int ret = 0;
    if (x.device().is_cuda()) {
        ret = xav::xpu::radius<float>(
                ctx,
                x.data_ptr<float>(),
                y.data_ptr<float>(),
                ptrx,
                ptry,
                x_res.data_ptr<int>(),
                y_res.data_ptr<int>(),
                batch_x_size.data(),
                x.size(0),
                y.size(0),
                batch_size,
                x.size(1),
                r,
                max_num_neighbors,
                num_workers,
                ignore_same_index);
    } else {
        ret = xav::cpu::radius<float>(
                ctx,
                x.data_ptr<float>(),
                y.data_ptr<float>(),
                ptrx,
                ptry,
                x_res.data_ptr<int>(),
                y_res.data_ptr<int>(),
                x.size(0),
                y.size(0),
                batch_size,
                x.size(1),
                r,
                max_num_neighbors,
                num_workers,
                ignore_same_index);
    }
    assert(ret == 0);
    auto mask = x_res != -1;
    return at::stack({y_res.masked_select(mask), x_res.masked_select(mask)}, 0);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("radius", &radius);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("radius", &radius);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("radius(Tensor x, Tensor y, Tensor? ptr_x, Tensor? ptr_y, float r, "
                                 "int max_num_neighbors, int num_workers, bool ignore_same_index=False) -> Tensor"));
}