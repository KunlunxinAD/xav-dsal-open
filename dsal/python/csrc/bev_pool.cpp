#include "xpytorch.hpp"

/*
  Function: pillar pooling (forward, cuda)
  Args:
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    out              : output features, FloatTensor[b, c, h_out, w_out]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
*/
void bev_pool_v2_forward(
        const at::Tensor& _depth,
        const at::Tensor& _feat,
        at::Tensor& _out,
        const at::Tensor& _ranks_depth,
        const at::Tensor& _ranks_feat,
        const at::Tensor& _ranks_bev,
        const at::Tensor& _interval_lengths,
        const at::Tensor& _interval_starts) {
    int c = _feat.size(4);
    int n_intervals = _interval_lengths.size(0);
    // const at::cuda::OptionalCUDAGuard device_guard(device_of(_depth));
    const float* depth = _depth.data_ptr<float>();
    const float* feat = _feat.data_ptr<float>();
    const int* ranks_depth = _ranks_depth.data_ptr<int>();
    const int* ranks_feat = _ranks_feat.data_ptr<int>();
    const int* ranks_bev = _ranks_bev.data_ptr<int>();

    const int* interval_lengths = _interval_lengths.data_ptr<int>();
    const int* interval_starts = _interval_starts.data_ptr<int>();

    float* out = _out.data_ptr<float>();

    auto ctx = xmlir_rt::getXpuKernelContext();
    xav::xpu::bev_pool_v2(
            ctx,
            c,
            n_intervals,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            out);
}

/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : grad of output bev feature, FloatTensor[b, c, h_out, w_out]
    depth_grad       : grad of input depth, FloatTensor[n, d, h, w]
    feat_grad        : grad of input feature, FloatTensor[n, h, w, c]
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
*/
void bev_pool_v2_backward(
        const at::Tensor& _out_grad,
        at::Tensor& _depth_grad,
        at::Tensor& _feat_grad,
        const at::Tensor& _depth,
        const at::Tensor& _feat,
        const at::Tensor& _ranks_depth,
        const at::Tensor& _ranks_feat,
        const at::Tensor& _ranks_bev,
        const at::Tensor& _interval_lengths,
        const at::Tensor& _interval_starts) {
    int c = _out_grad.size(4);
    int n_intervals = _interval_lengths.size(0);
    // const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
    const float* out_grad = _out_grad.data_ptr<float>();
    float* depth_grad = _depth_grad.data_ptr<float>();
    float* feat_grad = _feat_grad.data_ptr<float>();
    const float* depth = _depth.data_ptr<float>();
    const float* feat = _feat.data_ptr<float>();
    const int* ranks_depth = _ranks_depth.data_ptr<int>();
    const int* ranks_feat = _ranks_feat.data_ptr<int>();
    const int* ranks_bev = _ranks_bev.data_ptr<int>();
    const int* interval_lengths = _interval_lengths.data_ptr<int>();
    const int* interval_starts = _interval_starts.data_ptr<int>();

    auto ctx = xmlir_rt::getXpuKernelContext();

    xav::xpu::bev_pool_v2_grad(
            ctx,
            c,
            n_intervals,
            out_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            depth_grad,
            feat_grad);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("bev_pool_v2_forward", &bev_pool_v2_forward);
    m.impl("bev_pool_v2_backward", &bev_pool_v2_backward);
}

#if 0
void bev_pool_v2_forward_impl(
        const at::Tensor& _depth,
        const at::Tensor& _feat,
        at::Tensor& _out,
        const at::Tensor& _ranks_depth,
        const at::Tensor& _ranks_feat,
        const at::Tensor& _ranks_bev,
        const at::Tensor& _interval_lengths,
        const at::Tensor& _interval_starts) {
    C10_LOG_API_USAGE_ONCE("xav_dsal::bev_pool_v2_forward_impl");
    static auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("xav_dsal::bev_pool_v2_forward", "")
                             .typed<decltype(bev_pool_v2_forward)>();
    op.call(_depth, _feat, _out, _ranks_depth, _ranks_feat, _ranks_bev, _interval_lengths, _interval_starts);
}

void bev_pool_v2_backward_impl(
        const at::Tensor& _out_grad,
        at::Tensor& _depth_grad,
        at::Tensor& _feat_grad,
        const at::Tensor& _depth,
        const at::Tensor& _feat,
        const at::Tensor& _ranks_depth,
        const at::Tensor& _ranks_feat,
        const at::Tensor& _ranks_bev,
        const at::Tensor& _interval_lengths,
        const at::Tensor& _interval_starts) {
    C10_LOG_API_USAGE_ONCE("xav_dsal::bev_pool_v2_backward_impl");
    static auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("xav_dsal::bev_pool_v2_backward", "")
                             .typed<decltype(bev_pool_v2_backward)>();
    op.call(_out_grad,
            _depth_grad,
            _feat_grad,
            _depth,
            _feat,
            _ranks_depth,
            _ranks_feat,
            _ranks_bev,
            _interval_lengths,
            _interval_starts);
}
#endif

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "bev_pool_v2_forward(Tensor depth, Tensor feat, Tensor(a!) out, "
            "Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev, Tensor interval_length, Tensor "
            "interval_starts) -> ()"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "bev_pool_v2_backward(Tensor out_grad, Tensor(a!) depth_grad, Tensor(b!) feat_grad, "
            "Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev, Tensor "
            "interbal_lengths, Tensor interval_starts) -> ()"));
}

