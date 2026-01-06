#include "xpytorch.hpp"

at::Tensor deformable_aggregation_forward(
        at::Tensor& _mc_ms_feat,
        at::Tensor& _spatial_shape,
        at::Tensor& _scale_start_index,
        at::Tensor& _sampling_location,
        at::Tensor& _weights,
        bool variant = false) {
    AT_ASSERT(_scale_start_index.numel() > 0, "scale_start_index can't be empty");
    int batch_size = -1;
    int num_cams = -1;
    int num_feat = -1;
    int num_embeds = -1;
    int num_scale = -1;
    int num_anchors = -1;
    int num_pts = -1;
    int num_groups = -1;
    if (variant) {
        batch_size = _mc_ms_feat.size(0);
        num_cams = _mc_ms_feat.size(1);
        int feat = _mc_ms_feat.size(2);
        num_feat = num_cams * feat;
        num_embeds = _mc_ms_feat.size(3);
        num_scale = _spatial_shape.size(0);
        num_pts = 1;
        num_anchors = _sampling_location.size(1);
        num_groups = _weights.size(4);

        _mc_ms_feat = _mc_ms_feat.reshape({batch_size, num_feat, num_embeds}).contiguous();
        _spatial_shape = _spatial_shape.unsqueeze(0)
                                 .expand({num_cams, num_scale, _spatial_shape.size(1)})
                                 .contiguous()
                                 .to(at::kInt);
        _sampling_location = _sampling_location.unsqueeze(2).contiguous().to(at::kFloat);
        _weights = _weights.unsqueeze(2).contiguous().to(at::kFloat);
        at::Tensor& first_start_idx = _scale_start_index;
        int group_len = num_feat / num_cams;
        auto group_base = at::arange(0, num_cams, 1, at::kInt) * group_len;
        int first_elem = _scale_start_index[0].item<int>();
        at::Tensor relative_offsets = first_start_idx - first_elem;
        _scale_start_index = group_base.unsqueeze(1).to("cuda") + relative_offsets.unsqueeze(0);
        _scale_start_index = _scale_start_index.contiguous().to(at::kInt);

    } else {
        batch_size = _mc_ms_feat.size(0);
        num_feat = _mc_ms_feat.size(1);
        num_embeds = _mc_ms_feat.size(2);
        num_cams = _spatial_shape.size(0);
        num_scale = _spatial_shape.size(1);
        num_anchors = _sampling_location.size(1);
        num_pts = _sampling_location.size(2);
        num_groups = _weights.size(5);
    }

    auto ctx = xmlir_rt::getXpuKernelContext();

    const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
    const int* spatial_shape = _spatial_shape.data_ptr<int>();
    const int* scale_start_index = _scale_start_index.data_ptr<int>();
    const float* sampling_location = _sampling_location.data_ptr<float>();
    const float* weights = _weights.data_ptr<float>();

    auto _output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());
    float* output = _output.data_ptr<float>();

    auto kernel = xav::cpu::deformable_aggregation;
    if (_mc_ms_feat.device().is_cuda()) {
        kernel = xav::xpu::deformable_aggregation;
    }

    kernel(ctx,
           mc_ms_feat,
           spatial_shape,
           scale_start_index,
           sampling_location,
           weights,
           output,
           batch_size,
           num_feat,
           num_embeds,
           num_cams,
           num_scale,
           num_anchors,
           num_pts,
           num_groups);

    return _output;
}

void deformable_aggregation_backward(
        at::Tensor& _mc_ms_feat,
        at::Tensor& _spatial_shape,
        at::Tensor& _scale_start_index,
        at::Tensor& _sampling_location,
        at::Tensor& _weights,
        at::Tensor& _grad_output,
        at::Tensor& _grad_mc_ms_feat,
        at::Tensor& _grad_sampling_location,
        at::Tensor& _grad_weights,
        bool variant = false) {
    int batch_size = -1;
    int num_feat = -1;
    int feat = -1;
    int num_embeds = -1;
    int num_cams = -1;
    int num_scale = -1;
    int num_anchors = -1;
    int num_pts = -1;
    int num_groups = -1;
    auto new_grad_mc_ms_feat = _grad_mc_ms_feat;
    auto new_grad_sampling_location = _grad_sampling_location;
    auto new_grad_weights = _grad_weights;
    if (variant) {
        batch_size = _mc_ms_feat.size(0);
        num_cams = _mc_ms_feat.size(1);
        feat = _mc_ms_feat.size(2);
        num_feat = num_cams * feat;
        num_embeds = _mc_ms_feat.size(3);
        num_scale = _spatial_shape.size(0);
        num_pts = 1;
        num_anchors = _sampling_location.size(1);
        num_groups = _weights.size(4);

        _mc_ms_feat = _mc_ms_feat.reshape({batch_size, num_feat, num_embeds}).contiguous();
        new_grad_mc_ms_feat = _grad_mc_ms_feat.reshape({batch_size, num_feat, num_embeds}).contiguous();
        _spatial_shape = _spatial_shape.unsqueeze(0)
                                 .expand({num_cams, num_scale, _spatial_shape.size(1)})
                                 .contiguous()
                                 .to(at::kInt);
        _sampling_location = _sampling_location.unsqueeze(2).contiguous().to(at::kFloat);
        new_grad_sampling_location = _grad_sampling_location.unsqueeze(2).contiguous().to(at::kFloat);
        _weights = _weights.unsqueeze(2).contiguous().to(at::kFloat);
        new_grad_weights = _grad_weights.unsqueeze(2).contiguous().to(at::kFloat);
        at::Tensor& first_start_idx = _scale_start_index;
        int group_len = num_feat / num_cams;
        auto group_base = at::arange(0, num_cams, 1, at::kInt) * group_len;
        int first_elem = _scale_start_index[0].item<int>();
        at::Tensor relative_offsets = first_start_idx - first_elem;
        _scale_start_index = group_base.unsqueeze(1).to("cuda") + relative_offsets.unsqueeze(0);
        _scale_start_index = _scale_start_index.contiguous().to(at::kInt);
    } else {
        batch_size = _mc_ms_feat.size(0);
        num_feat = _mc_ms_feat.size(1);
        num_embeds = _mc_ms_feat.size(2);
        num_cams = _spatial_shape.size(0);
        num_scale = _spatial_shape.size(1);
        num_anchors = _sampling_location.size(1);
        num_pts = _sampling_location.size(2);
        num_groups = _weights.size(5);
    }
    auto ctx = xmlir_rt::getXpuKernelContext();

    const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
    const int* spatial_shape = _spatial_shape.data_ptr<int>();
    const int* scale_start_index = _scale_start_index.data_ptr<int>();
    const float* sampling_location = _sampling_location.data_ptr<float>();
    const float* weights = _weights.data_ptr<float>();
    const float* grad_output = _grad_output.data_ptr<float>();

    float* grad_mc_ms_feat = new_grad_mc_ms_feat.data_ptr<float>();
    float* grad_sampling_location = new_grad_sampling_location.data_ptr<float>();
    float* grad_weights = new_grad_weights.data_ptr<float>();

    auto kernel = xav::cpu::deformable_aggregation_grad;
    if (_mc_ms_feat.device().is_cuda()) {
        kernel = xav::xpu::deformable_aggregation_grad;
    }

    kernel(ctx,
           mc_ms_feat,
           spatial_shape,
           scale_start_index,
           sampling_location,
           weights,
           grad_output,
           grad_mc_ms_feat,
           grad_sampling_location,
           grad_weights,
           batch_size,
           num_feat,
           num_embeds,
           num_cams,
           num_scale,
           num_anchors,
           num_pts,
           num_groups);

    if (variant) {
        new_grad_mc_ms_feat = new_grad_mc_ms_feat.reshape({batch_size, num_cams, feat, num_embeds}).contiguous();
        new_grad_sampling_location = new_grad_sampling_location.squeeze(2);
        new_grad_weights = new_grad_weights.squeeze(2);
        _grad_mc_ms_feat.copy_(new_grad_mc_ms_feat);
        _grad_sampling_location.copy_(new_grad_sampling_location);
        _grad_weights.copy_(new_grad_weights);
    }
    return;
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("deformable_aggregation_forward", &deformable_aggregation_forward);
    m.impl("deformable_aggregation_backward", &deformable_aggregation_backward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("deformable_aggregation_forward", &deformable_aggregation_forward);
    m.impl("deformable_aggregation_backward", &deformable_aggregation_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "deformable_aggregation_forward(Tensor mc_ms_feat, Tensor spatial_shape, Tensor scale_start_index, Tensor "
            "sampling_location, Tensor weights, bool variant = False) -> Tensor"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "deformable_aggregation_backward(Tensor mc_ms_feat, Tensor spatial_shape, Tensor scale_start_index, Tensor "
            "sampling_location, Tensor weights, Tensor grad_output, Tensor(a!) grad_mc_ms_feat, Tensor(b!) "
            "grad_sampling_location, Tensor(c!) grad_weights, bool variant = False) -> ()"));
}
