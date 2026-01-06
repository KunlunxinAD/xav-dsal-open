#include "xpytorch.hpp"

at::Tensor smooth_cosine_loss_forward(
        at::Tensor emb_gt,
        at::Tensor offset_feat,
        double h_pix_,
        double w_pix_,
        int64_t ignore_label_) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    TORCH_CHECK(emb_gt.is_contiguous(), "emb_gt must be contiguous");
    TORCH_CHECK(offset_feat.is_contiguous(), "offset_feat must be contiguous");

    AT_ASSERTM(
            emb_gt.size(0) == offset_feat.size(0) && emb_gt.size(1) == offset_feat.size(1)
                    && emb_gt.size(2) == offset_feat.size(2),
            "emb_gt and offset_feat must have the same size");
    AT_ASSERTM(offset_feat.size(3) == 3, "offset_feat tensor must have size 3 at the last dim");
    AT_ASSERTM(offset_feat.dtype() == at::kFloat, "offset_feat tensor has to be of float32");

    float h_pix = static_cast<float>(h_pix_);
    float w_pix = static_cast<float>(w_pix_);
    int ignore_label = static_cast<int>(ignore_label_);

    const int batch_size = emb_gt.size(0);
    const int bev_h = emb_gt.size(1);
    const int bev_w = emb_gt.size(2);
    auto output = at::zeros({}, offset_feat.options().dtype(at::kFloat));

    auto kernel = xav::cpu::smooth_cosine_loss_forward;
    if (emb_gt.device().is_cuda()) {
        kernel = xav::xpu::smooth_cosine_loss_forward;
    }

    kernel(ctx,
           emb_gt.data_ptr<float>(),
           offset_feat.data_ptr<float>(),
           output.data_ptr<float>(),
           batch_size,
           bev_h,
           bev_w,
           h_pix,
           w_pix,
           ignore_label);

    return output;
}

void smooth_cosine_loss_backward(
        at::Tensor emb_gt,
        at::Tensor offset_feat,
        at::Tensor grad_offset_feat,
        at::Tensor grad_output,
        double h_pix_,
        double w_pix_,
        int64_t ignore_label_) {
    auto ctx = xmlir_rt::getXpuKernelContext();
    AT_ASSERTM(emb_gt.is_contiguous(), "emb_gt tensor has to be contiguous");
    AT_ASSERTM(offset_feat.is_contiguous(), "offset_feat tensor has to be contiguous");
    AT_ASSERTM(
            emb_gt.size(0) == offset_feat.size(0) && emb_gt.size(1) == offset_feat.size(1)
                    && emb_gt.size(2) == offset_feat.size(2),
            "emb_gt and offset_feat must have the same size");
    AT_ASSERTM(offset_feat.size(3) == 3, "offset_feat tensor must have size 3 at the last dim");
    AT_ASSERTM(offset_feat.dtype() == at::kFloat, "offset_feat tensor has to be of float32");

    float h_pix = static_cast<float>(h_pix_);
    float w_pix = static_cast<float>(w_pix_);
    int ignore_label = static_cast<int>(ignore_label_);
    const int batch_size = emb_gt.size(0);
    const int bev_h = emb_gt.size(1);
    const int bev_w = emb_gt.size(2);

    auto kernel = xav::cpu::smooth_cosine_loss_backward;
    if (emb_gt.device().is_cuda()) {
        kernel = xav::xpu::smooth_cosine_loss_backward;
    }

    kernel(ctx,
           emb_gt.data_ptr<float>(),
           offset_feat.data_ptr<float>(),
           grad_output.data_ptr<float>(),
           grad_offset_feat.data_ptr<float>(),
           batch_size,
           bev_h,
           bev_w,
           h_pix,
           w_pix,
           ignore_label);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("smooth_cosine_loss_forward", &smooth_cosine_loss_forward);
    m.impl("smooth_cosine_loss_backward", &smooth_cosine_loss_backward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("smooth_cosine_loss_forward", &smooth_cosine_loss_forward);
    m.impl("smooth_cosine_loss_backward", &smooth_cosine_loss_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def("smooth_cosine_loss_forward", &smooth_cosine_loss_forward);
    m.def(TORCH_SELECTIVE_SCHEMA(
            "smooth_cosine_loss_backward(Tensor emb_gt, Tensor offset_feat, Tensor grad_offset_feat, Tensor "
            "grad_output, float h_pix, float w_pix, int ignore_label) -> ()"));
}