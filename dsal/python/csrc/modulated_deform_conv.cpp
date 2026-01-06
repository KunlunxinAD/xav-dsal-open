#include "xpytorch.hpp"

namespace xav {
namespace cpu {

//
// paramter , description        , shape
// ----     , ----               , ----
// input   , input feature      , batch * channels * height * width
// weight  , convolution weight , channels_out * channels_kernel * height * width
// bias    , convolusion bias   , 1 * channels_out * 1 * 1
// offset  , deform offset      , batch * deformable_group * kernel_h * kernel_w * 2 * height_out * width_out
// mask    , deform mask        , batch * deformable_group * kernel_h * kernel_w * height_out * width_out
// output  , output feature     , batch * channels_out * height_out * width_out
// ones    , temp               , --
// columns , temp               , --
//
void modulated_deform_conv_forward(
        at::Tensor _input,
        at::Tensor _weight,
        at::Tensor _bias,
        at::Tensor _ones,
        at::Tensor _offset,
        at::Tensor _mask,
        at::Tensor& _output,
        at::Tensor& _columns,
        int kernel_h,
        int kernel_w,
        const int stride_h,
        const int stride_w,
        const int pad_h,
        const int pad_w,
        const int dilation_h,
        const int dilation_w,
        const int group,
        const int deformable_group,
        const bool with_bias) {
    using at::Tensor;

    at::DeviceGuard guard(_input.device());

    const int batch = _input.size(0);
    const int channels = _input.size(1);
    const int height = _input.size(2);
    const int width = _input.size(3);

    const int channels_out = _weight.size(0);
    const int channels_kernel = _weight.size(1);
    const int kernel_h_ = _weight.size(2);
    const int kernel_w_ = _weight.size(3);

    auto input = _input;
    auto weight = _weight;
    auto bias = _bias;
    auto ones = _ones;
    auto offset = _offset;
    auto mask = _mask;
    auto output = _output;

    auto float_type = input.scalar_type();
    if (float_type != at::ScalarType::Float) {
        input = _input.to(at::kFloat);
        weight = _weight.to(at::kFloat);
        bias = _bias.to(at::kFloat);
        ones = _ones.to(at::kFloat);
        offset = _offset.to(at::kFloat);
        mask = _mask.to(at::kFloat);
        output = _output.to(at::kFloat);
    }

    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR(
                "Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
                kernel_h_,
                kernel_w,
                kernel_h_,
                kernel_w_);
    if (channels != channels_kernel * group)
        AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).", channels, channels_kernel * group);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 || ones.size(0) * ones.size(1) < height_out * width_out) {
        // Resize plane and fill with ones...
        ones = at::ones({height_out, width_out}, input.options());
    }

    // resize output
    output = output.view({batch, channels_out, height_out, width_out}).zero_();
    // resize temporary columns
    at::Tensor columns = at::zeros({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());

    output = output.view({output.size(0), group, output.size(1) / group, output.size(2), output.size(3)});

    auto ctx = xmlir_rt::getXpuKernelContext();
    for (int b = 0; b < batch; b++) {
        //
        // parameter   , shape
        // ------------------------------------------------------------
        // data_im     , batch * channels_im * height * width
        // data_offset , batch * deformable_group * kernel_h * kernel_w * 2 * height_col * width_col
        // data_mask   , batch * deformable_group * kernel_h * kernel_w * height_col * width_col
        // data_col    , channels_im * kernel_h * kernel_w * batch * height_col * width_col
        //
        const float* data_im = input[b].data_ptr<float>();
        const float* data_offset = offset[b].data_ptr<float>();
        const float* data_mask = mask[b].data_ptr<float>();
        float* data_col = columns.data_ptr<float>();

        int num_kernels = channels * 1 * height_out * width_out;
        const int channel_per_deformable_group = channels / deformable_group;

        xav::cpu::modulated_deformable_im2col_kernel<float>(
                ctx,
                num_kernels,
                data_im,
                data_offset,
                data_mask,
                height,
                width,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                channel_per_deformable_group,
                1,
                channels,
                deformable_group,
                height_out,
                width_out,
                data_col);

        // divide into group
        weight = weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3)});
        columns = columns.view({group, columns.size(0) / group, columns.size(1)});

        for (int g = 0; g < group; g++) {
            output[b][g] = output[b][g].flatten(1).addmm_(weight[g].flatten(1), columns[g]).view_as(output[b][g]);
        }

        weight = weight.view({weight.size(0) * weight.size(1), weight.size(2), weight.size(3), weight.size(4)});
        columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    }

    output = output.view({output.size(0), output.size(1) * output.size(2), output.size(3), output.size(4)});

    if (with_bias) {
        output += bias.view({1, bias.size(0), 1, 1});
    }
    if (float_type != at::ScalarType::Float) {
        output = output.view(_output.sizes());
        _output.copy_(output.to(float_type));
    }
}

//
// paramter    , description                 , shape
// ----        , ----                        , ----
// input       , input feature               , batch * channels * height * width
// weight      , convolution weight          , channels_out * channels_kernel * height * width
// bias        , convolusion bias            , 1 * channels_out * 1 * 1
// offset      , deform offset          , batch * deformable_group * kernel_h * kernel_w * 2 * height_out * width_out
// mask        , deform mask            , batch * deformable_group * kernel_h * kernel_w * height_out * width_out
// output      , output feature              , batch * channels_out * height_out * width_out
// grad_input  , input feature gradient      , ==>
// grad_weight , convolution weight gradient , ==>
// grad_bias   , convolution bias gradient   , ==>
// grad_offset , deform offset gradient      , ==>
// grad_mask   , deform mask gradient        , ==>
// grad_output , output gradient(input)      , ==>
// ones        , temp                        , --
// columns     , temp                        , --
//
void modulated_deform_conv_backward(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias,
        at::Tensor ones,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor columns,
        at::Tensor& grad_input,
        at::Tensor& grad_weight,
        at::Tensor& grad_bias,
        at::Tensor& grad_offset,
        at::Tensor& grad_mask,
        at::Tensor grad_output,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        int dilation_h,
        int dilation_w,
        int group,
        int deformable_group,
        const bool with_bias) {
    using at::Tensor;

    at::DeviceGuard guard(input.device());

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
        AT_ERROR(
                "Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
                kernel_h_,
                kernel_w,
                kernel_h_,
                kernel_w_);
    if (channels != channels_kernel * group)
        AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).", channels, channels_kernel * group);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (ones.ndimension() != 2 || ones.size(0) * ones.size(1) < height_out * width_out) {
        // Resize plane and fill with ones...
        ones = at::ones({height_out, width_out}, input.options());
    }

    grad_input = grad_input.view({batch, channels, height, width});
    columns = at::zeros({channels * kernel_h * kernel_w, height_out * width_out}, input.options());

    grad_output = grad_output.view(
            {grad_output.size(0), group, grad_output.size(1) / group, grad_output.size(2), grad_output.size(3)});

    const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;

    auto ctx = xmlir_rt::getXpuKernelContext();

    for (int b = 0; b < batch; b++) {
        // divide int group
        columns = columns.view({group, columns.size(0) / group, columns.size(1)});
        weight = weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3)});

        //
        // attention to the addmm_ last two parameter,
        // addmm beta = 0.0, alpha = 1.0
        // so, columns's current value is erased by the new value of weight^T @ grad_output
        // this is the right logic.
        //
        for (int g = 0; g < group; g++) {
            columns[g].addmm_(weight[g].flatten(1).transpose(0, 1), grad_output[b][g].flatten(1), 0.0f, 1.0f);
        }

        columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
        weight = weight.view({weight.size(0) * weight.size(1), weight.size(2), weight.size(3), weight.size(4)});

        const float* grad_col_ptr = columns.data_ptr<float>();
        const float* data_im_ptr = input[b].data_ptr<float>();
        const float* data_offset_ptr = offset[b].data_ptr<float>();
        const float* data_mask_ptr = mask[b].data_ptr<float>();
        float* grad_offset_ptr = grad_offset[b].data_ptr<float>();
        float* grad_mask_ptr = grad_mask[b].data_ptr<float>();

        {
            //
            // parameter          , shape
            // ----               , ----
            // data_col(grad_col) , channels_im * kernel_h * kernel_w * batch * height_col * width_col
            // data_im            , batch * channels_im * height * width
            // data_offset        , batch * deformable_group * kernel_h * kernel_w * 2 * height_col * width_col
            // data_mask          , batch * deformable_group * kernel_h * kernel_w * height_col * width_col
            // grad_offset        , batch * deformable_group * kernel_h * kernel_w * 2 * height_col * width_col
            // grad_mask          , batch * deformable_group * kernel_h * kernel_w * height_col * width_col
            //
            const int num_kernels = 1 * height_out * width_out * 2 * kernel_h * kernel_w * deformable_group;
            const int offset_channels = 2 * kernel_h * kernel_w * deformable_group;

            xav::cpu::modulated_deformable_col2im_coord_kernel<float>(
                    ctx,
                    num_kernels,
                    grad_col_ptr,
                    data_im_ptr,
                    data_offset_ptr,
                    data_mask_ptr,
                    channels,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    channel_per_deformable_group,
                    batch,
                    offset_channels,
                    deformable_group,
                    height_out,
                    width_out,
                    grad_offset_ptr,
                    grad_mask_ptr);
        }

        {
            //
            // parameter          , shape
            // ----               , ----
            // data_col(grad_col) , channels_im * kernel_h * kernel_w * batch * height_col * width_col
            // grad_im            , batch * channels_im * height * width
            // data_offset        , batch * deformable_group * kernel_h * kernel_w * 2 * height_col * width_col
            // data_mask          , batch * deformable_group * kernel_h * kernel_w * height_col * width_col
            //
            const int num_kernels = channels * kernel_h * kernel_w * 1 * height_out * width_out;
            float* grad_im_ptr = grad_input[b].data_ptr<float>();

            xav::cpu::modulated_deformable_col2im_kernel<float>(
                    ctx,
                    num_kernels,
                    grad_col_ptr,
                    data_offset_ptr,
                    data_mask_ptr,
                    channels,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    channel_per_deformable_group,
                    batch,
                    deformable_group,
                    height_out,
                    width_out,
                    grad_im_ptr);
        }

        // gradient w.r.t. weight, dWeight should accumulate across the batch and
        // group
        {
            int num_kernels = channels * 1 * height_out * width_out;
            float* data_col_ptr = columns.data_ptr<float>();

            xav::cpu::modulated_deformable_im2col_kernel<float>(
                    ctx,
                    num_kernels,
                    data_im_ptr,
                    data_offset_ptr,
                    data_mask_ptr,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    channel_per_deformable_group,
                    1,
                    channels,
                    deformable_group,
                    height_out,
                    width_out,
                    data_col_ptr);
        }

        columns = columns.view({group, columns.size(0) / group, columns.size(1)});
        grad_weight = grad_weight.view(
                {group, grad_weight.size(0) / group, grad_weight.size(1), grad_weight.size(2), grad_weight.size(3)});

        if (with_bias)
            grad_bias = grad_bias.view({group, grad_bias.size(0) / group});

        for (int g = 0; g < group; g++) {
            grad_weight[g] = grad_weight[g]
                                     .flatten(1)
                                     .addmm_(grad_output[b][g].flatten(1), columns[g].transpose(0, 1))
                                     .view_as(grad_weight[g]);
            if (with_bias) {
                grad_bias[g]
                        = grad_bias[g].view({-1, 1}).addmm_(grad_output[b][g].flatten(1), ones.view({-1, 1})).view(-1);
            }
        }

        columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
        grad_weight = grad_weight.view(
                {grad_weight.size(0) * grad_weight.size(1),
                 grad_weight.size(2),
                 grad_weight.size(3),
                 grad_weight.size(4)});
        if (with_bias)
            grad_bias = grad_bias.view({grad_bias.size(0) * grad_bias.size(1)});
    }
    grad_output = grad_output.view(
            {grad_output.size(0) * grad_output.size(1), grad_output.size(2), grad_output.size(3), grad_output.size(4)});
}

}    // namespace cpu
}    // namespace xav

namespace xav {
namespace xpu {

//
// parameter , description        , shape
// ----------------------------------------------------------------------------------------------------
// input     , input              , batch * channels * height * width
// weight    , convolution weight , channels_out * channels_kernel * kernel_h * kernel_w
// offset    , deform_conv offset , batch * deform_group * kernel_h * kernel_w * 2 * height_out * width_out
// mask      , deform_conv mask   , batch * deform_group * kernel_h * kernel_w * height_out * width_out
// output    , output             , batch * channels_out * height_out * width_out
// ones      , temp               , --
// columns   , temp               , --
//
void modulated_deform_conv_forward(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias,
        at::Tensor ones,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor& output,
        at::Tensor& columns,
        int kernel_h,
        int kernel_w,
        const int stride_h,
        const int stride_w,
        const int pad_h,
        const int pad_w,
        const int dilation_h,
        const int dilation_w,
        const int group,
        const int deformable_group,
        const bool with_bias) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    auto float_type = input.scalar_type();
    AT_ASSERTM(
            float_type == at::ScalarType::Float || float_type == at::ScalarType::Half,
            "type only support float32 & float16");

    AT_DISPATCH_FLOAT_TWO_TYPES(input.scalar_type(), "modulated_deform_conv_forward", [&] {
        const scalar_t* input_data = reinterpret_cast<scalar_t*>(input.data_ptr<scalar_ptr_t>());
        const scalar_t* weight_data = reinterpret_cast<scalar_t*>(weight.data_ptr<scalar_ptr_t>());
        const scalar_t* offset_data = reinterpret_cast<scalar_t*>(offset.data_ptr<scalar_ptr_t>());
        const scalar_t* mask_data = reinterpret_cast<scalar_t*>(mask.data_ptr<scalar_ptr_t>());
        scalar_t* output_data = reinterpret_cast<scalar_t*>(output.data_ptr<scalar_ptr_t>());

        auto ctx = xmlir_rt::getXpuKernelContext();

        //
        // parameter , description        , shape
        // ----------------------------------------------------------------------------------------------------
        // input     , input              , batch * channels * height * width
        // weight    , convolution weight , channels_out * channels_kernel * kernel_h * kernel_w
        // offset    , deform_conv offset , batch * deform_group * kernel_h * kernel_w * 2 * height_out * width_out
        // mask      , deform_conv mask   , batch * deform_group * kernel_h * kernel_w * height_out * width_out
        // output    , output             , batch * channels_out * height_out * width_out
        //
        xav::xpu::deformable_conv<scalar_t, scalar_t, scalar_t, gemm_t>(
                ctx,
                input_data,
                weight_data,
                offset_data,
                mask_data,
                output_data,
                batch,
                channels,
                height,
                width,
                channels_out,
                {kernel_h, kernel_w},
                {stride_h, stride_w},
                {pad_h, pad_w},
                {dilation_h, dilation_w},
                group,
                deformable_group,
                nullptr,
                nullptr,
                nullptr,
                /*is_nchw=*/true);
    });
}

//
// paramter    , description                     , shape
// ----        , ----                            , ----
// input       , input feature                   , batch * channels * height * width
// weight      , convolution weight              , channels_out * channels_kernel * height * width
// bias        , convolusion bias                , 1 * channels_out * 1 * 1
// offset      , deform offset                   , batch * deformable_group * kernel_h * kernel_w * 2 * height_out *
// width_out mask        , deform mask                     , batch * deformable_group * kernel_h * kernel_w * height_out
// * width_out output      , output feature                  , batch * channels_out * height_out * width_out grad_input
// , input feature gradient          , ==> grad_weight , convolution weight gradient     , ==> grad_bias   , convolution
// bias gradient       , ==> grad_offset , deform offset gradient          , ==> grad_mask   , deform mask gradient ,
// ==> grad_output , output feature gradient(input ) , ==> ones        , temp                            , -- columns ,
// temp                            , --
//

void modulated_deform_conv_backward(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias,
        at::Tensor ones,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor columns,
        at::Tensor& grad_input,
        at::Tensor& grad_weight,
        at::Tensor& grad_bias,
        at::Tensor& grad_offset,
        at::Tensor& grad_mask,
        at::Tensor grad_output,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        int dilation_h,
        int dilation_w,
        int group,
        int deformable_group,
        const bool with_bias) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    const int channels_out = grad_output.size(1);

    auto float_type = input.scalar_type();
    AT_ASSERTM(
            float_type == at::ScalarType::Float || float_type == at::ScalarType::Half,
            "type only support float32 & float16");

    AT_DISPATCH_FLOAT_TWO_TYPES(input.scalar_type(), "modulated_deform_conv_backward", [&] {
        const scalar_t* input_data = reinterpret_cast<scalar_t*>(input.data_ptr<scalar_ptr_t>());
        const scalar_t* weight_data = reinterpret_cast<scalar_t*>(weight.data_ptr<scalar_ptr_t>());
        const scalar_t* offset_data = reinterpret_cast<scalar_t*>(offset.data_ptr<scalar_ptr_t>());
        const scalar_t* mask_data = reinterpret_cast<scalar_t*>(mask.data_ptr<scalar_ptr_t>());

        const scalar_t* grad_output_data = reinterpret_cast<scalar_t*>(grad_output.data_ptr<scalar_ptr_t>());
        scalar_t* grad_input_data = reinterpret_cast<scalar_t*>(grad_input.data_ptr<scalar_ptr_t>());
        scalar_t* grad_weight_data = reinterpret_cast<scalar_t*>(grad_weight.data_ptr<scalar_ptr_t>());
        scalar_t* grad_offset_data = reinterpret_cast<scalar_t*>(grad_offset.data_ptr<scalar_ptr_t>());
        scalar_t* grad_mask_data = reinterpret_cast<scalar_t*>(grad_mask.data_ptr<scalar_ptr_t>());

        auto ctx = xmlir_rt::getXpuKernelContext();

        xav::xpu::deformable_conv_grad<scalar_t, scalar_t, scalar_t, gemm_t>(
                ctx,
                input_data,
                weight_data,
                offset_data,
                mask_data,
                grad_output_data,
                grad_input_data,
                grad_weight_data,
                grad_offset_data,
                grad_mask_data,
                batch,
                channels,
                height,
                width,
                channels_out,
                {kernel_h, kernel_w},
                {stride_h, stride_w},
                {pad_h, pad_w},
                {dilation_h, dilation_w},
                group,
                deformable_group,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                /*is_nchw=*/true);
    });
}

}    // namespace xpu
}    // namespace xav

//
// parameter , description        , shape
// ----------------------------------------------------------------------------------------------------
// input   , input              , batch * channels * height * width
// weight  , convolution weight , channels_out * channels_kernel * kernel_h * kernel_w
// offset  , deform_conv offset , batch * deform_group * kernel_h * kernel_w * 2 * height_out * width_out
// mask    , deform_conv mask   , batch * deform_group * kernel_h * kernel_w * height_out * width_out
// output  , output             , batch * channels_out * height_out * width_out
// ones    , temp               , --
// columns , temp               , --
//
void modulated_deform_conv_forward(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias,
        at::Tensor ones,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor& output,
        at::Tensor& columns,
        int64_t kernel_h,
        int64_t kernel_w,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dilation_h,
        int64_t dilation_w,
        int64_t group,
        int64_t deformable_group,
        bool with_bias) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w) {
        AT_ERROR(
                "Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
                kernel_h_,
                kernel_w,
                kernel_h_,
                kernel_w_);
    }

    if (channels != channels_kernel * group) {
        AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).", channels, channels_kernel * group);
    }

    auto func = xav::cpu::modulated_deform_conv_forward;

    if (input.device().is_cuda()) {
        if (with_bias) {
            AT_ERROR("Modulated_deform_conv with bias is not supperted.");
        }

        func = xav::xpu::modulated_deform_conv_forward;
    }

    func(input,
         weight,
         bias,
         ones,
         offset,
         mask,
         output,
         columns,
         kernel_h,
         kernel_w,
         stride_h,
         stride_w,
         pad_h,
         pad_w,
         dilation_h,
         dilation_w,
         group,
         deformable_group,
         with_bias);
}

void modulated_deform_conv_backward(
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias,
        at::Tensor ones,
        at::Tensor offset,
        at::Tensor mask,
        at::Tensor columns,
        at::Tensor& grad_input,
        at::Tensor& grad_weight,
        at::Tensor& grad_bias,
        at::Tensor& grad_offset,
        at::Tensor& grad_mask,
        at::Tensor grad_output,
        int64_t kernel_h,
        int64_t kernel_w,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dilation_h,
        int64_t dilation_w,
        int64_t group,
        int64_t deformable_group,
        bool with_bias) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);
    if (kernel_h_ != kernel_h || kernel_w_ != kernel_w) {
        AT_ERROR(
                "Input shape and kernel shape won't match: (%d x %d vs %d x %d).",
                kernel_h_,
                kernel_w,
                kernel_h_,
                kernel_w_);
    }

    if (channels != channels_kernel * group) {
        AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).", channels, channels_kernel * group);
    }

    auto func = xav::cpu::modulated_deform_conv_backward;

    if (input.device().is_cuda()) {
        if (with_bias) {
            AT_ERROR("Modulated_deform_conv with bias is not supperted.");
        }

        func = xav::xpu::modulated_deform_conv_backward;
    }

    func(input,
         weight,
         bias,
         ones,
         offset,
         mask,
         columns,
         grad_input,
         grad_weight,
         grad_bias,
         grad_offset,
         grad_mask,
         grad_output,
         kernel_h,
         kernel_w,
         stride_h,
         stride_w,
         pad_h,
         pad_w,
         dilation_h,
         dilation_w,
         group,
         deformable_group,
         with_bias);
}

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("modulated_deform_conv_forward", &modulated_deform_conv_forward);
    m.impl("modulated_deform_conv_backward", &modulated_deform_conv_backward);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("modulated_deform_conv_forward", &modulated_deform_conv_forward);
    m.impl("modulated_deform_conv_backward", &modulated_deform_conv_backward);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "modulated_deform_conv_forward( Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset, "
            "Tensor mask, Tensor(a!) output, Tensor(b!) columns, int kernel_h, int kernel_w, int stride_h, int "
            "stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int group, int deformable_group, bool "
            "with_bias) ->() "));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "modulated_deform_conv_backward( Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset, "
            "Tensor mask, Tensor columns, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias, "
            "Tensor(d!) grad_offset, Tensor(e!) grad_mask, Tensor grad_output, int kernel_h, int kernel_w, int "
            "stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int group, int "
            "deformable_group, bool with_bias) -> ()"));
}
