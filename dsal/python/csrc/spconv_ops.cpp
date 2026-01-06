// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xpytorch.hpp"

#include "spconv_torch_utils.hpp"

template <unsigned NDim>
std::vector<torch::Tensor> getIndicePair(
        torch::Tensor indices,
        int64_t batchSize,
        std::vector<int64_t> outSpatialShape,
        std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding,
        int64_t _subM,
        int64_t _transpose) {
    bool subM = _subM != 0;
    bool transpose = _transpose != 0;
    auto numAct = indices.size(0);
    auto coorDim = indices.size(1) - 1;    // batchIdx + xyz
    TV_ASSERT_RT_ERR(!transpose, "not implement");
    TV_ASSERT_RT_ERR(NDim == coorDim, "error");
    TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(outPadding.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
    auto kernelVolume = kernelSize[0];
    for (int i = 1; i < kernelSize.size(); ++i) {
        kernelVolume *= kernelSize[i];
    }
    TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");

    auto ctx = xmlir_rt::getXpuKernelContext();

    auto outputVolume = outSpatialShape[0];
    for (int i = 1; i < outSpatialShape.size(); ++i) {
        outputVolume *= outSpatialShape[i];
    }
    torch::Tensor indicePairs
            = torch::full({kernelVolume, 2, numAct}, -1, torch::dtype(torch::kInt32).device(indices.device()));
    torch::Tensor indiceNum = torch::zeros({kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
    torch::Tensor gridOut
            = torch::full({batchSize * outputVolume}, -1, torch::dtype(torch::kInt32).device(indices.device()));
    int64_t numActOut = -1;

    std::vector<int> outSpatialShape32;
    std::vector<int> kernelSize32;
    std::vector<int> stride32;
    std::vector<int> padding32;
    std::vector<int> dilation32;

    auto indicePairUnique = torch::full(
            {indicePairs.numel() / 2 + 1},
            std::numeric_limits<int>::max(),
            torch::dtype(torch::kInt32).device(indices.device()));

    for (int i = 0; i < NDim; ++i) {
        outSpatialShape32.push_back(outSpatialShape[i]);
        kernelSize32.push_back(kernelSize[i]);
        if (subM) {
            stride32.push_back(1);
            padding32.push_back(kernelSize[i] / 2);
            dilation32.push_back(dilation[i]);
        } else {
            stride32.push_back(stride[i]);
            padding32.push_back(padding[i]);
            dilation32.push_back(dilation[i]);
        }
    }

    torch::Tensor tensor_numActOut = torch::zeros({1}, torch::dtype(torch::kInt64).device(indices.device()));

    if (subM) {
        auto kernel = xav::xpu::get_indice_pairs_subm<int, int, NDim>::eval;
        if (indices.device().type() == torch::kCPU) {
            kernel = xav::cpu::get_indice_pairs_subm<int, int, NDim>::eval;
        }
        int ret = kernel(
                ctx,
                indices.data_ptr<int>(),
                gridOut.data_ptr<int>(),
                indicePairs.data_ptr<int>(),
                indiceNum.data_ptr<int>(),
                numAct,
                kernelSize32,
                stride32,
                padding32,
                dilation32,
                outSpatialShape32);

        return {indices, indicePairs, indiceNum};
    } else {
        torch::Tensor outInds = torch::zeros(
                {numAct * kernelVolume, coorDim + 1}, torch::dtype(torch::kInt32).device(indices.device()));
        auto kernel = xav::xpu::get_indice_pairs_conv<int, int, NDim>::eval;
        if (indices.device().type() == torch::kCPU) {
            kernel = xav::cpu::get_indice_pairs_conv<int, int, NDim>::eval;
        }
        int ret = kernel(
                ctx,
                indices.data_ptr<int>(),
                outInds.data_ptr<int>(),
                gridOut.data_ptr<int>(),
                indicePairs.data_ptr<int>(),
                indiceNum.data_ptr<int>(),
                tensor_numActOut.data_ptr<int64_t>(),
                numAct,
                // torch2tv(gridOut).dim(0),
                batchSize * outputVolume,
                kernelSize32,
                stride32,
                padding32,
                dilation32,
                outSpatialShape32);

        numActOut = tensor_numActOut.item<int64_t>();   
        // in xpu implementation, numActOut is unique num of indice_unique, include INT_MAX
        if (indices.device().type() != torch::kCPU) {
            numActOut -= 1;
        }
        return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
    }
}

#define DECLARE_GETINDICEPAIR(NDim)                          \
    template std::vector<torch::Tensor> getIndicePair<NDim>( \
            torch::Tensor indices,                           \
            int64_t batchSize,                               \
            std::vector<int64_t> outSpatialShape,            \
            std::vector<int64_t> spatialShape,               \
            std::vector<int64_t> kernelSize,                 \
            std::vector<int64_t> stride,                     \
            std::vector<int64_t> padding,                    \
            std::vector<int64_t> dilation,                   \
            std::vector<int64_t> outPadding,                 \
            int64_t _subM,                                   \
            int64_t _transpose);

DECLARE_GETINDICEPAIR(2)
DECLARE_GETINDICEPAIR(3)
DECLARE_GETINDICEPAIR(4)

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("get_indice_pairs_2d", &getIndicePair<2>);
    m.impl("get_indice_pairs_3d", &getIndicePair<3>);
    m.impl("get_indice_pairs_4d", &getIndicePair<4>);
}
TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("get_indice_pairs_2d", &getIndicePair<2>);
    m.impl("get_indice_pairs_3d", &getIndicePair<3>);
    m.impl("get_indice_pairs_4d", &getIndicePair<4>);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "get_indice_pairs_2d(Tensor indices, int batchSize, "
            "int[] outSpatialShape, int[] spatialShape, int[] kernelSize, int[] stride, int[] padding, int[] dilation, "
            "int[] outPadding, int subM, int transpose) -> Tensor[]"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "get_indice_pairs_3d(Tensor indices, int batchSize, "
            "int[] outSpatialShape, int[] spatialShape, int[] kernelSize, int[] stride, int[] padding, int[] dilation, "
            "int[] outPadding, int subM, int transpose) -> Tensor[]"));

    m.def(TORCH_SELECTIVE_SCHEMA(
            "get_indice_pairs_4d(Tensor indices, int batchSize, "
            "int[] outSpatialShape, int[] spatialShape, int[] kernelSize, int[] stride, int[] padding, int[] dilation, "
            "int[] outPadding, int subM, int transpose) -> Tensor[]"));
}

template <unsigned NDim>
std::vector<torch::Tensor> getIndicePairPreGrid(
        torch::Tensor indices,
        torch::Tensor gridOut,
        int64_t batchSize,
        std::vector<int64_t> outSpatialShape,
        std::vector<int64_t> spatialShape,
        std::vector<int64_t> kernelSize,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        std::vector<int64_t> outPadding,
        int64_t _subM,
        int64_t _transpose) {
    bool subM = _subM != 0;
    bool transpose = _transpose != 0;
    auto numAct = indices.size(0);
    auto coorDim = indices.size(1) - 1;    // batchIdx + xyz
    TV_ASSERT_RT_ERR(!transpose, "not implement");
    TV_ASSERT_RT_ERR(NDim == coorDim, "error");
    TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(outPadding.size() == coorDim, "error");
    TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
    auto kernelVolume = kernelSize[0];
    for (int i = 1; i < kernelSize.size(); ++i) {
        kernelVolume *= kernelSize[i];
    }
    TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
    auto outputVolume = outSpatialShape[0];
    for (int i = 1; i < outSpatialShape.size(); ++i) {
        outputVolume *= outSpatialShape[i];
    }
    TV_ASSERT_INVALID_ARG(gridOut.numel() >= outputVolume * batchSize, "error");

    auto ctx = xmlir_rt::getXpuKernelContext();

    torch::Tensor indicePairs
            = torch::full({kernelVolume, 2, numAct}, -1, torch::dtype(torch::kInt32).device(indices.device()));
    torch::Tensor indiceNum = torch::zeros({kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));

    torch::Tensor tensor_numActOut = torch::zeros({1}, torch::dtype(torch::kInt32).device(indices.device()));

    int64_t numActOut = -1;
    std::vector<int> outSpatialShape32(NDim);
    std::vector<int> kernelSize32(NDim);
    std::vector<int> stride32(NDim);
    std::vector<int> padding32(NDim);
    std::vector<int> dilation32(NDim);

    auto indicePairUnique = torch::full(
            {indicePairs.numel() / 2 + 1},
            std::numeric_limits<int>::max(),
            torch::dtype(torch::kInt32).device(indices.device()));

    for (int i = 0; i < NDim; ++i) {
        outSpatialShape32.push_back(outSpatialShape[i]);
        kernelSize32.push_back(kernelSize[i]);
        if (subM) {
            stride32.push_back(1);
            padding32.push_back(kernelSize[i] / 2);
            dilation32.push_back(dilation[i]);
        } else {
            stride32.push_back(stride[i]);
            padding32.push_back(padding[i]);
            dilation32.push_back(dilation[i]);
        }
    }
    if (subM) {
        if (indices.device().type() == torch::kCPU) {
            int ret = xav::cpu::get_indice_pairs_subm<int, int, NDim>::eval(
                    ctx,
                    indices.data_ptr<int>(),
                    gridOut.data_ptr<int>(),
                    indicePairs.data_ptr<int>(),
                    indiceNum.data_ptr<int>(),
                    numAct,
                    kernelSize32,
                    stride32,
                    padding32,
                    dilation32,
                    outSpatialShape32);

            assert(ret == 0);
            gridOut.fill_(-1);    // why?
        } else {
            int ret = xav::xpu::get_indice_pairs_subm<int, int, NDim>::eval(
                    ctx,
                    indices.data_ptr<int>(),
                    gridOut.data_ptr<int>(),
                    indicePairs.data_ptr<int>(),
                    indiceNum.data_ptr<int>(),
                    numAct,
                    kernelSize32,
                    stride32,
                    padding32,
                    dilation32,
                    outSpatialShape32);
            assert(ret == 0);
        }

        return {indices, indicePairs, indiceNum};
    } else {
        torch::Tensor outInds = torch::zeros(
                {numAct * kernelVolume, coorDim + 1}, torch::dtype(torch::kInt32).device(indices.device()));
        if (indices.device().type() == torch::kCPU) {
            int ret = xav::cpu::get_indice_pairs_conv<int, int, NDim>::eval(
                    ctx,
                    indices.data_ptr<int>(),
                    outInds.data_ptr<int>(),
                    gridOut.data_ptr<int>(),
                    indicePairs.data_ptr<int>(),
                    indiceNum.data_ptr<int>(),
                    tensor_numActOut.data_ptr<int64_t>(),
                    numAct,
                    batchSize * outputVolume,
                    kernelSize32,
                    stride32,
                    padding32,
                    dilation32,
                    outSpatialShape32);

            assert(ret == 0);
            numActOut = tensor_numActOut.item<int64_t>() - 1;
            gridOut.fill_(-1);
        } else {
            int ret = xav::xpu::get_indice_pairs_conv<int, int, NDim>::eval(
                    ctx,
                    indices.data_ptr<int>(),
                    outInds.data_ptr<int>(),
                    gridOut.data_ptr<int>(),
                    indicePairs.data_ptr<int>(),
                    indiceNum.data_ptr<int>(),
                    tensor_numActOut.data_ptr<int64_t>(),
                    numAct,
                    batchSize * outputVolume,
                    kernelSize32,
                    stride32,
                    padding32,
                    dilation32,
                    outSpatialShape32);

            assert(ret == 0);
            numActOut = tensor_numActOut.item<int64_t>() - 1;
        }
        return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
    }
}

#define DECLARE_GETINDICEPAIRPREGRID(NDim)                          \
    template std::vector<torch::Tensor> getIndicePairPreGrid<NDim>( \
            torch::Tensor indices,                                  \
            torch::Tensor gridOut,                                  \
            int64_t batchSize,                                      \
            std::vector<int64_t> outSpatialShape,                   \
            std::vector<int64_t> spatialShape,                      \
            std::vector<int64_t> kernelSize,                        \
            std::vector<int64_t> stride,                            \
            std::vector<int64_t> padding,                           \
            std::vector<int64_t> dilation,                          \
            std::vector<int64_t> outPadding,                        \
            int64_t _subM,                                          \
            int64_t _transpose);

DECLARE_GETINDICEPAIRPREGRID(2);
DECLARE_GETINDICEPAIRPREGRID(3);

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("get_indice_pairs_grid_2d", &getIndicePairPreGrid<2>);
    m.impl("get_indice_pairs_grid_3d", &getIndicePairPreGrid<3>);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("get_indice_pairs_grid_2d", &getIndicePairPreGrid<2>);
    m.impl("get_indice_pairs_grid_3d", &getIndicePairPreGrid<3>);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
            "get_indice_pairs_grid_2d(Tensor indices, Tensor gridOut, int batchSize,"
            "int[] outSpatialShape, int[] spatialShape, int[] kernelSize, int[] stride, int[] padding, int[] dilation,"
            "int[] outPadding, int subM, int transpose) -> Tensor[]"));
    m.def(TORCH_SELECTIVE_SCHEMA(
            "get_indice_pairs_grid_3d(Tensor indices, Tensor gridOut, int batchSize,"
            "int[] outSpatialShape, int[] spatialShape, int[] kernelSize, int[] stride, int[] padding, int[] dilation,"
            "int[] outPadding, int subM, int transpose) -> Tensor[]"));
}

template <typename T>
torch::Tensor indiceConv(
        torch::Tensor features,
        torch::Tensor filters,
        torch::Tensor indicePairs,
        torch::Tensor indiceNum,
        int64_t numActOut,
        int64_t _inverse,
        int64_t _subM) {
    auto ctx = xmlir_rt::getXpuKernelContext();

    bool subM = _subM != 0;
    bool inverse = _inverse != 0;
    auto device = features.device().type();
    auto ndim = filters.dim() - 2;
    auto kernelVolume = indicePairs.size(0);
    auto numInPlanes = features.size(1);
    auto numOutPlanes = filters.size(ndim + 1);
    auto indicePairNumCpu = indiceNum.to({torch::kCPU});
    auto indicePairMaxSizeIter
            = std::max_element(indicePairNumCpu.data_ptr<int>(), indicePairNumCpu.data_ptr<int>() + kernelVolume);
    int indicePairMaxOffset = indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
    int indicePairMaxSize = *indicePairMaxSizeIter;

    /*if (_subM){
      std::vector<int> indicePairNumVec(indicePairNumCpu.data_ptr<int>(),
    indicePairNumCpu.data_ptr<int>() + kernelVolume);
      indicePairNumVec.erase(indicePairNumVec.begin() + indicePairMaxOffset);

      auto indicePairVecMaxSizeIter = std::max_element(
          indicePairNumVec.begin(), indicePairNumVec.end());
      indicePairMaxSize = *indicePairVecMaxSizeIter;
    }*/

    auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());
    // auto indicePairOptions =
    //     torch::TensorOptions().dtype(torch::kInt64).device(indicePairs.device());

    torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
    torch::Tensor inputBuffer = torch::zeros({indicePairMaxSize, numInPlanes}, options);
    torch::Tensor outputBuffer = torch::zeros({indicePairMaxSize, numOutPlanes}, options);
    filters = filters.view({-1, numInPlanes, numOutPlanes});
    if (subM) {    // the center index of subm conv don't need gather and scatter
                   // add.
        torch::mm_out(output, features, filters[indicePairMaxOffset]);
    }

    using xpu_scalar_t = typename xav::utils::map_torch_type<T>::type;

    double totalGatherTime = 0;
    double totalGEMMTime = 0;
    double totalSAddTime = 0;
    for (int i = 0; i < kernelVolume; ++i) {
        auto nHot = indicePairNumCpu.data_ptr<int>()[i];
        if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
            continue;
        }
        auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(), {nHot, numOutPlanes}, options);
        auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(), {nHot, numInPlanes}, options);

        auto gather_kernel = xav::xpu::sparse_gather<xpu_scalar_t, int>;
        if (device == torch::kCPU) {
            gather_kernel = xav::cpu::sparse_gather<xpu_scalar_t, int>;
        }

        int ret = gather_kernel(
                ctx,
                (xpu_scalar_t*)(tv::torch2tv<T>(inputBuffer).data()),
                (const xpu_scalar_t*)(tv::torch2tv<const T>(features).data()),
                tv::torch2tv<const int>(indicePairs).subview(i, inverse).data(),
                numInPlanes,
                nHot);

        assert(ret == 0);

        torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);

        auto scatter_kernel = xav::xpu::sparse_scatter_add<xpu_scalar_t, int>;
        if (device == torch::kCPU) {
            scatter_kernel = xav::cpu::sparse_scatter_add<xpu_scalar_t, int>;
        }

        ret = scatter_kernel(
                ctx,
                (xpu_scalar_t*)(tv::torch2tv<T>(output).data()),
                (const xpu_scalar_t*)(tv::torch2tv<const T>(outputBuffer).data()),
                tv::torch2tv<const int>(indicePairs).subview(i, !inverse).data(),
                numActOut,    // output.size(0)
                numOutPlanes,
                nHot);

        assert(ret == 0);
    }

    return output;
}

template <typename T>
std::vector<torch::Tensor> indiceConvBackward(
        torch::Tensor features,
        torch::Tensor filters,
        torch::Tensor outGrad,
        torch::Tensor indicePairs,
        torch::Tensor indiceNum,
        int64_t _inverse,
        int64_t _subM) {
    auto ctx = xmlir_rt::getXpuKernelContext();

    bool subM = _subM != 0;
    bool inverse = _inverse != 0;

    auto device = features.device().type();
    auto ndim = filters.dim() - 2;
    auto kernelVolume = indicePairs.size(0);
    auto numInPlanes = features.size(1);
    auto numOutPlanes = filters.size(ndim + 1);
    auto indicePairNumCpu = indiceNum.to({torch::kCPU});
    auto indicePairMaxSizeIter
            = std::max_element(indicePairNumCpu.data_ptr<int>(), indicePairNumCpu.data_ptr<int>() + kernelVolume);
    int indicePairMaxOffset = indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
    int indicePairMaxSize = *indicePairMaxSizeIter;
    auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());
    auto filterShape = filters.sizes();
    torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
    torch::Tensor filtersGrad = torch::zeros(filterShape, options);
    torch::Tensor inputBuffer = torch::zeros({indicePairMaxSize, numInPlanes}, options);
    torch::Tensor outputBuffer = torch::zeros({indicePairMaxSize, numOutPlanes}, options);

    filters = filters.view({-1, numInPlanes, numOutPlanes});
    filtersGrad = filtersGrad.view({-1, numInPlanes, numOutPlanes});
    if (subM) {
        auto filterGradSub = filtersGrad[indicePairMaxOffset];
        torch::mm_out(filterGradSub, features.t(), outGrad);
        torch::mm_out(inputGrad, outGrad, filters[indicePairMaxOffset].t());
    }

    using xpu_scalar_t = typename xav::utils::map_torch_type<T>::type;

    for (int i = 0; i < kernelVolume; ++i) {
        auto nHot = indicePairNumCpu.data_ptr<int>()[i];
        if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
            continue;
        }

        auto gather_kernel = xav::xpu::sparse_gather<xpu_scalar_t, int>;
        if (device == torch::kCPU) {
            gather_kernel = xav::cpu::sparse_gather<xpu_scalar_t, int>;
        }

        gather_kernel(
                ctx,
                (xpu_scalar_t*)(tv::torch2tv<T>(inputBuffer).data()),
                (const xpu_scalar_t*)(tv::torch2tv<const T>(features).data()),
                tv::torch2tv<const int>(indicePairs).subview(i, inverse).data(),
                numInPlanes,
                nHot);

        gather_kernel(
                ctx,
                (xpu_scalar_t*)(tv::torch2tv<T>(outputBuffer).data()),
                (const xpu_scalar_t*)(tv::torch2tv<const T>(outGrad).data()),
                tv::torch2tv<const int>(indicePairs).subview(i, !inverse).data(),
                numOutPlanes,
                nHot);

        auto filterGradSub = filtersGrad[i];
        auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(), {nHot, numOutPlanes}, options);
        auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(), {nHot, numInPlanes}, options);

        torch::mm_out(filterGradSub, inputBufferBlob.t(), outputBufferBlob);
        torch::mm_out(inputBufferBlob, outputBufferBlob, filters[i].t());

        auto scatter_kernel = xav::xpu::sparse_scatter_add<xpu_scalar_t, int>;
        if (device == torch::kCPU) {
            scatter_kernel = xav::cpu::sparse_scatter_add<xpu_scalar_t, int>;
        }

        scatter_kernel(
                ctx,
                (xpu_scalar_t*)(tv::torch2tv<T>(inputGrad).data()),
                (const xpu_scalar_t*)(tv::torch2tv<const T>(inputBuffer).data()),
                tv::torch2tv<const int>(indicePairs).subview(i, inverse).data(),
                inputGrad.size(0),
                numInPlanes,
                nHot);
    }
    return {inputGrad, filtersGrad.view(filterShape)};
}

#define DECLARE_INDICECONV(T)             \
    template torch::Tensor indiceConv<T>( \
            torch::Tensor features,       \
            torch::Tensor filters,        \
            torch::Tensor indicePairs,    \
            torch::Tensor indiceNum,      \
            int64_t numActOut,            \
            int64_t _inverse,             \
            int64_t _subM);

#define DECLARE_INDICECONV_BACKWARD(T)                         \
    template std::vector<torch::Tensor> indiceConvBackward<T>( \
            torch::Tensor features,                            \
            torch::Tensor filters,                             \
            torch::Tensor outGrad,                             \
            torch::Tensor indicePairs,                         \
            torch::Tensor indiceNum,                           \
            int64_t _inverse,                                  \
            int64_t _subM);

DECLARE_INDICECONV(float);
DECLARE_INDICECONV(at::Half);
DECLARE_INDICECONV_BACKWARD(float);
DECLARE_INDICECONV_BACKWARD(at::Half);

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("indice_conv_fp32", &indiceConv<float>);
    m.impl("indice_conv_backward_fp32", &indiceConvBackward<float>);
    m.impl("indice_conv_half", &indiceConv<at::Half>);
    m.impl("indice_conv_backward_half", &indiceConvBackward<at::Half>);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("indice_conv_fp32", &indiceConv<float>);
    m.impl("indice_conv_backward_fp32", &indiceConvBackward<float>);
    m.impl("indice_conv_half", &indiceConv<at::Half>);
    m.impl("indice_conv_backward_half", &indiceConvBackward<at::Half>);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("indice_conv_fp32(Tensor features, Tensor filters, "
                                 "Tensor indicePairs, Tensor indiceNum,"
                                 "int numActOut, int inverse, "
                                 "int subM) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA("indice_conv_backward_fp32(Tensor features, Tensor filters, "
                                 "Tensor outGrad, Tensor indicePairs, "
                                 "Tensor indiceNum, int inverse, "
                                 "int subM) -> Tensor[]"));
    m.def(TORCH_SELECTIVE_SCHEMA("indice_conv_half(Tensor features, Tensor filters, "
                                 "Tensor indicePairs, Tensor indiceNum,"
                                 "int numActOut, int inverse, "
                                 "int subM) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA("indice_conv_backward_half(Tensor features, Tensor filters, "
                                 "Tensor outGrad, Tensor indicePairs, "
                                 "Tensor indiceNum, int inverse, "
                                 "int subM) -> Tensor[]"));
}

#if 0
template <typename T>
torch::Tensor fusedIndiceConvBatchNorm(
        torch::Tensor features,
        torch::Tensor filters,
        torch::Tensor bias,
        torch::Tensor indicePairs,
        torch::Tensor indiceNum,
        int64_t numActOut,
        int64_t _inverse,
        int64_t _subM) {
    auto ctx = xmlir_rt::getXpuKernelContext();

    bool subM = _subM != 0;
    bool inverse = _inverse != 0;
    auto device = features.device().type();
    auto ndim = filters.dim() - 2;
    auto kernelVolume = indicePairs.size(0);
    auto numInPlanes = features.size(1);
    auto numOutPlanes = filters.size(ndim + 1);
    auto indicePairNumCpu = indiceNum.to({torch::kCPU});
    auto indicePairMaxSizeIter
            = std::max_element(indicePairNumCpu.data_ptr<int>(), indicePairNumCpu.data_ptr<int>() + kernelVolume);
    int indicePairMaxOffset = indicePairMaxSizeIter - indicePairNumCpu.data_ptr<int>();
    int indicePairMaxSize = *indicePairMaxSizeIter;

    auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());

    torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options).copy_(bias);
    torch::Tensor inputBuffer = torch::zeros({indicePairMaxSize, numInPlanes}, options);
    torch::Tensor outputBuffer = torch::zeros({indicePairMaxSize, numOutPlanes}, options);
    filters = filters.view({-1, numInPlanes, numOutPlanes});
    if (subM) {    // the center index of subm conv don't need gather and scatter
                   // add.
        torch::mm_out(output, features, filters[indicePairMaxOffset]);
    }
    double totalGatherTime = 0;
    double totalGEMMTime = 0;
    double totalSAddTime = 0;
    for (int i = 0; i < kernelVolume; ++i) {
        auto nHot = indicePairNumCpu.data_ptr<int>()[i];
        if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
            continue;
        }
        // auto timer = spconv::CudaContextTimer<>();
        auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr<T>(), {nHot, numOutPlanes}, options);
        auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr<T>(), {nHot, numInPlanes}, options);

        auto gather_kernel = xav::xpu::sparse_gather<T, int>;
        if (device == torch::kCPU) {
            gather_kernel = xav::cpu::sparse_gather<T, int>;
        } else {
            gather_kernel = xav::gpu::sparse_gather<T, int>;
        }
        gather_kernel(
                ctx,
                tv::torch2tv<T>(inputBuffer),
                tv::torch2tv<const T>(features),
                tv::torch2tv<const int>(indicePairs).subview(i, inverse),
                nHot);
        torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);

        auto scatter_kernel = xav::xpu::sparse_scatter_add<T, int>;
        if (device == torch::kCPU) {
            scatter_kernel = xav::cpu::sparse_scatter_add<T, int>;
        }
        scatter_kernel(
                ctx,
                tv::torch2tv<T>(output),
                tv::torch2tv<const T>(outputBuffer),
                tv::torch2tv<const int>(indicePairs).subview(i, !inverse),
                nHot,
                true);
    }
    return output;
}
#endif

// test gather and scatter_add kernel seperately
template <typename T>
void spconv_gather(
        at::Tensor& buffer,
        const at::Tensor& features,
        const at::Tensor& indices,
        int64_t num_planes,
        int64_t num_pairs) {
    auto ctx = xmlir_rt::getXpuKernelContext();

    using xpu_scalar_t = typename xav::utils::map_torch_type<T>::type;

    printf("Success in spconvGather! num_pairs: %d, num_planes: %d \n", num_pairs, num_planes);
    auto gather_kernel = xav::cpu::sparse_gather<xpu_scalar_t, int>;
    if (features.device().is_cuda()) {
        printf("Is cuda! call xpu kernel\n");
        gather_kernel = xav::xpu::sparse_gather<xpu_scalar_t, int>;
    }

    gather_kernel(
            ctx,
            (xpu_scalar_t*)(tv::torch2tv<T>(buffer).data()),
            (const xpu_scalar_t*)(tv::torch2tv<const T>(features).data()),
            (const int*)(tv::torch2tv<const int>(indices).data()),
            num_planes,
            num_pairs);
}

template <typename T>
void spconv_scatter_add(
        at::Tensor& out_features,
        const at::Tensor& buffer,
        const at::Tensor& indices,
        int64_t out_act_num,
        int64_t num_planes,
        int64_t num_pairs) {
    auto ctx = xmlir_rt::getXpuKernelContext();

    using xpu_scalar_t = typename xav::utils::map_torch_type<T>::type;

    printf("Success in spconvScatterAdd! out_act_num: %d, num_pairs: %d, num_planes: %d \n",
           out_act_num,
           num_pairs,
           num_planes);
    auto scatter_kernel = xav::cpu::sparse_scatter_add<xpu_scalar_t, int>;
    if (out_features.device().is_cuda()) {
        printf("Is cuda! call xpu kernel\n");
        scatter_kernel = xav::xpu::sparse_scatter_add<xpu_scalar_t, int>;
    }

    scatter_kernel(
            ctx,
            (xpu_scalar_t*)(tv::torch2tv<T>(out_features).data()),
            (const xpu_scalar_t*)(tv::torch2tv<const T>(buffer).data()),
            (const int*)(tv::torch2tv<const int>(indices).data()),
            out_act_num,
            num_planes,
            num_pairs);
}

#define DECLARE_GATHER(T)               \
    template void spconv_gather<T>(     \
            at::Tensor & buffer,        \
            const at::Tensor& features, \
            const at::Tensor& indices,  \
            int64_t num_planes,         \
            int64_t num_pairs);

#define DECLARE_SCATTER_ADD(T)             \
    template void spconv_scatter_add<T>(   \
            at::Tensor & out_features,     \
            const at::Tensor& buffer,      \
            const at::Tensor& indices,     \
            int64_t out_act_num,           \
            int64_t num_planes,            \
            int64_t num_pairs);

DECLARE_GATHER(float);
DECLARE_GATHER(at::Half);
DECLARE_SCATTER_ADD(float);
DECLARE_SCATTER_ADD(at::Half);

TORCH_LIBRARY_IMPL(xav_dsal, CUDA, m) {
    m.impl("spconv_gather_fp32", &spconv_gather<float>);
    m.impl("spconv_gather_fp16", &spconv_gather<at::Half>);
    m.impl("spconv_scatter_add_fp32", &spconv_scatter_add<float>);
    m.impl("spconv_scatter_add_fp16", &spconv_scatter_add<at::Half>);
}

TORCH_LIBRARY_IMPL(xav_dsal, CPU, m) {
    m.impl("spconv_gather_fp32", &spconv_gather<float>);
    m.impl("spconv_gather_fp16", &spconv_gather<at::Half>);
    m.impl("spconv_scatter_add_fp32", &spconv_scatter_add<float>);
    m.impl("spconv_scatter_add_fp16", &spconv_scatter_add<at::Half>);
}

TORCH_LIBRARY_FRAGMENT(xav_dsal, m) {
    m.def(TORCH_SELECTIVE_SCHEMA("spconv_gather_fp32(Tensor(a!) buffer, Tensor features,"
                                 "Tensor indices, int num_planes, int num_pairs) -> ()"));
    m.def(TORCH_SELECTIVE_SCHEMA("spconv_gather_fp16(Tensor(a!) buffer, Tensor features,"
                                 "Tensor indices, int num_planes, int num_pairs) -> ()"));
    m.def(TORCH_SELECTIVE_SCHEMA(
            "spconv_scatter_add_fp32(Tensor(a!) out_features, Tensor buffer,"
            "Tensor indices, int out_act_num, int num_planes, int num_pairs) -> ()"));
    m.def(TORCH_SELECTIVE_SCHEMA(
            "spconv_scatter_add_fp16(Tensor(a!) out_features, Tensor buffer,"
            "Tensor indices, int out_act_num, int num_planes, int num_pairs) -> ()"));
}