#include <nncase/kernels/stackvm/tensor_ops.h>
#include "shape_infer.h"

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;


result<value_t> nncase::kernels::stackvm::conv2d_shape(value_t input, value_t weights, value_t padding, value_t stride, value_t dilation, [[maybe_unused]] value_t groups, value_t output, kernel_context&)
{
    try_dims(in_shape, input);
    try_dims(w_shape, weights);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_strides(strides, stride);
    try_strides(dilations, dilation);
    try_output(out_mem, output, dt_int64, dims_t{4});
    auto out_shape =
            conv2d_infer_shape(in_shape, w_shape,
                               strides_value, dilations, pads);
    for (int i = 0; i < 4; ++i) {
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];
    }
    KERNEL_FINISH;
}

size_t ComputeOutSize(int inputSize, int weightSize, const strides_t &strides, dims_t outPaddings, paddings_t paddings, const strides_t &dilations, int offset)
{
    return (strides[offset] * (inputSize - 1L))
           + outPaddings[offset]
           + (((weightSize - 1L)
               * dilations[offset]) + 1L) - paddings[offset].before - paddings[offset].after;
}

dims_t conv2d_transpose_infer_shape(gsl::span<const size_t> in_shape, gsl::span<const size_t> w_shape, const strides_t &strides, paddings_t paddings, const dims_t &outPadding, const strides_t &dilations, int group)
{
    auto iN = in_shape[0];
    auto iH = in_shape[2];
    auto iW = in_shape[3];
    auto oc = w_shape[0] * group;
    auto wH = w_shape[2];
    auto wW = w_shape[3];

    auto oh = ComputeOutSize(iH, wH, strides, outPadding, paddings, dilations, 0);
    auto ow = ComputeOutSize(iW, wW, strides, outPadding, paddings, dilations, 1);
    auto out_shape = dims_t{iN, oc, oh, ow};
    return out_shape;
}

result<value_t> nncase::kernels::stackvm::conv2d_transpose_shape(value_t input, value_t weights, value_t stride, value_t dilation, value_t padding, value_t output_padding, value_t groups, value_t output, kernel_context&)
{
    try_dims(input_shape, input);
    try_dims(weights_shape, weights);
    try_strides(strides_value, stride);
    try_paddings(pads, padding);
    try_dims(out_padding, output_padding);
    try_to_integer(groups_value, groups);
    try_strides(strides, stride);
    try_strides(dilations, dilation);

    auto out_shape = conv2d_transpose_infer_shape(input_shape, weights_shape, strides, pads, out_padding, dilations, groups_value);
    try_output(out_mem, output, dt_int64, dims_t{4});
    for (int i = 0; i < 4; ++i) {
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];
    }
    KERNEL_FINISH;
}

result<dims_t> to_dims(tensor shape)
{
    try_dims(shape_value, shape);
    return ok(shape_value);
}

result<value_t> nncase::kernels::stackvm::broadcast_shape(value_t inputs, value_t output, kernel_context&)
{
    try_tuple_input(tuple_mem, inputs);
    auto begin = inputs_tuple->fields().begin();
    auto out_shape = std::accumulate(std::next(begin), inputs_tuple->fields().end(), to_dims(begin->as<tensor>().unwrap()).unwrap(), [&](auto sum, auto field){
        auto shape = to_dims(field.template as<tensor>().unwrap()).unwrap();
        auto result = kernels::detail::get_binary_output_shape(
                shape, sum);

        return dims_t(result.begin(), result.end());
    });
    try_output(out_mem, output, dt_int64, dims_t{out_shape.size()});
    for (int i = 0; i < out_shape.size(); ++i) {
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];
    }

    KERNEL_FINISH;
}

result<value_t> nncase::kernels::stackvm::mat_mul_shape(value_t lhs, value_t rhs, value_t output, kernel_context&)
{
    try_dims(lhs_shape, lhs);
    try_dims(rhs_shape, rhs);
    try_var(out_shape,
            matmul_infer_shape(lhs_shape, rhs_shape));
    try_output(out_mem, output, dt_int64, dims_t{out_shape.size()});
    for (int i = 0; i < out_shape.size(); ++i) {
        OUT_CAST(int64_t, out_mem)[i] = out_shape[i];
    }
    KERNEL_FINISH;
}