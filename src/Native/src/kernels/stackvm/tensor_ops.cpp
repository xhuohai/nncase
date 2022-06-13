/* This file is generated by tools/stackvm_gen/IsaGen at 06/01/2022 17:14:34
 * +08:00.
 *
 * Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nncase/kernels/stackvm/ref_ops.h>
#include <nncase/kernels/stackvm/opt_ops.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include "shape_infer.h"

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<value_t> nncase::kernels::stackvm::batch_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t scale, [[maybe_unused]] value_t bias, [[maybe_unused]] value_t input_mean,
    [[maybe_unused]] value_t input_var, [[maybe_unused]] value_t epsilon, [[maybe_unused]] value_t momentum, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::clamp([[maybe_unused]] value_t input, [[maybe_unused]] value_t min,
                                                [[maybe_unused]] value_t max, [[maybe_unused]] value_t output,
                                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::compare([[maybe_unused]] compare_op_t compare_op,
                                                  [[maybe_unused]] value_t lhs, [[maybe_unused]] value_t rhs,
                                                  [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::conv2d_transpose(
    [[maybe_unused]] pad_mode_t pad_mode, [[maybe_unused]] value_t input, [[maybe_unused]] value_t weights, [[maybe_unused]] value_t bias,
    [[maybe_unused]] value_t output_shape, [[maybe_unused]] value_t stride, [[maybe_unused]] value_t padding,
    [[maybe_unused]] value_t output_padding, [[maybe_unused]] value_t dilation, [[maybe_unused]] value_t groups,
    [[maybe_unused]] value_t fused_clamp, [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::expand([[maybe_unused]] value_t input, [[maybe_unused]] value_t shape,
                                                 [[maybe_unused]] value_t output,
                                                 [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::flatten([[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
                                                  [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::gather(value_t input, value_t axis,
                                                 value_t index, value_t output,
                                                 kernel_context &context) {
    try_input(input_mem, input);
    try_integer_input(index_mem, index);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_positive_axis(axis_value, axis, input_tensor);
    auto out_shape = gather_infer_shape(input_tensor->shape(), index_tensor->shape(), axis_value);
    try_output(out_mem, output, dtype, out_shape);
    CONTIGUOUS_KERNEL(gather, input_tensor, 
                      typecode, input_mem, out_mem,
                      input_tensor->shape(), output_tensor->shape(),
                      input_tensor->strides(), output_tensor->strides(),
                      index_mem, index_tensor->shape(), axis_value, context);

    return ok(output);
}

result<value_t> nncase::kernels::stackvm::gather_nd(value_t input,
                                                    value_t batch_dims,
                                                    value_t index,
                                                    value_t output,
                                                    kernel_context &context){
    try_input(input_mem, input);
    try_input(index_mem, index);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_to_scalar(batch_dims_value, batch_dims, int64_t);
    auto out_shape = gather_nd_infer_shape(input_tensor->shape(), index_tensor->shape(), batch_dims_value);
    try_output(out_mem, output, dtype, out_shape);
    auto indices = reinterpret_cast<const int64_t*>(index_mem);
    CONTIGUOUS_KERNEL(gather_nd, input_tensor,
                      typecode, input_mem, out_mem,
                      input_tensor->shape(), output_tensor->shape(),
                      input_tensor->strides(), output_tensor->strides(),
                      indices, index_tensor->shape(), batch_dims_value, context);
    return ok(output);
}

result<value_t> nncase::kernels::stackvm::get_item([[maybe_unused]] value_t input, [[maybe_unused]] value_t index,
                                                   [[maybe_unused]] value_t output,
                                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::hard_sigmoid([[maybe_unused]] value_t input, [[maybe_unused]] value_t alpha,
                                       [[maybe_unused]] value_t beta, [[maybe_unused]] value_t output,
                                       [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::instance_normalization(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t scale, [[maybe_unused]] value_t bias, [[maybe_unused]] value_t epsilon, [[maybe_unused]] value_t output,
    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::l2_normalization([[maybe_unused]] value_t input, [[maybe_unused]] value_t output,
                                           [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::log_softmax([[maybe_unused]] value_t input,
                                                      [[maybe_unused]] value_t axis,
                                                      [[maybe_unused]] value_t output,
                                                      [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::lp_normalization([[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
                                           [[maybe_unused]] value_t p, [[maybe_unused]] value_t output,
                                           [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::lrn([[maybe_unused]] value_t input, [[maybe_unused]] value_t alpha,
                                              [[maybe_unused]] value_t beta, [[maybe_unused]] value_t bias,
                                              [[maybe_unused]] value_t size, [[maybe_unused]] value_t output,
                                              [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::lstm(
    [[maybe_unused]] lstmdirection_t direction, [[maybe_unused]] lstmlayout_t layout,
    [[maybe_unused]] std::vector<std::string> activations, [[maybe_unused]] value_t x, [[maybe_unused]] value_t w, [[maybe_unused]] value_t r,
    [[maybe_unused]] value_t b, [[maybe_unused]] value_t sequence_lens, [[maybe_unused]] value_t initial_h, [[maybe_unused]] value_t initial_c,
    [[maybe_unused]] value_t p, [[maybe_unused]] value_t activation_alpha, [[maybe_unused]] value_t activation_beta, [[maybe_unused]] value_t clip,
    [[maybe_unused]] value_t hidden_size, [[maybe_unused]] value_t input_forget, [[maybe_unused]] value_t output_size,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::normal([[maybe_unused]] typecode_t type, [[maybe_unused]] value_t mean,
                                                 [[maybe_unused]] value_t scale, [[maybe_unused]] value_t seed,
                                                 [[maybe_unused]] value_t shape, [[maybe_unused]] value_t output,
                                                 [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::normal_like(
    [[maybe_unused]] typecode_t type, [[maybe_unused]] value_t input, [[maybe_unused]] value_t mean, [[maybe_unused]] value_t scale, [[maybe_unused]] value_t seed,
    [[maybe_unused]] value_t shape, [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}


result<value_t> nncase::kernels::stackvm::prod([[maybe_unused]] value_t input, [[maybe_unused]] value_t output,
                                               [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::quant_param_of([[maybe_unused]] quant_mode_t quant_mode, [[maybe_unused]] value_t range,
                                         [[maybe_unused]] value_t bits, [[maybe_unused]] value_t output,
                                         [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::range([[maybe_unused]] value_t begin, [[maybe_unused]] value_t end,
                                                [[maybe_unused]] value_t step, [[maybe_unused]] value_t output,
                                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::range_of([[maybe_unused]] value_t input,
                                                   [[maybe_unused]] value_t output,
                                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}


result<value_t> nncase::kernels::stackvm::relu6([[maybe_unused]] value_t input, [[maybe_unused]] value_t output,
                                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::require( [[maybe_unused]] std::string message,
                                                  [[maybe_unused]] value_t predicate,
                                                  [[maybe_unused]] value_t value, [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::reshape([[maybe_unused]] value_t input, [[maybe_unused]] value_t shape,
                                                  [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::reverse_sequence(
    [[maybe_unused]] value_t input, [[maybe_unused]] value_t seq_lens, [[maybe_unused]] value_t batch_axis, [[maybe_unused]] value_t time_axis,
    [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::select([[maybe_unused]] value_t predicate,
                                                 [[maybe_unused]] value_t true_value,
                                                 [[maybe_unused]] value_t false_value,
                                                 [[maybe_unused]] value_t output,
                                                 [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::selu([[maybe_unused]] value_t input, [[maybe_unused]] value_t alpha,
                                               [[maybe_unused]] value_t gamma, [[maybe_unused]] value_t output,
                                               [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::shape_of([[maybe_unused]] value_t input,
                                                   [[maybe_unused]] value_t output,
                                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::size_of([[maybe_unused]] value_t input, [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}


result<value_t> nncase::kernels::stackvm::softmax([[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
                                                  [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::softplus([[maybe_unused]] value_t input,
                                                   [[maybe_unused]] value_t output,
                                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::softsign([[maybe_unused]] value_t input,
                                                   [[maybe_unused]] value_t output,
                                                   [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t>
nncase::kernels::stackvm::space_to_batch([[maybe_unused]] value_t input, [[maybe_unused]] value_t block_shape,
                                         [[maybe_unused]] value_t paddings, [[maybe_unused]] value_t output,
                                         [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::split([[maybe_unused]] value_t input, [[maybe_unused]] value_t axis,
                                                [[maybe_unused]] value_t sections,
                                                [[maybe_unused]] value_t output,
                                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::squeeze([[maybe_unused]] value_t input, [[maybe_unused]] value_t dim,
                                                  [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::stack([[maybe_unused]] value_t inputs, [[maybe_unused]] value_t axis,
                                                [[maybe_unused]] value_t output,
                                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::tile([[maybe_unused]] value_t input, [[maybe_unused]] value_t repeats,
                                               [[maybe_unused]] value_t output,
                                               [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::uniform([[maybe_unused]] typecode_t type, [[maybe_unused]] value_t high,
                                                  [[maybe_unused]] value_t low, [[maybe_unused]] value_t seed,
                                                  [[maybe_unused]] value_t shape, [[maybe_unused]] value_t output,
                                                  [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::uniform_like(
    [[maybe_unused]] typecode_t type, [[maybe_unused]] value_t input, [[maybe_unused]] value_t high, [[maybe_unused]] value_t low, [[maybe_unused]] value_t seed,
    [[maybe_unused]] value_t shape, [[maybe_unused]] value_t output, [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::unsqueeze([[maybe_unused]] value_t input, [[maybe_unused]] value_t dim,
                                                    [[maybe_unused]] value_t output,
                                                    [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::where([[maybe_unused]] value_t cond, [[maybe_unused]] value_t x,
                                                [[maybe_unused]] value_t y, [[maybe_unused]] value_t output,
                                                [[maybe_unused]] kernel_context &context) {
    return err(std::errc::not_supported);
}
