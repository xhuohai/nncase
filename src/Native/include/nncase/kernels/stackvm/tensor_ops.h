/* This file is generated by tools/stackvm_gen/IsaGen at 2022/11/2 16:04:35
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
#pragma once
#include <nncase/kernels/kernel_context.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/error.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/stackvm/opcode.h>
#include <nncase/tensor.h>
#include <nncase/value.h>

BEGIN_NS_NNCASE_KERNELS_MODULE(stackvm)

NNCASE_API result<value_t>
batch_normalization(value_t input, value_t scale, value_t bias,
                    value_t input_mean, value_t input_var, value_t epsilon,
                    value_t momentum, value_t output = nullptr,
                    kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
batch_to_space(value_t input, value_t block_shape, value_t crops,
               value_t output = nullptr,
               kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
binary(runtime::stackvm::binary_op_t binary_op, value_t lhs, value_t rhs,
       value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
broadcast(value_t input, value_t shape, value_t output = nullptr,
          kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
cast(typecode_t new_type, runtime::stackvm::cast_mode_t cast_mode,
     value_t input, value_t output = nullptr,
     kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
celu(value_t input, value_t alpha, value_t output = nullptr,
     kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
clamp(value_t input, value_t min, value_t max, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
compare(runtime::stackvm::compare_op_t compare_op, value_t lhs, value_t rhs,
        value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
concat(value_t input, value_t axis, value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
constant_of_shape(value_t shape, value_t value, value_t output = nullptr,
                  kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
conv2d(runtime::stackvm::pad_mode_t pad_mode, value_t input, value_t weights,
       value_t bias, value_t stride, value_t padding, value_t dilation,
       value_t groups, value_t fused_clamp, value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
conv2d_transpose(runtime::stackvm::pad_mode_t pad_mode, value_t input,
                 value_t weights, value_t bias, value_t output_shape,
                 value_t stride, value_t padding, value_t output_padding,
                 value_t dilation, value_t groups, value_t fused_clamp,
                 value_t output = nullptr,
                 kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
cum_sum(value_t input, value_t axis, value_t exclusive, value_t reverse,
        value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
dequantize(typecode_t target_type, value_t input, value_t dequant_param,
           value_t output = nullptr,
           kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
elu(value_t input, value_t alpha, value_t output = nullptr,
    kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
expand(value_t input, value_t shape, value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
fake_dequantize(typecode_t target_type, value_t input, value_t dequant_param,
                value_t output = nullptr,
                kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
fake_quantize(typecode_t target_type, value_t input, value_t quant_param,
              value_t output = nullptr,
              kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
flatten(value_t input, value_t axis, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
gather(value_t input, value_t axis, value_t index, value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
gather_nd(value_t input, value_t batch_dims, value_t index,
          value_t output = nullptr,
          kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
get_item(value_t input, value_t index, value_t output = nullptr,
         kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
hard_sigmoid(value_t input, value_t alpha, value_t beta,
             value_t output = nullptr,
             kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
hard_swish(value_t input, value_t output = nullptr,
           kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
hardmax(value_t input, value_t axis, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
instance_normalization(value_t input, value_t scale, value_t bias,
                       value_t epsilon, value_t output = nullptr,
                       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
l2_normalization(value_t input, value_t output = nullptr,
                 kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
leaky_relu(value_t input, value_t alpha, value_t output = nullptr,
           kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
log_softmax(value_t input, value_t axis, value_t output = nullptr,
            kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
lp_normalization(value_t input, value_t axis, value_t p,
                 value_t output = nullptr,
                 kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
lrn(value_t input, value_t alpha, value_t beta, value_t bias, value_t size,
    value_t output = nullptr,
    kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
lstm(runtime::stackvm::lstmdirection_t direction,
     runtime::stackvm::lstmlayout_t layout,
     std::vector<std::string> activations, value_t x, value_t w, value_t r,
     value_t b, value_t sequence_lens, value_t initial_h, value_t initial_c,
     value_t p, value_t activation_alpha, value_t activation_beta, value_t clip,
     value_t hidden_size, value_t input_forget, value_t output_size,
     value_t output = nullptr,
     kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
mat_mul(value_t lhs, value_t rhs, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
normal(typecode_t type, value_t mean, value_t scale, value_t seed,
       value_t shape, value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
normal_like(typecode_t type, value_t input, value_t mean, value_t scale,
            value_t seed, value_t output = nullptr,
            kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
one_hot(runtime::stackvm::one_hot_mode_t one_hot_mode, value_t indices,
        value_t depth, value_t values, value_t axis, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
pad(runtime::stackvm::pad_mode_t pad_mode, value_t input, value_t pads,
    value_t value, value_t output = nullptr,
    kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
prelu(value_t input, value_t slope, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
prod(value_t input, value_t output = nullptr,
     kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
quant_param_of(runtime::stackvm::quant_mode_t quant_mode, value_t range,
               value_t bits, value_t output = nullptr,
               kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
quantize(typecode_t target_type, value_t input, value_t quant_param,
         value_t output = nullptr,
         kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
range(value_t begin, value_t end, value_t step, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
range_of(value_t input, value_t output = nullptr,
         kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
reduce(runtime::stackvm::reduce_op_t reduce_op, value_t input, value_t axis,
       value_t init_value, value_t keep_dims, value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
reduce_arg(runtime::stackvm::reduce_arg_op_t reduce_arg_op, value_t input,
           value_t axis, value_t keep_dims, value_t select_last_index,
           value_t output = nullptr,
           kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
reduce_window2d(runtime::stackvm::reduce_op_t reduce_op, value_t input,
                value_t init_value, value_t filter, value_t stride,
                value_t padding, value_t dilation, value_t ceil_mode,
                value_t count_include_pad, value_t output = nullptr,
                kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
relu(value_t input, value_t output = nullptr,
     kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
relu6(value_t input, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
require(std::string message, value_t predicate, value_t value,
        value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
reshape(value_t input, value_t shape, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t> resize_image(
    runtime::stackvm::image_resize_mode_t resize_mode,
    runtime::stackvm::image_resize_transformation_mode_t transformation_mode,
    runtime::stackvm::image_resize_nearest_mode_t nearest_mode,
    bool is_tfresize, value_t input, value_t roi, value_t new_size,
    value_t cubic_coeff_a, value_t exclude_outside, value_t extrapolation_value,
    value_t output = nullptr,
    kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
reverse_sequence(value_t input, value_t seq_lens, value_t batch_axis,
                 value_t time_axis, value_t output = nullptr,
                 kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
select(value_t predicate, value_t true_value, value_t false_value,
       value_t output = nullptr,
       kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
selu(value_t input, value_t alpha, value_t gamma, value_t output = nullptr,
     kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
shape_of(value_t input, value_t output = nullptr,
         kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
sigmoid(value_t input, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
size_of(value_t input, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
slice(value_t input, value_t begins, value_t ends, value_t axes,
      value_t strides, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
softmax(value_t input, value_t axis, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
softplus(value_t input, value_t output = nullptr,
         kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
softsign(value_t input, value_t output = nullptr,
         kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
space_to_batch(value_t input, value_t block_shape, value_t paddings,
               value_t output = nullptr,
               kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
split(value_t input, value_t axis, value_t sections, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
squeeze(value_t input, value_t dim, value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
stack(value_t inputs, value_t axis, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
tile(value_t input, value_t repeats, value_t output = nullptr,
     kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
top_k(value_t x, value_t k, value_t axis, value_t largest, value_t sorted,
      value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
transpose(value_t input, value_t perm, value_t output = nullptr,
          kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
unary(runtime::stackvm::unary_op_t unary_op, value_t input,
      value_t output = nullptr,
      kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
uniform(typecode_t type, value_t high, value_t low, value_t seed, value_t shape,
        value_t output = nullptr,
        kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
uniform_like(typecode_t type, value_t input, value_t high, value_t low,
             value_t seed, value_t output = nullptr,
             kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
unsqueeze(value_t input, value_t dim, value_t output = nullptr,
          kernel_context &context = default_kernel_context());

NNCASE_API result<value_t>
where(value_t cond, value_t x, value_t y, value_t output = nullptr,
      kernel_context &context = default_kernel_context());

END_NS_NNCASE_KERNELS_MODULE
