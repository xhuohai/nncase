/* Copyright 2019-2021 Canaan Inc.
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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/ternary.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

ternary::ternary(datatype_t input_a_type, datatype_t input_bc_type, shape_t input_a_shape, shape_t input_b_shape, shape_t input_c_shape)
{
    add_input("input_a", input_a_type, input_a_shape);
    add_input("input_b", input_bc_type, input_b_shape);
    add_input("input_c", input_bc_type, input_c_shape);
    add_output("output", input_bc_type, get_binary_output_shape(get_binary_output_shape(input_a_shape, input_b_shape), input_c_shape));
}

bool ternary::properties_equal([[maybe_unused]] node &other) const
{
    return false;
}