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
#include <nncase/ir/math/binary.h>
#include <nncase/ir/math/clamp.h>
#include <nncase/ir/math/functional.h>
#include <nncase/ir/math/unary.h>

using namespace nncase;
using namespace nncase::ir;

call F::unary(unary_op_t unary_op, F::fexpr input) {
    return call(math::unary(unary_op), {input});
}

call F::binary(binary_op_t unary_op, F::fexpr lhs, F::fexpr rhs) {
    return call(math::binary(unary_op), {lhs, rhs});
}

call F::clamp(F::fexpr input, F::fexpr min, F::fexpr max) {
    return call(math::clamp(), {input, min, max});
}
