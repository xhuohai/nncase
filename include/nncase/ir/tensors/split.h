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
#pragma once
#include "../op.h"
#include "nncase/runtime/datatypes.h"
#include "opcode.h"

namespace nncase::ir::tensors {
/** @brief Split operator node */
class NNCASE_API split_node : public op_node {
    DEFINE_OBJECT_KIND(op_node, op_tensors_split)
  public:
    split_node(int32_t axis);

    /** @brief Get the axis of the concat expression */
    int32_t axis() const noexcept { return axis_; }
    /** @brief Set the axis of the concat expression */
    void axis(int32_t value) noexcept { axis_ = value; }

  private:
    int32_t axis_;
};

using split = object_t<split_node>;
} // namespace nncase::ir::tensors