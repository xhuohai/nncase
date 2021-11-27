﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.Math
{
    /// <summary>
    /// Clamp expression.
    /// </summary>
    public sealed record Clamp() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Clamp), 0, "input");

        /// <summary>
        /// Gets min.
        /// </summary>
        public static readonly ParameterInfo Min = new(typeof(Clamp), 1, "min", IsScalar());

        /// <summary>
        /// Gets max.
        /// </summary>
        public static readonly ParameterInfo Max = new(typeof(Clamp), 2, "max", IsScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType min, TensorType max) =>input;

    }
}