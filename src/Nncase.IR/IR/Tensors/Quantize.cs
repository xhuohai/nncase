// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.IR;

namespace Nncase.IR.Tensors
{

    public sealed record Quantize(DataType TargetType) : Op
    {
        public static ParameterInfo Input = new(typeof(Quantize), 0, "Input");
        
        public static ParameterInfo QuantParam = new(typeof(Quantize), 1, "QuantParam");

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}