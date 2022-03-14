// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Transpose"/>.
/// </summary>
public class TransposeEvaluator : IEvaluator<Transpose>, ITypeInferencer<Transpose>, ICostEvaluator<Transpose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Transpose tr)
    {
        var input = context.GetTorchArgumentValue(tr, Transpose.Input);
        var perm = context.GetArgumentValueAsTensor<long>(tr, Transpose.Perm);
        return input.permute(perm.ToArray()).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Transpose target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Transpose.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Transpose target)
    {
        var returnType = context.GetReturnType<TensorType>();
        var arithm = returnType.Shape.Prod().FixedValue;
        return new(arithm, arithm * returnType.DType.SizeInBytes);
    }

    private IRType Visit(ITypeInferenceContext context, Transpose target, TensorType input)
    {
        var permExpr = context.GetArgument(target, Transpose.Perm);
        return TypeInference.TransposeType(input, permExpr);
    }
}