// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="MatMul"/>.
/// </summary>
public class MatMulEvaluator : IEvaluator<MatMul>, ITypeInferencer<MatMul>, ICostEvaluator<MatMul>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, MatMul matMul)
    {
        var input = context.GetOrtArgumentValue(matMul, MatMul.Lhs);
        var other = context.GetOrtArgumentValue(matMul, MatMul.Rhs);
        return OrtKI.MatMul(input, other).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, MatMul target)
    {
        var lhs = context.CheckArgumentType<TensorType>(target, MatMul.Lhs);
        var rhs = context.CheckArgumentType<TensorType>(target, MatMul.Rhs);
        return Visit(lhs, rhs);
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, MatMul target)
    {
        var lhs = context.GetArgumentType<TensorType>(target, MatMul.Lhs);
        var rhs = context.GetArgumentType<TensorType>(target, MatMul.Rhs);
        var outputType = context.GetReturnType<TensorType>();

        if (lhs.Shape.IsUnranked)
        {
            Console.WriteLine("unrank");
        }
        var macPerElement = lhs.Shape[^1].IsFixed ? lhs.Shape[^1].FixedValue : 1;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };
    }

    private IRType Visit(TensorType lhs, TensorType rhs)
    {
        if (lhs.Shape.IsUnranked || rhs.Shape.IsUnranked)
        {
            return new TensorType(lhs.DType, Shape.Unranked);
        }

        // if (lhs.Shape[^1].IsUnknown || rhs.Shape[^2].IsUnknown)
        // {
        //     return new TensorType(lhs.DType, Shape.Unranked);
        // }

        if (lhs.DType != rhs.DType)
        {
            return new InvalidType("MatMul lhs and rhs have different DType");
        }

        if (lhs.Shape[^1] != rhs.Shape[^2] && lhs.Shape[^1] != Dimension.Unknown && rhs.Shape[^2] != Dimension.Unknown)
        {
            return new InvalidType("MatMul lhs and rhs have not compatiable shape");
        }

        if (lhs.Shape.Count == 2 && rhs.Shape.Count == 2)
        {
            return new TensorType(lhs.DType, new[] { lhs.Shape[0], rhs.Shape[1] });
        }

        var bigShape = lhs.Shape.Rank > rhs.Shape.Rank
            ? lhs.Shape
            : rhs.Shape;
        // batch and channel
        var front = bigShape.ToArray()[..(bigShape.Count - 2)];
        var end = new[] { lhs.Shape[^2], rhs.Shape[^1] };
        return new TensorType(lhs.DType, front.Concat(end).ToArray());
    }
}