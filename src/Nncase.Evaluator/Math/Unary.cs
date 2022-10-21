// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Unary"/>.
/// </summary>
public class UnaryEvaluator : IEvaluator<Unary>, ITypeInferencer<Unary>, ICostEvaluator<Unary>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unary unary)
    {
        var input_tensor = context.GetArgumentValueAsTensor(unary, Unary.Input);
        if (input_tensor.Shape.IsScalar)
        {

            if (input_tensor.ElementType == DataTypes.Int32)
            {
                return Value.FromTensor(Tensor.FromScalar<int>(compute_int(input_tensor.ToScalar<int>(), unary.UnaryOp)));
            }
            else if (input_tensor.ElementType == DataTypes.Float32)
            {
                return Value.FromTensor(Tensor.FromScalar<float>(compute_float(input_tensor.ToScalar<float>(), unary.UnaryOp)));
            }
        }
        var input = context.GetOrtArgumentValue(unary, Unary.Input);
        var result = unary.UnaryOp switch
        {
            UnaryOp.Abs => OrtKI.Abs(input),
            UnaryOp.Acos => OrtKI.Acos(input),
            UnaryOp.Acosh => OrtKI.Acosh(input),
            UnaryOp.Asin => OrtKI.Asin(input),
            UnaryOp.Asinh => OrtKI.Asinh(input),
            UnaryOp.Ceil => OrtKI.Ceil(input),
            UnaryOp.Cos => OrtKI.Cos(input),
            UnaryOp.Cosh => OrtKI.Cosh(input),
            UnaryOp.Exp => OrtKI.Exp(input),
            UnaryOp.Floor => OrtKI.Floor(input),
            UnaryOp.Log => OrtKI.Log(input),
            UnaryOp.Neg => OrtKI.Neg(input),
            UnaryOp.Round => OrtKI.Round(input),
            UnaryOp.Rsqrt => OrtKI.Rsqrt(input),
            UnaryOp.Sin => OrtKI.Sin(input),
            UnaryOp.Sinh => OrtKI.Sinh(input),
            UnaryOp.Sign => OrtKI.Sign(input),
            UnaryOp.Sqrt => OrtKI.Sqrt(input),
            UnaryOp.Square => OrtKI.Square(input),
            UnaryOp.Tanh => OrtKI.Tanh(input),
            UnaryOp.BitwiseNot => throw new NotSupportedException("NotSupported UnaryOp BitwiseNot"),
            UnaryOp.LogicalNot => OrtKI.Not(input),
            _ => throw new ArgumentOutOfRangeException(nameof(unary.UnaryOp)),
        };
        return result.ToValue();
    }

    private int compute_int(int input, UnaryOp op) => op switch
    {
        UnaryOp.Ceil => input,
        UnaryOp.Floor => input,
        UnaryOp.Neg => -input,
        _ => throw new ArgumentOutOfRangeException($"NotSupported {nameof(op)} For Int"),
    };

    private float compute_float(float input, UnaryOp op) => op switch
    {
        UnaryOp.Abs => System.MathF.Abs(input),
        UnaryOp.Acos => System.MathF.Acos(input),
        UnaryOp.Acosh => System.MathF.Acosh(input),
        UnaryOp.Asin => System.MathF.Asin(input),
        UnaryOp.Asinh => System.MathF.Asinh(input),
        UnaryOp.Ceil => System.MathF.Ceiling(input),
        UnaryOp.Cos => System.MathF.Cos(input),
        UnaryOp.Cosh => System.MathF.Cosh(input),
        UnaryOp.Exp => System.MathF.Exp(input),
        UnaryOp.Floor => System.MathF.Floor(input),
        UnaryOp.Log => System.MathF.Log(input),
        UnaryOp.Neg => -input,
        UnaryOp.Round => System.MathF.Round(input),
        UnaryOp.Rsqrt => 1.0f / System.MathF.Sqrt(input),
        UnaryOp.Sin => System.MathF.Sin(input),
        UnaryOp.Sinh => System.MathF.Sinh(input),
        UnaryOp.Sign => System.MathF.Sign(input),
        UnaryOp.Sqrt => System.MathF.Sqrt(input),
        UnaryOp.Square => input * input,
        UnaryOp.Tanh => System.MathF.Tanh(input),
        _ => throw new ArgumentOutOfRangeException($"NotSupported {nameof(op)} For Float"),
    };

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unary target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Unary.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unary target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Unary.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, CostUtility.GetCPUCyclesOfUnary(target.UnaryOp)),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}