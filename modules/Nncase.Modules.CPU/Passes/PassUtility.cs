﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

public static class PassUtility
{
    public static bool IsCpuSupported(Op op)
    {
        if (op.GetType().Namespace == "Nncase.IR.CPU" || op.GetType().Namespace == "Nncase.IR.CustomCPU")
        {
            return true;
        }

        return op is IR.Math.Unary or IR.Math.Binary { BinaryOp: BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div } or IR.Math.MatMul or IR.NN.Conv2D { PadMode: PadMode.Constant } or IR.NN.Softmax or IR.NN.LayerNorm or IR.NN.InstanceNormalization or IR.Imaging.ResizeImage { IsTFResize: false } or IR.Tensors.Unsqueeze or IR.Tensors.Reshape or IR.Tensors.Slice or IR.Tensors.Concat or IR.Tensors.Transpose or IR.NN.Swish or IR.Tensors.Gather or IR.NN.Pad { PadMode: PadMode.Constant } or IR.Math.Reduce or IR.Math.ReduceArg or IR.Math.Clamp or IR.NN.Erf or IR.Tensors.Cast or IR.Tensors.Expand or IR.Tensors.Where or IR.Math.Compare or IR.Tensors.ScatterND;
    }

    public static bool IsCpuSupported(Op op, Expr expr, IEnumerable<Expr> arguments)
    {
        if (!IsCpuSupported(op))
        {
            return false;
        }

        if (!op.Parameters.Zip(arguments).All(p => (p.First.ParameterKind == ParameterKind.Input && p.Second.CheckedType switch { TensorType t => t.Shape.IsRanked, _ => true }) || (p.First.ParameterKind == ParameterKind.Attribute && (p.Second is TensorConst || p.Second is IR.None))))
        {
            return false;
        }

        switch (op)
        {
            case IR.Imaging.ResizeImage:
                var roi = arguments.Skip(IR.Imaging.ResizeImage.Roi.Index).First();
                if (roi is not IR.None && roi.CheckedShape.Size != 0)
                {
                    return false;
                }

                break;
            case IR.Tensors.Slice slice:
                if (((TensorConst)arguments.Skip(IR.Tensors.Slice.Strides.Index).First()).Value.ToArray<int>().Any(s => s < 0))
                {
                    return false;
                }

                if (((TensorConst)arguments.Skip(IR.Tensors.Slice.Begins.Index).First()).Value.ToArray<int>().Any(s => s < 0))
                {
                    return false;
                }

                if (((TensorConst)arguments.Skip(IR.Tensors.Slice.Ends.Index).First()).Value.ToArray<int>().Any(s => s < 0))
                {
                    return false;
                }

                break;
            case IR.NN.Conv2D conv2d:
                if (((TensorConst)arguments.Skip(IR.NN.Conv2D.FusedClamp.Index).First()).Value.ToArray<float>() is var clamp)
                {
                    return clamp.SequenceEqual(new[] { float.NegativeInfinity, float.PositiveInfinity });
                }

                break;
            case IR.Math.Binary binary:
                if (arguments.Any(x => x.CheckedType is AnyType || x is If))
                {
                    return false;
                }

                break;
            case IR.Math.Reduce reduce:
                var axis = ((TensorConst)arguments.ToArray()[1]).Value.ToArray<int>().OrderBy(x => x).ToArray();
                bool consecutiveAixs = axis.Length <= 1 || axis.Zip(axis.Skip(1)).All(p => p.First == p.Second - 1);
                if (reduce.ReduceOp == ReduceOp.Prod ||
                 arguments.ToArray()[0].CheckedDataType == DataTypes.Float16 ||
                 !consecutiveAixs)
                {
                    return false;
                }

                break;

            case IR.Tensors.Cast cast:
                var inType = arguments.ToArray()[0].CheckedDataType;
                if (inType == DataTypes.Float16 || inType == DataTypes.BFloat16 || cast.NewType == DataTypes.Float16 || cast.NewType == DataTypes.BFloat16)
                {
                    return false;
                }

                break;

            case IR.Tensors.Expand expand:
                if (arguments.ToArray()[0].CheckedShape.Rank != arguments.ToArray()[1].CheckedShape.Size)
                {
                    return false;
                }

                break;

            case IR.Tensors.Where where:
                if (arguments.ToArray()[0].CheckedShape != expr.CheckedShape)
                {
                    return false;
                }

                break;
            default:
                break;
        }

        return true;
    }
}