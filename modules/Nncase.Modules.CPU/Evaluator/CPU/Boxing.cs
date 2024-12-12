﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable SA1010, SA1008
using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Utilities;

namespace Nncase.Evaluator.IR.CPU;

public sealed class BoxingEvaluator : ITypeInferencer<Boxing>, ICostEvaluator<Boxing>, IEvaluator<Boxing>
{
    private const int _burstLength = 256;

    public IRType Visit(ITypeInferenceContext context, Boxing target)
    {
        var check = (DistributedType inv, DistributedType outv) =>
            {
                // TODO: add more invalid cases
                if (inv.NdSBP.Distinct().Count() == 1 && outv.NdSBP.Distinct().Count() == 1 && inv.NdSBP[0] == outv.NdSBP[0])
                {
                    return (IRType)new InvalidType("Same NDSBP");
                }

                if (inv.NdSBP.Any(sbp => sbp is SBPPartialSum))
                {
                    DistributedUtility.TryGetDividedTensorType(inv, out var inType);
                    DistributedUtility.TryGetDividedTensorType(outv, out var outType);

                    var nonPartialSumPos = Enumerable.Range(0, inv.NdSBP.Count).Where(i => inv.NdSBP[i] is not SBPPartialSum);
                    if (nonPartialSumPos.Any(i => inv.NdSBP[i] is SBPSplit && outv.NdSBP[i] is SBPBroadCast))
                    {
                        return new InvalidType("Not supported input is Splite output is BroadCast");
                    }

                    var partialSumPos = Enumerable.Range(0, inv.NdSBP.Count).Where(i => inv.NdSBP[i] is SBPPartialSum);
                    if (partialSumPos.Any(i => inv.NdSBP[i] is SBPPartialSum && outv.NdSBP[i] is SBPSplit))
                    {
                        return new InvalidType("Not supported input is Partial output is Split");
                    }

                    return outv;
                }
                else
                {
                    return outv;
                }
            };

        return (context.GetArgumentType(target, Boxing.Input), target.NewType) switch
        {
            (InvalidType inv, _) => inv,
            (DistributedType inv, DistributedType outv) => check(inv, outv),
            _ => target.NewType,
        };
    }

    public Cost Visit(ICostEvaluateContext context, Boxing target)
    {
        var inType = context.GetArgumentType<IRType>(target, Boxing.Input);
        var returnType = context.GetReturnType<IRType>();
        var cost = new Cost() { [CostFactorNames.MemoryLoad] = 0, [CostFactorNames.MemoryStore] = 0 };
        switch (inType, returnType)
        {
            case (TensorType tensorType, DistributedType distTensorType):
                switch (context.CompileOptions.TargetOptions)
                {
                    case Targets.CpuTargetOptions { UnifiedMemoryArch: true }:
                        break;
                    default:
                        cost = new Cost()
                        {
                            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(tensorType),
                            [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(distTensorType) / DistributedUtility.GetDividedTensorEfficiency(distTensorType, _burstLength)),
                        };
                        break;
                }

                break;
            case (DistributedType distTensorType, TensorType tensorType):
                switch (context.CompileOptions.TargetOptions)
                {
                    case Targets.CpuTargetOptions { UnifiedMemoryArch: true }:
                        break;
                    default:
                        cost = new Cost()
                        {
                            [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(distTensorType) / DistributedUtility.GetDividedTensorEfficiency(distTensorType, _burstLength)),
                            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(tensorType),
                        };
                        break;
                }

                break;

            case (DistributedType a, DistributedType b) when a.Placement == b.Placement && a.NdSBP != b.NdSBP:
                {
                    var fullLoadStore = new Cost()
                    {
                        [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(a) / DistributedUtility.GetDividedTensorEfficiency(a, _burstLength)),
                        [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(b) / DistributedUtility.GetDividedTensorEfficiency(b, _burstLength)),
                    };

                    float scatterPart = 1;
                    float gatherPart = 1;
                    for (int i = 0; i < a.Placement.Rank; i++)
                    {
                        switch (a.NdSBP[i], b.NdSBP[i])
                        {
                            case (SBPSplit { Axis: int ax }, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPSplit { Axis: int bx }:
                                        if (ax != bx)
                                        {
                                            // when split different axis, need global load store.
                                            return fullLoadStore;
                                        }

                                        break;
                                    case SBPBroadCast:
                                        scatterPart *= a.Placement.Hierarchy[i];
                                        gatherPart *= a.Placement.Hierarchy[i];
                                        break;
                                    default:
                                        throw new NotSupportedException("split to partial");
                                }

                                break;
                            case (SBPBroadCast, SBPBroadCast or SBPSplit):
                                // no cost.
                                cost += new Cost()
                                {
                                    [CostFactorNames.CPUCycles] = 1,
                                };
                                break;
                            case (SBPPartialSum, SBP sbpout):
                                switch (sbpout)
                                {
                                    case SBPPartialSum:
                                        break;
                                    case SBPBroadCast or SBPSplit:
                                        gatherPart *= a.Placement.Hierarchy[i];
                                        if (i == 0)
                                        {
                                            scatterPart *= a.Placement.Hierarchy[i];
                                        }

                                        break;
                                }

                                break;
                            case (SBPBroadCast, SBPPartialSum):
                                // note this case only for tests.
                                cost += new Cost()
                                {
                                    [CostFactorNames.CPUCycles] = 1,
                                };
                                break;
                            default:
                                throw new NotSupportedException($"{a} to {b}");
                        }
                    }

                    if (gatherPart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.MemoryStore] = (UInt128)((gatherPart - 1) * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(a)) / gatherPart),
                        };
                    }

                    if (scatterPart > 1f)
                    {
                        cost += new Cost()
                        {
                            [CostFactorNames.MemoryLoad] = (UInt128)((scatterPart - 1) * (float)CostUtility.GetMemoryAccess(DistributedUtility.GetDividedTensorType(b)) / scatterPart),
                        };
                    }
                }

                break;
            case (DistributedType a, DistributedType b) when a.TensorType != b.TensorType && a.Placement == b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(a) / DistributedUtility.GetDividedTensorEfficiency(a, _burstLength)),
                    [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(b) / DistributedUtility.GetDividedTensorEfficiency(b, _burstLength)),
                };
                break;
            case (DistributedType a, DistributedType b) when a.Placement != b.Placement:
                cost = new Cost()
                {
                    [CostFactorNames.MemoryStore] = (UInt128)((float)CostUtility.GetMemoryAccess(a) / DistributedUtility.GetDividedTensorEfficiency(a, _burstLength)),
                    [CostFactorNames.MemoryLoad] = (UInt128)((float)CostUtility.GetMemoryAccess(b) / DistributedUtility.GetDividedTensorEfficiency(b, _burstLength)),
                };
                break;
            case (DistributedType a, DistributedType b) when a == b:
                throw new InvalidOperationException($"the boxing inType == outType");
            default:
                throw new NotSupportedException($"{inType} {returnType}");
        }

        return cost;
    }

    public IValue Visit(IEvaluateContext context, Boxing target)
    {
        var input = context.GetArgumentValueAsTensor(target, Boxing.Input);
        return target.NewType switch
        {
            TensorType t => Value.FromTensor(Tensor.FromBytes(input.ElementType, input.BytesBuffer.ToArray(), t.Shape)),
            DistributedType d => Value.FromTensor(Tensor.FromBytes(input.ElementType, input.BytesBuffer.ToArray(), d.TensorType.Shape), d.NdSBP, d.Placement),
            _ => Value.FromTensor(input),
        };
    }
}
