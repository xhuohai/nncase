// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using CommunityToolkit.HighPerformance.Helpers;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;

namespace Nncase.Schedule.TileTree;

public sealed class WorkItemsEqualityComparer : IEqualityComparer<List<Grid>>
{
    public static readonly WorkItemsEqualityComparer Instance = new WorkItemsEqualityComparer();

    public bool Equals(List<Grid>? x, List<Grid>? y)
    {
        return x is not null && y is not null && x.Count == y.Count && x.Zip(y).All(p => GridEquals(p.First, p.Second));
    }

    public bool GridEquals(Grid x, Grid y)
    {
        return VarTypeEqualityComparer.Instance.Equals(x.DomainParameter, y.DomainParameter) &&
            x.BodyParameters.SequenceEqual(y.BodyParameters, VarTypeEqualityComparer.Instance) &&
            x.AccessMaps.SequenceEqual(y.AccessMaps) &&
            x.Buffers.SequenceEqual(y.Buffers, ExprTypeEqualityComparer.Instance) &&
            x.Reads.SequenceEqual(y.Reads, ExprTypeEqualityComparer.Instance) &&
            new ExprStructuralEqualityVisitor(new[] { (x.DomainParameter, y.DomainParameter) }.Concat(Enumerable.Range(0, x.BodyParameters.Length).Select(i => (x.BodyParameters[i], y.BodyParameters[i])))
            .ToDictionary(p => p.Item1, p => p.Item2)).Visit(x.Body, y.Body);
    }

    public int GridHashCode([DisallowNull] Grid obj)
    {
        return HashCode.Combine(
            obj.DomainParameter.TypeAnnotation,
            HashCode<IRType>.Combine(Enumerable.Range(0, obj.BodyParameters.Length).Select(i => obj.BodyParameters[i].TypeAnnotation).ToArray()),
            HashCode<AffineMap>.Combine(obj.AccessMaps),
            HashCode<IRType>.Combine(Enumerable.Range(0, obj.Buffers.Length).Select(i => obj.Buffers[i].CheckedType).ToArray()),
            HashCode<IRType>.Combine(Enumerable.Range(0, obj.Reads.Length).Select(i => obj.Reads[i].CheckedType).ToArray()),
            new ExprStructuralHashCodeVisitor().Visit(obj.Body));
    }

    public int GetHashCode([DisallowNull] List<Grid> obj)
    {
        return Enumerable.Range(0, obj.Count).Select(i => GridHashCode(obj[i])).Aggregate(obj.Count, HashCode.Combine);
    }
}

internal sealed class VarTypeEqualityComparer : IEqualityComparer<Var>
{
    public static readonly VarTypeEqualityComparer Instance = new VarTypeEqualityComparer();

    public bool Equals(Var? x, Var? y) => x is not null && y is not null && x.TypeAnnotation.Equals(y.TypeAnnotation);

    public int GetHashCode([DisallowNull] Var obj) => obj.TypeAnnotation.GetHashCode();
}

internal sealed class ExprTypeEqualityComparer : IEqualityComparer<Expr>
{
    public static readonly ExprTypeEqualityComparer Instance = new ExprTypeEqualityComparer();

    public bool Equals(Expr? x, Expr? y) => x is not null && y is not null && x.CheckedType.Equals(y.CheckedType);

    public int GetHashCode([DisallowNull] Expr obj) => obj.CheckedType.GetHashCode();
}

internal sealed class ExprStructuralEqualityVisitor : ExprVisitor<bool, Unit, Expr>
{
    public ExprStructuralEqualityVisitor(Dictionary<Var, Var> varMap)
    {
        VarMap = varMap;
    }

    public Dictionary<Var, Var> VarMap { get; }

    protected override bool DispatchVisit(Expr expr, Expr context)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        return MarkVisited(expr, VisitExpr(expr, context));
    }

    private bool VisitExpr(Expr expr, Expr other)
    {
        if (expr.GetType() == other.GetType() && expr.Operands.Length == other.Operands.Length)
        {
            if (expr is Var lhs)
            {
                if (VarMap.TryGetValue(lhs, out var target))
                {
                    return target.Equals(other);
                }

                return other is Var rhs && lhs.TypeAnnotation.Equals(rhs.TypeAnnotation);
            }

            return Enumerable.Range(0, expr.Operands.Length).All(i => Visit(expr.Operands[i], other.Operands[i]));
        }

        return false;
    }
}

internal sealed class ExprStructuralHashCodeVisitor : ExprVisitor<int, Unit>
{
    protected override int DispatchVisit(Expr expr)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        return MarkVisited(expr, VisitExpr(expr));
    }

    private int VisitExpr(Expr expr)
    {
        if (expr.Operands.Length == 0)
        {
            if (expr is Var @var)
            {
                return @var.TypeAnnotation.GetHashCode();
            }

            return expr.GetHashCode();
        }
        else
        {
            return Enumerable.Range(0, expr.Operands.Length).Select(i => Visit(expr.Operands[i])).Aggregate(expr.GetType().GetHashCode(), HashCode.Combine);
        }
    }
}
