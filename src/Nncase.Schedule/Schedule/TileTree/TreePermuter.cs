// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileTree;

public sealed class TreePermuter : ITreeNodeVisitor<AffineMap?, Unit>
{
    public TreePermuter(int op, int level, int[] perm)
    {
        TargetOpId = op;
        TargetLevel = level;
        Perm = perm;
    }

    public int TargetOpId { get; }

    public int TargetLevel { get; }

    public int[] Perm { get; }

    public Unit Visit(ScopeNode value, AffineMap? arg1)
    {
        foreach (var item in value.Children)
        {
            item.Accept(this, arg1);
        }

        return default;
    }

    public Unit Visit(TileNode value, AffineMap? arg1)
    {
        if (value.OpId == TargetOpId && value.Level == TargetLevel)
        {
            value.DomainPermutation = AffineMap.Permutation(Perm);
            return value.Child.Accept(this, AffineMap.Permutation(Perm.InvPermutation()));
        }

        if (arg1 is AffineMap map)
        {
            value.DomainRelation = value.DomainRelation with { Map = map * value.DomainRelation.Map };
            arg1 = null;
        }

        return value.Child.Accept(this, arg1);
    }

    public Unit Visit(OpNode value, AffineMap? arg1)
    {
        if (arg1 is AffineMap map)
        {
            value.DomainRelation = value.DomainRelation with { Map = map * value.DomainRelation.Map };
        }

        return default;
    }
}
