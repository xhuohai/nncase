// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using System.Text;
using GiGraph.Dot.Entities.Edges;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Html.Table;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Records;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileTree;

internal sealed class TreePrinter : ITreeNodeVisitor<TreePrinter.Context, TreePrinter.Result>
{
    public TreePrinter()
    {
        Graph = new DotGraph(true);
    }

    public DotGraph Graph { get; }

    /// <summary>
    /// Get the axis map from current domain to parent domain.
    /// </summary>
    public static Dictionary<int, int> GetDimsMap(ITileAbleNode value)
    {
        var map = new Dictionary<int, int>();
        var relation = value.DomainRelation.Map;
        for (int i = 0; i < relation.Results.Length; i++)
        {
            if (relation.Results[i] is { Offset: AffineDim dim, Extent: AffineExtent ext } && dim.Position == ext.Position)
            {
                map[i] = dim.Position;
            }
        }

        return map;
    }

    public Result Visit(ScopeNode value, Context context)
    {
        var nodes = new List<GiGraph.Dot.Entities.Nodes.DotNode>();
        var relations = new List<DomainRelation>();
        foreach (var child in value.Children)
        {
            var ret = child.Accept(this, context);
            nodes.AddRange(ret.Nodes);
            relations.AddRange(ret.Relations);
        }

        return new(nodes, relations);
    }

    public Result Visit(TileNode value, Context context)
    {
        var (pid, pdomain) = context;

        var rank = value.DomainPermutation.Results.Length;
        pdomain ??= AffineMap.Identity(rank);

        var lables = new List<string>();
        var curdomain = pdomain * value.DomainRelation.Map * value.DomainPermutation;
        for (int i = 0; i < rank; i++)
        {
            var indent = string.Join(string.Empty, Enumerable.Repeat("    ", i));
            var label = GetDisplayString(curdomain.Results[i].Extent, Array.Empty<AffineSymbol>(), $"Op{value.OpId}_L{value.Level}_d", $"Op{value.OpId}_L{value.Level}_d");
            lables.Add($"{indent}For {label}");
        }

        var node = Graph.Nodes.Add(value.ToString());
        node.ToRecordNode(rb1 => rb1.AppendRecord(rb2 => rb2.AppendField($"Op{value.OpId}").AppendField(string.Join('\n', lables))).AppendField(value.DomainPermutation.ToString()), true);

        var result = value.Child.Accept(this, context with { ParentOpId = value.OpId, ParentDomain = curdomain });

        for (int i = 0; i < result.Nodes.Count; i++)
        {
            Graph.Edges.Add(new DotEdge(node.Id, result.Nodes[i].Id), edge =>
            {
                var r = result.Relations[i];
                edge.Label = $"Op{r.DomainOp} -> Op{r.RangeOp}{System.Environment.NewLine}{r.Map}";
            });
        }

        return new(new() { node }, new() { value.DomainRelation });
    }

    public Result Visit(OpNode value, Context context)
    {
        var (pid, pdomain) = context;

        var node = Graph.Nodes.Add($"{value}");

        node.ToRecordNode(rb => rb.AppendRecord(rb1 =>
        {
            rb1.AppendField(value.ToString()).
                AppendField($"{value.Op.GetType().Name}({value.Op.DisplayProperty()})").
                AppendFields(value.DomainBounds.Select((d, i) => $"d{i} : {d}"));
        }).AppendRecord(
            rb2 =>
            {
                for (int i = 0; i < value.ReadAccesses.Length; i++)
                {
                    rb2.AppendField($"read {value.ReadAccesses[i]}");
                }

                rb2.AppendField($"write {value.WriteAccess}");
            }));

        return new(new() { node }, new() { value.DomainRelation });
    }

    private string GetDisplayString(AffineExpr expr, ReadOnlySpan<AffineSymbol> symbols, string offsetPrefix = "t", string extentPrefix = "d")
    {
        return expr switch
        {
            AffineConstant e => e.Value.ToString(),
            AffineExtent e => $"{offsetPrefix}{e.Position}",
            AffineDim e => $"{extentPrefix}{e.Position}",
            AffineSymbol e => e.ToString(),
            AffineAddBinary e => $"({GetDisplayString(e.Lhs, symbols, offsetPrefix, extentPrefix)} + {GetDisplayString(e.Rhs, symbols, offsetPrefix, extentPrefix)})",
            AffineMulBinary e => $"({GetDisplayString(e.Lhs, symbols, offsetPrefix, extentPrefix)} * {GetDisplayString(e.Rhs, symbols, offsetPrefix, extentPrefix)})",
            AffineDivBinary e => $"({GetDisplayString(e.Lhs, symbols, offsetPrefix, extentPrefix)} {IR.F.Affine.ToString(e.BinaryOp)} {GetDisplayString(e.Rhs, symbols, offsetPrefix, extentPrefix)})",
            _ => throw new System.Diagnostics.UnreachableException(),
        };
    }

    internal record Context(int ParentOpId, AffineMap? ParentDomain)
    {
        public static Context Default => new(-1, null);
    }

    internal record Result(List<DotNode> Nodes, List<DomainRelation> Relations)
    {
    }
}
