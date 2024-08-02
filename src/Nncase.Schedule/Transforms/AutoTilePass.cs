// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.Schedule.TileTree;

namespace Nncase.Passes.Transforms;

public sealed class AutoTilePass : ModulePass
{
    public AutoTilePass(string moduleKind, CompileOptions compileOptions)
    {
        ModuleKind = moduleKind;
        CompileOptions = compileOptions;
        WorkItem = 0;
    }

    public string ModuleKind { get; }

    public CompileOptions CompileOptions { get; }

    public int WorkItem { get; set; }

    public static List<List<Grid>> GatherWorkLists(IReadOnlyList<Grid> collects)
    {
        List<List<Grid>> workLists = new();
        for (int i = collects.Count - 1; i >= 0;)
        {
            // start find single input.
            if (collects[i] is Grid sinkNode)
            {
                var current = sinkNode;
                var workItems = new List<Grid>() { current };
                int j = i - 1;
                while (j >= 0)
                {
                    var currentReads = current.Reads.AsValueEnumerable().Where(read => read is Grid producer && producer.Users.Count() == 2).ToArray();
                    if (!(currentReads.Length == 1 && currentReads[0] == collects[j]))
                    {
                        break;
                    }

                    current = collects[j];
                    workItems.Add(current);
                    j--;
                }

                i = j;
                workLists.Insert(0, workItems);
            }
            else
            {
                i--;
            }
        }

        return workLists;
    }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var funcNums = input.Functions.Count;
        for (int i = 0; i < funcNums; i++)
        {
            // top sorted
            var collects = ExprCollector.Collect(input.Functions[i]).OfType<Grid>().ToArray();
            using var pinner = new ExprPinner(collects);
            var tiledMemo = new Dictionary<List<Grid>, (PrimFunctionWrapper, ArgumentsLocationInfo[])>(WorkItemsEqualityComparer.Instance);
            var worklists = GatherWorkLists(collects);
            var post = input.Functions[i];

            for (int j = 0; j < worklists.Count; j++)
            {
                var workitems = worklists[j];
                var rewriter = new AutoTileRewriter(tiledMemo, workitems, ModuleKind, WorkItem++, CompileOptions);
                post = (BaseFunction)rewriter.Rewrite(post);
            }

            input.Replace(i, post);
        }

        return Task.FromResult(input);
    }

    private sealed class AutoTileRewriter : ExprRewriter
    {
        private readonly string _moduleKind;
        private readonly int _number;

        public AutoTileRewriter(Dictionary<List<Grid>, (PrimFunctionWrapper Wrapper, ArgumentsLocationInfo[] Locs)> tiledMemo, List<Grid> workitems, string moduleKind, int number, CompileOptions compileOptions)
        {
            TiledMemo = tiledMemo;
            Workitems = workitems;
            _moduleKind = moduleKind;
            _number = number;
            CompileOptions = compileOptions;
        }

        public Dictionary<List<Grid>, (PrimFunctionWrapper Wrapper, ArgumentsLocationInfo[] Locs)> TiledMemo { get; }

        public List<Grid> Workitems { get; }

        public Grid Root => Workitems[0];

        public CompileOptions CompileOptions { get; }

        protected override Expr RewriteLeafGrid(Grid grid)
        {
            if (grid == Root)
            {
                Call call;
                if (TiledMemo.TryGetValue(Workitems, out var result))
                {
                    var (wrapper, locs) = result;
                    call = new Call(wrapper, locs.Select(loc =>
                    {
                        var expr = Workitems[^(loc.OpId + 1)].Reads[loc.Index];
                        return loc.Cached ? TilingUtilities.GetUninitialized(expr) : expr;
                    }).ToArray());
                }
                else
                {
                    call = TreeTiler.Tile(grid, _moduleKind, _number, CompileOptions.TargetOptions, out var argumentLocation);
                    TiledMemo.Add(Workitems, ((PrimFunctionWrapper)call.Target, argumentLocation));
                }

                return call;
            }

            return grid;
        }
    }
}
