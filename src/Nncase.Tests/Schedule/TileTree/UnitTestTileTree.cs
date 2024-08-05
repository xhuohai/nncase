// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Passes;
using Nncase.Schedule.TileTree;
using Xunit;

namespace Nncase.Tests.Schedule.TileTreeTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTileTree : TestClassBase
{
    public UnitTestTileTree()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.CpuTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR;
#endif
    }

    [Fact]
    public void TestDomainRelationPermutation()
    {
        // [0,1,2] -> [0,1,2]
        var domain = 3;
        var lists = Enumerable.Range(0, domain).Permutate().ToList();
        Assert.Equal(Enumerable.Range(1, domain).Aggregate(1, (a, b) => a * b), lists.Count);
        Assert.Equal(new[] { 0, 2, 1 }, lists[1]);

        var map1 = AffineMap.Permutation(lists[1]);
        var remap1 = AffineMap.Permutation(lists[1].InvPermutation());

        var idmap = AffineMap.Identity(domain);

        Assert.Equal(idmap, map1 * remap1);
    }

    [Fact]
    public void TestWorkItemsEquality()
    {
        var func1 = GetFunctionSample();
        var post1 = CompilerServices.Rewrite(func1, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());

        var func2 = GetFunctionSample();
        var post2 = CompilerServices.Rewrite(func2, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());

        var worklists1 = ExprCollector.Collect(post1).OfType<Grid>().ToList();
        var worklists2 = ExprCollector.Collect(post2).OfType<Grid>().ToList();

        Assert.True(WorkItemsEqualityComparer.Instance.Equals(new List<Grid>() { worklists1[0] }, new List<Grid>() { worklists2[0] }));
        Assert.True(WorkItemsEqualityComparer.Instance.Equals(new List<Grid>() { worklists1[1] }, new List<Grid>() { worklists2[1] }));
        Assert.True(WorkItemsEqualityComparer.Instance.Equals(new List<Grid>() { worklists1[2] }, new List<Grid>() { worklists2[2] }));
        Assert.True(WorkItemsEqualityComparer.Instance.Equals(new List<Grid>() { worklists1[0], worklists1[1] }, new List<Grid>() { worklists2[0], worklists2[1] }));
        Assert.True(WorkItemsEqualityComparer.Instance.Equals(new List<Grid>() { worklists1[2] }, new List<Grid>() { worklists2[2] }));
    }

    [Fact]
    public void TestWorkItemsEquality2()
    {
        var func = GetFunctionSample3();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerBinary() }, new());
        var grids = ExprCollector.Collect(post).OfType<Grid>().ToList();
        var worklists = Nncase.Passes.Transforms.AutoTilePass.GatherWorkLists(grids);
        Assert.True(WorkItemsEqualityComparer.Instance.Equals(worklists[0], worklists[1]));
        Assert.Equal(WorkItemsEqualityComparer.Instance.GetHashCode(worklists[0]), WorkItemsEqualityComparer.Instance.GetHashCode(worklists[1]));
        var hashset = new HashSet<List<Grid>>(WorkItemsEqualityComparer.Instance)
        {
            worklists[0],
        };
        Assert.Contains(worklists[1], hashset);
    }

    [Fact]
    public void TestGetArgumentInfo()
    {
        var func = GetFunctionSample();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = CompileOptions.TargetOptions.MemoryCapacities.Length - 1;
        Nncase.Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");
        var m1 = root.Root<ITreeNode>().Clone();
        m1.Merge(2, 1, 2);
        m1.Dump("final");

        var res = TreeSolverInitializer.Init(m1, totalLevel, CompileOptions.TargetOptions, out _, out _, out _, out _);
        Assert.Equal(3, res.Inputs.Count);
        Assert.Single(res.Outputs);
        Assert.Equal(2, res.DefUseMap.Keys.Count);
    }

    [Fact]
    public void TestGetArgumentInfo2()
    {
        var func = GetFunctionSample2();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), new Passes.Rules.CPU.Affine.LowerBinary() }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = CompileOptions.TargetOptions.MemoryCapacities.Length - 1;
        Nncase.Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");
        var m1 = root.Root<ITreeNode>().Clone();
        m1.Merge(1, 0, 2);
        m1.Merge(1, 0, 1);
        m1.Dump("final");

        var res = TreeSolverInitializer.Init(m1, totalLevel, CompileOptions.TargetOptions, out _, out _, out _, out _);
        Assert.Equal(4, res.Inputs.Count);
        Assert.Single(res.Outputs);
        Assert.Single(res.DefUseMap.Keys);
    }

    [Fact]
    public void TestGetArgumentInfo3()
    {
        var func = GetFunctionSample2();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), new Passes.Rules.CPU.Affine.LowerBinary() }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = CompileOptions.TargetOptions.MemoryCapacities.Length - 1;
        Nncase.Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");
        var m1 = root.Root<ITreeNode>().Clone();
        m1.Merge(1, 0, 2);
        m1.Dump("final");

        var res = TreeSolverInitializer.Init(m1, totalLevel, CompileOptions.TargetOptions, out _, out _, out _, out _);

        // when merge point at top level, should put the cache buffer into defuse map.
        Assert.Equal(4, res.Inputs.Count);
        Assert.Single(res.Outputs);
        Assert.Equal(2, res.DefUseMap.Keys.Count);
    }

    [Fact]
    public void TestPermute1()
    {
        var func = GetFunctionSample();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = 2;
        Nncase.Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");
        root.Reorder(0, 2, new[] { 0, 2, 1 });
        root.Dump("reordered");
    }

    [Fact]
    public void TestMerge()
    {
        var func = GetFunctionSample();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        // Nncase.Schedule.TreeTiler.BuildTree(grid, CompileOptions.TargetOptions);
        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = 2;
        Nncase.Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");

        root.Merge(2, 1, 2);
        var m1 = root.Clone();
        m1.Dump("0");
        m1.Merge(2, 0, 2);
        var m2 = m1.Clone();
        m2.Dump("1");
        m2.Merge(1, 0, 1);
        var m3 = m2.Clone();
        m3.Dump("2");
        m3.Merge(2, 1, 1);
        var m4 = m3.Clone();
        m4.Dump("3");
    }

    private Function GetFunctionSample()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 256 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 256, 384 }));
            var c = IR.F.Tensors.MatMul(a, b);
            var d = IR.F.Math.Exp(c);
            var e = new Var(new TensorType(DataTypes.Float32, new[] { 384, 512 }));
            var f = IR.F.Tensors.MatMul(d, e);
            func = new(f, a, b, e);
        }

        return func;
    }

    private Function GetFunctionSample2()
    {
        Function func;
        {
            var ashape = new[] { 1, 64, 384, 128 };
            var bshape = new[] { 1, 64, 128, 384 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, ashape));
            var b = new IR.Var("b", new IR.TensorType(DataTypes.Float32, bshape));
            var c = IR.F.Tensors.MatMul(a, b);
            var dshape = new[] { 1 };
            var d = new IR.Var("d", new IR.TensorType(DataTypes.Float32, dshape));
            var e = IR.F.Math.Binary(BinaryOp.Div, c, d);
            var fshape = new[] { 1, 1, 384, 384 };
            var f = new IR.Var("f", new IR.TensorType(DataTypes.Float32, fshape));
            var g = IR.F.Math.Binary(BinaryOp.Add, e, f);
            func = new IR.Function("main", g, a, b, d, f);
        }

        return func;
    }

    private Function GetFunctionSample3()
    {
        Function func;
        {
            var ashape = new[] { 1, 64, 384, 128 };
            var bshape = new[] { 1, 1, 384, 128 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, ashape));
            var b = new IR.Var("b", new IR.TensorType(DataTypes.Float32, bshape));
            var c = IR.F.Math.Add(a, b);
            var d = IR.F.Tensors.Reshape(c, ashape);
            var e = new IR.Var("e", new IR.TensorType(DataTypes.Float32, bshape));
            var f = IR.F.Math.Add(d, e);
            var g = IR.F.Tensors.Reshape(f, ashape);
            func = new IR.Function("main", g, a, b, e);
        }

        return func;
    }
}
