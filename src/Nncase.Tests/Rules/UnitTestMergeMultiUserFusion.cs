// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Quantization;
using Nncase.Tests.ReWrite.FusionTest;
using Nncase.Tests.TestFixture;
using Nncase.Tests.TransformTest;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.Rules;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestMergeMultiUserFusion : TransformTestBase
{
    [Fact]
    public async Task TestSimple()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));

        var callee = MakeSingleSimpleFusionCall(Abs, inputVar);
        var caller0 = MakeSingleSimpleFusionCall(Sqrt, callee);
        var caller1 = MakeSingleSimpleFusionCall(Ceil, callee);
        var output = new IR.Tuple(caller0, caller1);
        var dict = new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } };
        await RunTest(output, new[] { inputVar }, dict);
    }

    [Fact]
    public async Task TestHasSameInput()
    {
        // tr = transpose(input)
        // callee = Abs(tr)
        // callee + tr | callee - tr
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var tr = Transpose(inputVar, new[] { 3, 2, 1, 0 });
        var callee = MakeSingleSimpleFusionCall(Abs, tr);
        var caller0 = MakeSimpleFusionCall(args => args[0] + args[1], callee, tr);
        var caller1 = MakeSimpleFusionCall(args => args[0] - args[1], callee, tr);
        var output = new IR.Tuple(caller0, caller1);
        var dict = new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } };
        await RunTest(output, new[] { inputVar }, dict);
    }


    // 被合并的几个call互为参数
    [Fact]
    public async Task TestComplexExpr()
    {
        // tr = transpose(input)
        // f = fusion_multi_user(tr)
        // leakyRelu = LeakyRelu(f)
        // complexFusion(LeakyRelu, f)
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var tr = Transpose(inputVar, new[] { 3, 2, 1, 0 });
        var f = MakeSingleSimpleFusionCall(Abs, tr);
        var leakyRelu = MakeSingleSimpleFusionCall(expr => LeakyRelu(expr, 0.1), f);
        var complexFusion = MakeSimpleFusionCall(args => args[0] - args[1], leakyRelu, f);
        var output = new IR.Tuple(leakyRelu, complexFusion);
        var dict = new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } };
        await RunTest(output, new[] { inputVar }, dict);
    }


    [Fact]
    public async Task TestWithRing()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var leakyRelu = MakeSingleSimpleFusionCall(expr => LeakyRelu(expr, 0.1), inputVar);
        var abs = MakeSingleSimpleFusionCall(Abs, leakyRelu);
        var sp = ShapeOf(abs);
        var data = ConstantOfShape(sp, 0f);
        var binary = MakeSimpleFusionCall(args => args[0] - args[1], leakyRelu, data);
        var output = binary;
        var dict = new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } };
        await RunTest(output, new[] { inputVar }, dict);
    }

    [Fact]
    public async Task TestSeqNoRing()
    {
        var input = Testing.Rand<float>(1, 3, 24, 24);
        var con = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var abs0 = Softmax(new[] { 1f }, 0);
        var abs1 = Softmax(new[] { 2f }, 0);
        var mm1 = MakeSingleSimpleFusionCall(expr => IR.F.Math.MatMul(expr, con), Softmax(inputVar, 0));
        var body = MakeSimpleFusionCall(
            expr => IR.F.Math.MatMul(expr[0], Testing.Rand<float>(1, 3, 24, 24)) * expr[1] * expr[2], mm1, abs0, abs1);
        await RunTest(body, new[] { inputVar }, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public async Task MergeFusionTuple()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar1 = new Var(new TensorType(input1.ElementType, input1.Shape));
        var input2 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar2 = new Var(new TensorType(input2.ElementType, input2.Shape));
        var a1 = MakeSingleSimpleFusionCall(Abs, inputVar0);
        var a2 = MakeSingleSimpleFusionCall(Abs, inputVar1);
        var a3 = MakeSingleSimpleFusionCall(Abs, inputVar2);
        TestMatched<MergeTupleFusion>(
            new IR.Tuple(a1, a2, a3),
            new Dictionary<Var, IValue>
            {
                { inputVar0, Value.FromTensor(input0) },
                { inputVar1, Value.FromTensor(input1) },
                { inputVar2, Value.FromTensor(input2) },
            });
    }

    [Fact]
    public async Task MergeFusionTupleWithSameInput()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var a1 = MakeSingleSimpleFusionCall(Abs, inputVar0);
        var a2 = MakeSingleSimpleFusionCall(Abs, inputVar0);
        var a3 = MakeSingleSimpleFusionCall(Abs, inputVar0);
        TestMatched<MergeTupleFusion>(new IR.Tuple(a1, a2, a3),
            new Dictionary<Var, IValue> { { inputVar0, Value.FromTensor(input0) } });
    }

    [Fact]
    public async Task MergeUserWithSameInput()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var input1 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar1 = new Var(new TensorType(input1.ElementType, input1.Shape));
        var s0 = Softmax(inputVar0, 0);
        var s1 = Softmax(inputVar1, 0);
        var s2 = Softmax(inputVar1, 1);
        var call = MakeSimpleFusionCall(expr => expr[0] + expr[1], s0, s1);
        var user = MakeSimpleFusionCall(expr => expr[0] / expr[1] + expr[2], call, s1, s2);
        await RunTest(
            user,
            new[] { inputVar0, inputVar1 },
            new Dictionary<Var, IValue>
            {
                { inputVar0, Value.FromTensor(input0) },
                { inputVar1, Value.FromTensor(input1) }
            });
    }

    [Fact]
    public async Task TestTupleGetItemFusionSimple()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var call = MakeSingleSimpleFusionCall(expr => new IR.Tuple(expr + 1f, expr - 1f), Softmax(inputVar0, 0));
        var abs0 = MakeSingleSimpleFusionCall(Abs, call[0]);
        var abs1 = MakeSingleSimpleFusionCall(Abs, call[1]);
        await RunTest(new IR.Tuple(new[] { abs0, abs1 }), new[] { inputVar0 },
            new Dictionary<Var, IValue> { { inputVar0, Value.FromTensor(input0) } });
    }

    [Fact]
    public async Task TestTupleGetItemMultiUser()
    {
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var call = MakeSingleSimpleFusionCall(expr => new IR.Tuple(expr + 1f, expr - 1f), Softmax(inputVar0, 0));
        var abs00 = MakeSingleSimpleFusionCall(Abs, call[0]);
        var abs01 = MakeSingleSimpleFusionCall(Abs, call[0]);
        var abs10 = MakeSingleSimpleFusionCall(Abs, call[1]);
        var abs11 = MakeSingleSimpleFusionCall(Abs, call[1]);
        await RunTest(new IR.Tuple(new[] { abs00, abs01, abs10, abs11 }), new[] { inputVar0 },
            new Dictionary<Var, IValue> { { inputVar0, Value.FromTensor(input0) } });
    }

    [Fact]
    public async Task TestGetItemWithRing()
    {
        // 29用到了28，所以其实是有环的
        // %26 = %ShapeOf_269_Gather_270_Gather_272(%23) 2 -1002975247: // (i64, i64)
        // %27 = GetItem(%26, const(i32 : 0)) 1 -1105276988: // i64
        // %28 = %Reshape_271(%27) 2 761954617: // i64[1]
        // %29 = GetItem(%26, const(i32 : 1)) 2 195526821: // i64
        // %30 = %Binary_166_MatMul_1(%23) 2 1373803502: // f32[?,?,?]
        // %31 = %Reshape_276_Binary_275_Gather_274_ShapeOf_273(%30) 1 1087046740: // i64[1]
        // %32 = (const(i64[1] : {-1L}), %31, const(i64[1] : {24L})): // (i64[1], i64[1], i64[1])
        // %33 = %Concat_277(%32) 2 708748629: // i64[3]
        // %34 = %MatMul_0_MatMul_2(%23, %24, %25, %28, %29, %30, %33) 1 1042934643: // f32[?,?,?]
        // %39 = %MatMul_3_MatMul_4_Binary_167_MatMul_5_Binary_168(%23, %37, %35, %38, %33, %28, %29) 1 274933965: // f32[?,?,?]
        var input0 = Testing.Rand<float>(1, 3, 24, 24);
        var inputVar0 = new Var(new TensorType(input0.ElementType, input0.Shape));
        var s = Softmax(inputVar0, 0);
        // 26
        var call = MakeSingleSimpleFusionCall(expr => new IR.Tuple(expr + 1f, expr - 1f), s);
        // 27
        var a0 = call[0];
        // 28
        var abs = MakeSingleSimpleFusionCall(Abs, a0);
        // 29
        var a1 = call[1];
        // 39
        var compute0 = MakeSimpleFusionCall(expr => expr[0] * expr[1], a1, abs);
        var compute1 = MakeSimpleFusionCall(expr => expr[0] * expr[1], a1, abs);
        var res = MakeSimpleFusionCall(expr => expr[0] + expr[1], compute0, compute1);
        await RunTest(res, new[] { inputVar0 },
            new Dictionary<Var, IValue> { { inputVar0, Value.FromTensor(input0) } });
    }
    private static async Task RunTestNotMatch(Expr body, Var[] inputVar, Dictionary<Var, IValue> dict)
    {
        var module = MakeModule(body, inputVar);
        var preResult = body.Evaluate(dict);
        var preHash = body.GetHashCode();
        var post = await new MergeBucketFusion().RunAsync(module, new());
        var postHash = ((Function)post.Entry!).Body.GetHashCode();
        Assert.Equal(postHash, preHash);
    }

    private static async Task RunTest(Expr body, Var[] inputVar, Dictionary<Var, IValue> dict)
    {
        var module = MakeModule(body, inputVar);
        DumpScope.Current.DumpIR(module.Entry!, "origin");
        var preResult = body.Evaluate(dict);
        var preHash = body.GetHashCode();
        var post = await new MergeBucketFusion().RunAsync(module, new());
        DumpScope.Current.DumpIR(post.Entry!, "post");
        var newBody = ((Function)post.Entry!).Body;
        var postHash = newBody.GetHashCode();
        Assert.NotEqual(postHash, preHash);
        var postResult = ((Function)(post.Entry!)).Body.Evaluate(dict);
        if (!Comparator.AllEqual(preResult, postResult))
        {
            ValueDumper.DumpTensors(preResult.AsTensors().Select(Value.FromTensor).ToArray(),
                Path.Join(DumpScope.Current.Directory, "preResult"));
            ValueDumper.DumpTensors(postResult.AsTensors().Select(Value.FromTensor).ToArray(),
                Path.Join(DumpScope.Current.Directory, "postResult"));
            // var list = preResult.AsTensors().Zip(postResult.AsTensors()).ToArray();
            // for (int i = 0; i < list.Length; i++)
            // {
            //
            // }
            Comparator.Compare(preResult, postResult);
        }

        var visitor = new FusionCounterVisitor();
        visitor.Visit(newBody);
        Assert.Equal(1, visitor.Count);
    }

    private static IRModule MakeModule(Expr output, Var[] inputVar) => new(new Function("main", output, inputVar));

    private static Call MakeSingleSimpleFusionCall(Func<Expr, Expr> ctor, Expr arg)
    {
        var v = new Var(arg.CheckedType);
        var abs = ctor(v);
        var f = new BucketFusion("stackvm", abs, new[] { v }, new Var[] { });
        var c = new Call(f, arg);
        return c;
    }

    private static Call MakeSimpleFusionCall(Func<Expr[], Expr> ctor, params Expr[] args)
    {
        var paramList = args.Select(x => new Var(x.CheckedType)).ToArray();
        var abs = ctor(paramList);
        var f = new BucketFusion("stackvm", abs, paramList, new Var[] { });
        var c = new Call(f, args);
        return c;
    }
}