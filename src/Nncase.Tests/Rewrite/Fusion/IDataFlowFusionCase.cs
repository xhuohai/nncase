using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;

internal static class FusionBuilder
{
    static int Count = 0;

    public static Fusion MakeConv2DFusion(bool mask)
    {
        var fusion_1_input = new Var($"fusion_{Count}_input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var weights = IR.F.Random.Normal(DataTypes.Float32, 0, 1, Count, new[] { 3, 3, 1, 1 }).Evaluate().AsTensor();
        var bias = IR.F.Random.Normal(DataTypes.Float32, 0, 1, Count, new[] { 3 }).Evaluate().AsTensor();
        var fusion_1 = new Fusion($"fusion_{Count}_{mask}", Callable.StackVMModuleKind, IR.F.NN.Conv2D(fusion_1_input, weights, bias, new[] { 1, 1 }, new[,] { { 0, 0 }, { 0, 0 } }, new[] { 1, 1 }, PadMode.Constant, 1), new[] { fusion_1_input });
        Count++;
        return fusion_1;
    }

    public static Fusion MakeBinaryFusion(BinaryOp binaryOp, bool mask)
    {
        var fusion_2_input = new Var[] { new($"fusion_{Count}_input_lhs", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 })), new($"fusion_{Count}_input_rhs", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 })) };
        var fusion_2 = new Fusion($"fusion_{Count}_{mask}", Callable.StackVMModuleKind, IR.F.Math.Binary(binaryOp, fusion_2_input[0], fusion_2_input[1]), fusion_2_input);
        Count++;
        return fusion_2;
    }

    public static Call MakeMultiSingleCall(Expr input, bool[] masks)
    {
        var last_output = input;
        foreach (var mask in masks)
        {
            last_output = new Call(MakeConv2DFusion(mask), last_output);
        }
        return (Call)last_output;
    }

}

public interface IDataFlowFusionCase
{
    Expr BuildBody(Var input);

    int FinalFusionCount { get; }
}

/// <summary>
/// cycle type 0:
///  x = fusion1(input)         
///  y = fusion2(x)        =>  
///  z = fusion3(y)          z = fusion3_2_1(input)
/// </summary>    
internal class DataFlowType0FusionCase : IDataFlowFusionCase
{
    public static Expr BuildBodyCore(Expr input)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(true), v_1);
        return v_2;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
    }

    public int FinalFusionCount => 1;
}


/// <summary>
/// cycle type 0:
///  v0 = fusion1(input)         
///  v1 = fusion2(v0)        =>  
///  v2 = fusion3(v1)            v2 = fusion3_2_1(input)
///  v3 = fusion4(v2)            v3 = fusion4(v2)
///  v4 = fusion5(v3)            v4 = fusion5(v3)
/// </summary>    
internal class DataFlowType0NotFusionCase : IDataFlowFusionCase
{
    public static Expr BuildBodyCore(Expr input)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(true), v_1);
        var v_3 = new Call(FusionBuilder.MakeConv2DFusion(false), v_2);
        var v_4 = new Call(FusionBuilder.MakeConv2DFusion(true), v_3);
        return v_4;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input);
    }

    public int FinalFusionCount => 3;
}


/// <summary>
/// cycle type 1:
///             input
///            /    \
///         /         \
///        |      y = fusion2(input)
///         \        /
///          \     /
///     fusion3(x,y)
/// </summary>    
internal class DataFlowType1FusionCaseRight : IDataFlowFusionCase
{
    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
            return new Call(fusion_3, new[] { v_0, input }); // 1,3,224,224
        return new Call(fusion_3, new[] { input, v_0 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }

    public int FinalFusionCount => 1;
}

internal sealed class DataFlowType1FusionCaseLeft : DataFlowType1FusionCaseRight
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }
}

/// <summary>
/// cycle type 2:
///             input                         
///            /    \                         
///         /         \                       
///        |      v0 = fusion1(input)         
///        |      v1 = fusion2(v0)            
///        |      v2 = fusion3(v1)            
///         \        /                        
///          \     /                          
///     fusion3(input,v2)            =>         fusion?(input)                           
/// </summary>    
internal class DataFlowType2FusionCaseLeft : IDataFlowFusionCase
{

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(true), v_1);

        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
            return new Call(fusion_3, new[] { v_2, input }); // 1,3,224,224
        return new Call(fusion_3, new[] { input, v_2 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }

    public int FinalFusionCount => 1;
}

internal sealed class DataFlowType2FusionCaseRight : DataFlowType2FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

/// <summary>
/// cycle type 3:
///             input                                      input                         
///            /    \                                     /    \                         
///         /         \                                /         \                       
///        |      v0 = fusion1(input)                 |           |   
///        |      v1 = fusion2(v0)                    |      v1 = fusion2_1(input)            
///        |      v2 = fusion3(v1)                    |      v2 = fusion3(v1)            
///         \        /                                 \        /                        
///          \     /                                    \     /                          
///     fusion3(input,v2)            =>              fusion3(input,v2)
/// </summary>    
internal class DataFlowType3FusionCaseLeft : IDataFlowFusionCase
{

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(false), v_1);

        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
            return new Call(fusion_3, new[] { v_2, input }); // 1,3,224,224
        return new Call(fusion_3, new[] { input, v_2 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }

    public int FinalFusionCount => 3;
}

internal sealed class DataFlowType3FusionCaseRight : DataFlowType3FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}


/// <summary>
/// cycle type 3:
///             input                                      input                         
///            /    \                                     /    \                         
///         /         \                                /         \                       
///        |      v0 = fusion1(input)                 |      v1 = fusion2_1(input)
///        |      v1 = fusion2(v0)                    |           |
///        |      v2 = fusion3(v1)                    |      v2 = fusion3(v1)
///        |      v3 = fusion4(v2)                    |           |
///        |      v4 = fusion5(v3)                    |      v4 = fusion5_4(v2)
///         \        /                                 \        /                        
///          \     /                                    \     /                          
///     fusion6(input,v4)            =>              fusion6(input,v4)
/// </summary>    
internal class DataFlowType4FusionCaseLeft : IDataFlowFusionCase
{

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v_0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v_1 = new Call(FusionBuilder.MakeConv2DFusion(true), v_0);
        var v_2 = new Call(FusionBuilder.MakeConv2DFusion(false), v_1);
        var v_3 = new Call(FusionBuilder.MakeConv2DFusion(true), v_2);
        var v_4 = new Call(FusionBuilder.MakeConv2DFusion(true), v_3);

        var fusion_3 = FusionBuilder.MakeBinaryFusion(BinaryOp.Add, true);
        if (left)
            return new Call(fusion_3, new[] { v_4, input }); // 1,3,224,224
        return new Call(fusion_3, new[] { input, v_4 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }

    public int FinalFusionCount => 4;
}

internal sealed class DataFlowType4FusionCaseRight : DataFlowType4FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}


/// <summary>
/// cycle type 5 = type 2 + fusion:
///             input                         
///            /    \                         
///         /         \                       
///        |      v0 = fusion1(input)         
///        |      v1 = fusion2(v0)            
///        |      v2 = fusion3(v1)            
///         \        /                        
///          \     /                          
///     v3 = fusion4(input,v2)            =>       fusion?(input)
///             |
///         fusion5(v3)
/// </summary>    
internal class DataFlowType5FusionCaseLeft : IDataFlowFusionCase
{

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v3 = DataFlowType2FusionCaseLeft.BuildBodyCore(input, left);
        return new Call(FusionBuilder.MakeConv2DFusion(true), new[] { v3 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }

    public int FinalFusionCount => 1;
}

internal class DataFlowType5FusionCaseRight : DataFlowType5FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

/// cycle type 6 = type 3 + fusion:
///             input                                      input                         
///            /    \                                     /    \                         
///         /         \                                /         \                       
///        |      v0 = fusion1(input)                 |           |   
///        |      v1 = fusion2(v0)                    |      v1 = fusion2_1(input)            
///        |      v2 = fusion3(v1)                    |      v2 = fusion3(v1)            
///         \        /                                 \        /                        
///          \     /                                    \     /                          
///     v3 = fusion4(input,v2)            =>            fusion5_4(input,v2)
///             |
///     fusion5(input,v3)                            
/// </summary>    
internal class DataFlowType6FusionCaseLeft : IDataFlowFusionCase
{

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v3 = DataFlowType3FusionCaseLeft.BuildBodyCore(input, left);
        return new Call(FusionBuilder.MakeConv2DFusion(true), new[] { v3 }); // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }

    public int FinalFusionCount => 3;
}

internal class DataFlowType6FusionCaseRight : DataFlowType6FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}

///             input                 
///        v0 = fusion0(input)        
///        v1 = fusion1(v0)                    v0 = fusion1_0(input)
///            /    \                               /    \                         
///         /         \                          /         \                       
///        |      v2 = fusion2(v1)             |            |   
///        |      v3 = fusion3(v2)             |       v2 = fusion3_2(v1)            
///        |      v4 = fusion4_f(v3)           |       v3 = fusion4_f(v2)
///         \        /                           \        /                        
///          \     /                              \     /                          
///     fusion5(input,v4)            =>          fusion9_8(v0,v3)
internal class DataFlowType6_1FusionCaseLeft : IDataFlowFusionCase
{

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v0 = new Call(FusionBuilder.MakeConv2DFusion(true), input);
        var v1 = new Call(FusionBuilder.MakeConv2DFusion(true), v0);
        var v3 = DataFlowType3FusionCaseLeft.BuildBodyCore(v1, left);
        // return new Call(FusionBuilder.MakeConv2DFusion(true), new[] { v3 }); // 1,3,224,224
        return v3;
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }

    public int FinalFusionCount => 4;
}

internal class DataFlowType6_1FusionCaseRight : DataFlowType6_1FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}


/// cycle type 7 : type 5 + 6
///             input                         
///            /    \                         
///         /         \                       
///        |      v0 = fusion0(input)         
///        |      v1 = fusion1(v0)            
///        |      v2 = fusion2(v1)            
///         \        /                        
///          \     /                          
///     v3 = fusion3(input,v2)          =>        
///     v4 = fusion4(v3)                          v4 = fusion4_3_2_1_0(input)
///            /    \                                     /    \                         
///         /         \                                /         \                       
///        |      v5 = fusion5(v4)                    |           |   
///        |      v6 = fusion6(v5)                    |      v6 = fusion6_5(v4)            
///        |      v7 = fusion7_f(v6)                  |      v7 = fusion7_f(v6)
///         \        /                                 \        /                        
///          \     /                                    \     /                          
///     v9 = fusion8(v4,v7)            =>           v10 = fusion9_8(v4,v7)
///             |
///     v10 = fusion9(v9)                            
/// </summary>    
internal class DataFlowType7FusionCaseLeft : IDataFlowFusionCase
{

    public static Expr BuildBodyCore(Expr input, bool left)
    {
        var v3 = DataFlowType5FusionCaseLeft.BuildBodyCore(input, left);
        var v9 = DataFlowType6FusionCaseLeft.BuildBodyCore(v3, !left);
        return v9; // 1,3,224,224
    }

    public Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, true);
    }

    public int FinalFusionCount => 4;
}

internal class DataFlowType7FusionCaseRight : DataFlowType7FusionCaseLeft
{
    public new Expr BuildBody(Var input)
    {
        return BuildBodyCore(input, false);
    }
}