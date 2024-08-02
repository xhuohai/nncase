// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensorTypes
{
    [Fact]
    public void TestTensorTypeEquality()
    {
        var a = new TensorType(DataTypes.Float32, new[] { 1, 2, 3 });
        var b = new TensorType(DataTypes.Float32, new[] { 1, 2, 3 });
        Assert.Equal(a, b);
        Assert.StrictEqual(b, a);
        Assert.Equal(b.GetHashCode(), a.GetHashCode());
    }
}
