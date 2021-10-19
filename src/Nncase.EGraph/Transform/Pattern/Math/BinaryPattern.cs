﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Transform.Pattern.Math
{
    /// <summary>
    /// Binary expression.
    /// </summary>
    public record BinaryPattern(BinaryOp BinaryOp) : OpPattern(ImmutableArray.Create(new ParameterInfoPattern("lhs"), new ParameterInfoPattern("rhs")))
    {
    }
}