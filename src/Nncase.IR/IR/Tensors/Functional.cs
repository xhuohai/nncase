﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Tensors;

namespace Nncase.IR.F
{
    /// <summary>
    /// NN functional helper.
    /// </summary>
    public static class Tensors
    {
        /// <summary>
        /// Call sigmoid.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Transpose(Expr input, Expr perm) => new Call(new Transpose(), input, perm);

        public static Call Concat(Tuple input, Expr axis) => new Call(new Concat(), input, axis);
    }
}
