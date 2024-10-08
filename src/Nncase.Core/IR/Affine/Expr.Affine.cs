﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Affine;

namespace Nncase.IR;

public partial class Expr
{
    public Load this[AffineMap region] => F.Affine.Load(this, region);
}