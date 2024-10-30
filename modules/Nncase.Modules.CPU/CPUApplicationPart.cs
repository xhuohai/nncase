﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Nncase.Hosting;

namespace Nncase;

/// <summary>
/// CPU application part extensions.
/// </summary>
public static class CPUApplicationPart
{
    /// <summary>
    /// Add CPU assembly.
    /// </summary>
    /// <param name="registrator">Service registrator.</param>
    /// <returns>Configured service registrator.</returns>
    public static IRegistrator AddCPU(this IRegistrator registrator)
    {
        return registrator.RegisterModule<CPUModule>()
            .RegisterModule<Evaluator.IR.CPU.CPUModule>()
            .RegisterModule<Evaluator.TIR.CPU.CPUModule>()
            .RegisterModule<Evaluator.CustomCPU.CPUModule>();
    }
}
