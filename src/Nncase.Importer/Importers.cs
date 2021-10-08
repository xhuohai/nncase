﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Importer.TFLite;
using Nncase.IR;

namespace Nncase
{
    /// <summary>
    /// Graph importers.
    /// </summary>
    public static class Importers
    {
        /// <summary>
        /// Import tflite model.
        /// </summary>
        /// <param name="tflite">tflite model stream.</param>
        /// <returns>Imported IR module.</returns>
        public static Module ImportTFLite(Stream tflite)
        {
            var model = new byte[tflite.Length];
            tflite.Read(model);
            var importer = new TFLiteImporter(model);
            return importer.Import();
        }

        /// <summary>
        /// Import tflite model.
        /// </summary>
        /// <param name="tfliteFileName">tflite model file name.</param>
        /// <returns>Imported IR module.</returns>
        public static Module ImportTFLite(string tfliteFileName)
        {
            using (var model = File.OpenRead(tfliteFileName))
            {
                return ImportTFLite(model);
            }
        }
    }
}
