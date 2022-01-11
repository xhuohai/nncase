﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Nncase.Transform
{
    /// <summary>
    /// Options for running pass.
    /// </summary>
    public class RunPassOptions
    {
        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="target"> target device </param>
        /// <param name="dumpLevel"> int level </param>
        /// <param name="dumpDir"> dir </param>
        public RunPassOptions(ITarget target, int dumpLevel, string dumpDir)
        {
            Target = target;
            DumpLevel = dumpLevel;
            DumpDir = dumpDir;
            PassName = "";
        }
        /// <summary>
        /// copy construct
        /// </summary>
        /// <param name="other"></param>
        public RunPassOptions(RunPassOptions other)
        {
            Target = other.Target;
            DumpLevel = other.DumpLevel;
            DumpDir = other.DumpDir;
            PassName = other.PassName;
        }


        /// <summary>
        /// Gets target.
        /// </summary>
        public ITarget Target { get; }

        /// <summary>
        /// Dump level 0 = do nothing
        /// Dump level 1 = print to std output
        /// Dump level 2 = print dump to file
        /// </summary>
        public int DumpLevel { private set; get; }

        public string DumpDir { private set; get; }

        public string PassName { private set; get; }

        public RunPassOptions SetName(string name) { PassName = name; return this; }

        /// <summary>
        /// set the dumpDir
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public RunPassOptions SetDir(string path) { DumpDir = path; return this; }

        /// <summary>
        /// return "{DumpDir}/{PassName}"
        /// </summary>
        public string FullDumpDir { get => Path.Combine(DumpDir, PassName); }

        public static RunPassOptions Invalid => new RunPassOptions(null!, -1, "");
    }
}