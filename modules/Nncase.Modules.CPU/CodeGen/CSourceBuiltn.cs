// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.CPU;

public static class CSourceBuiltn
{

    public const string BufferType = "buffer_t";

    public const string BufferStruct = @"typedef struct buffer {
    void *vaddr;
    size_t paddr;
    int *shape;
    int *stride;
    int rank;
} buffer_t;";

    public const string MethodTable = @"typedef struct nncase_method_table {
  float (*float_unary_asin)(float);
} nncase_mt_t;";

    public const string Include = @"#include <stdint.h>
#include <stddef.h>
";

    public const string FixedParameters = "nncase_mt_t* nncase_mt, void* data, void* rdata";

    public const string MainPrologue = $@"void _start(char* name, buffer_t** buffers, {FixedParameters}) {{";

    public const string MainEpilogue = @"}";

    public static string Header => $@"
{Include}

{MethodTable}

{BufferStruct}

int strcmp(const char* s1,const char* s2) {{
  while(*s1 && *s2) {{
    if(*s1 != *s2) {{
      break;
    }}
    s1++;
    s2++; 
  }}
  return (*s1 - *s2) || (*s1 - '\0') || (*s2 - '\0'); 
}}

static nncase_mt_t *nncase_mt;
";


}