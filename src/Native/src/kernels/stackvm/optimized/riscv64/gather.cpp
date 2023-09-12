/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../opt_ops.h"
#include <cstring>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;
#include <stdio.h>
#include "../../debug_info.h"
#include <typeinfo>
#include <cxxabi.h>

#define _STR(x) #x
#define STR(x) _STR(x)
#define _CONNECT(a, b) a##b
#define CONNECT(a, b) _CONNECT(a, b)
#define vsetvli_macro(evl, avl, elen, mlen)                                             \
    "vsetvli " STR(evl) "," STR(avl) "," STR(CONNECT(e, elen)) "," STR(CONNECT(m, mlen)) ";"
#define vle_len_macro(eew,vd, rs) \
STR(CONNECT(vle, eew)) ".v" " " STR(vd) "," STR(rs) ";"

#define vse_len_macro(eew,vd, rs) \
STR(CONNECT(vse, eew)) ".v" " " STR(vd) "," STR(rs) ";"

#define vluxei_len_macro(ilen,vd, rs, vindex) \
STR(CONNECT(vluxei, ilen)) ".v" " " STR(vd) "," STR(rs) "," STR(vindex) ";"

#define vsllvi_len_macro(vd,vsrc,shift_bits) \
"vsll.vi " STR(vd) ", " STR(vsrc) ", " STR(shift_bits) ";"

#define slli_len_macro(rd,rs,shift_bits) \
"slli " STR(rd) ", " STR(rs) ", " STR(shift_bits) ";"

#define REGISTER_GATHER_IMPL(src_date_type_len, index_date_type_len, src_lmul, index_lmul, src_bit_shift, index_bit_shift)    \
static void s##src_date_type_len##_i##index_date_type_len##_impl(const void*src, void* dst, const void* index_ptr, int index_count)   \
{                                                                                             \
            __asm volatile(                                                                   \
            "mv a0, %[index_count];"                                                          \
            "mv a2, %[dst];"                                                                  \
            "mv a3, %[index_ptr];"                                                            \
        "gater_STRAT%=:;"                                                                     \
            vsetvli_macro(t0, a0, index_date_type_len, index_lmul)                            \
            "sub a0, a0, t0;"                                                                 \
            vle_len_macro(index_date_type_len,v8,(a3))                                        \
            vsllvi_len_macro(v8,v8,src_bit_shift)                                             \
            slli_len_macro(t1,t0, index_bit_shift)                                            \
            "add a3,a3, t1;"                                                                  \
            vsetvli_macro(x0, x0, src_date_type_len, src_lmul)                                \
            vluxei_len_macro(index_date_type_len,v16, (%[src]), v8)                           \
            vse_len_macro(src_date_type_len,v16, (a2))                                        \
            slli_len_macro(t1,t0, src_bit_shift)                                              \
            "add a2,a2, t1;"                                                                  \
            "bnez a0, gater_STRAT%=;"                                                         \
            :                                                                                 \
            :[src] "r"(src),[index_count]"r"(index_count),                                    \
            [dst] "r"(dst), [index_ptr] "r"(index_ptr)                                        \
            : "t0", "a0", "a1", "a2", "a3","a4","t1","t2","t3", "ft0", "v8", "v16");          \
}                                                                                              



#define REGISTER_GATHER_IMPL2(src_date_type_len, index_date_type_len, src_lmul, index_lmul, src_bit_shift, index_bit_shift)    \
static void s##src_date_type_len##_i##index_date_type_len##_impl2(const void*src, void* dst, const void* index_ptr, int index_count, int block_size)   \
{                                                                                             \
            __asm volatile(                                                                   \
            "mv a0, %[index_count];"                                                          \
            "mv a1, %[block_size];"                                                           \
            "mv a2, %[dst];"                                                                  \
            "mv a3, %[index_ptr];"                                                            \
            "addi t4,x0, 8;"                                                                  \
        "gater_STRAT%=:;"                                                                     \
            vsetvli_macro(t0, a0, index_date_type_len, index_lmul)                            \
            "sub a0, a0, t0;"                                                                 \
            vle_len_macro(index_date_type_len,v8,(a3))                                        \
            vsllvi_len_macro(v8,v8,src_bit_shift)                                             \
            slli_len_macro(t1,t0, index_bit_shift)                                            \
            "add a3,a3, t1;"                                                                  \
            "mv a4, a2;"                                                                      \
        "blocke_loop%=:;"                                                                     \
            vsetvli_macro(x0, x0, src_date_type_len, src_lmul)                                \
            vluxei_len_macro(index_date_type_len,v16, (%[src]), v8)                           \
            "vsse32.v v16, (a4), t4;"                                                         \
            "addi a1, a1, -1;"                                                                \
            "beqz a1, over%=;"                                                                \
            "addi a4, a4, 4;"                                                                 \
            vsetvli_macro(x0, x0, index_date_type_len, index_lmul)                            \
            "vadd.vi v8, v8, 4;"                                                              \
            "j blocke_loop%=;"                                                                \
        "over%=:;"                                                                            \
            slli_len_macro(t1,t0, src_bit_shift)                                              \
            "add a2,a2, t1;"                                                                  \
            "bnez a0, gater_STRAT%=;"                                                         \
            :                                                                                 \
            :[src] "r"(src),[index_count]"r"(index_count),                                    \
            [dst] "r"(dst), [index_ptr] "r"(index_ptr), [block_size]"r"(block_size)           \
            : "t0", "a0", "a1", "a2", "a3","a4","t1","t2","t3","t4", "ft0", "v8", "v16");     \
}

REGISTER_GATHER_IMPL(32, 64, 1, 2, 2, 3)
REGISTER_GATHER_IMPL(64, 64, 2, 2, 3, 3)

REGISTER_GATHER_IMPL2(32, 64, 1, 2, 2, 3)

namespace {
template <class T, class IndicesT>
void kvx(const T*src, T* dst, const IndicesT* index_ptr, int index_count, int block_size)
{
    #if(0)
    {
        for(int i = 0; i < index_count; ++i)
        {
            memcpy(dst + i * block_size, src + index_ptr[i] * block_size, block_size * sizeof(T));
        }
    }
    #else
        printf("+++++++++++risv64 -index_count : %d, block_size:%d, %d,%d\n", index_count, (int)(block_size), (int)(block_size* sizeof(T)), (int)sizeof(T));
            (void)block_size;
            (void)index_ptr;
            s32_i64_impl2(src, dst, index_ptr, index_count, block_size);
        // if(block_size * sizeof(T) * 8 == 32)
        // {
        //     s32_i64_impl(src, dst, index_ptr, index_count);
        // }
        // else if(block_size * sizeof(T) * 8 == 64)
        // {
        //     s64_i64_impl(src, dst, index_ptr, index_count);
        // }
        
        //     __asm volatile(
        //     "mv a0, %[index_count];" // count 
        //     "mv a2, %[dst];"
        //     "mv a3, %[index_ptr];"
        // "gater_STRAT%=:;"
        //     vsetvli_macro(t0, a0, 64, 2) //"vsetvli t0, a0, e64, m2;"
        //     "sub a0, a0, t0;"  // count --
        //     vle_len_macro(64,v8,(a3)) // "vle64.v v8, (a3);"
        //     "vsll.vi v8, v8, 2;"
        //     "slli t1,t0, 3;"
        //     "add a3,a3, t1;"
        //     vsetvli_macro(x0, x0, 32, 1) // "vsetvli x0, x0, e32, m1;"
        //     vluxei_len_macro(64,v16, (%[src]), v8)  // "vluxei64.v v16, (%[src]), v8;"
        //     vse_len_macro(32,v16, (a2)) //"vse32.v v16, (a2);"
        //     "slli t1,t0, 2;"
        //     "add a2,a2, t1;"
        //     "bnez a0, gater_STRAT%=;"
        //     // "mv %[__index_ount], t0;"
        //     ://[__index_ount]"=r"(__index_ount)
        //     :[src] "r"(src),[index_count]"r"(index_count),
        //     [dst] "r"(dst), [index_ptr] "r"(index_ptr)
        //     : "t0", "a0", "a1", "a2", "a3","a4","t1","t2","t3", "ft0", "v8", "v16");
    #endif
}

template <class T, class IndicesT>
result<void>
gather_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
            NNCASE_UNUSED gsl::span<const size_t> out_shape,
            NNCASE_UNUSED gsl::span<const size_t> in_strides,
            NNCASE_UNUSED gsl::span<const size_t> out_strides,
            const IndicesT *indices, gsl::span<const size_t> indices_shape,
            size_t axis, NNCASE_UNUSED kernel_context &context) noexcept {
                printf("-------------axis: %d -----------\n", (int)axis);
                const char* p = abi::__cxa_demangle(typeid(*indices).name(), nullptr, nullptr, nullptr);
                printf("1-------%s, %s， %d, %d\n", typeid(*indices).name(), p, 
                (int)sizeof(IndicesT), (int)sizeof(long));
                print_runtime_shape(in_shape, "in_shape");
                print_runtime_shape(out_shape, "out_shape");
                print_runtime_shape(indices_shape, "indices_shape");
                printf("!!!!!!!!!!!! call gather .... \n");

                print_vector_by_type(indices, compute_size(indices_shape), 16, "indices", 3);
                int len_a = compute_size(in_shape);
                int len_b = compute_size(out_shape);
                print_vector_by_type(input, len_a, 16, "input", 1);
    size_t outer_count =
        std::accumulate(in_shape.begin(), in_shape.begin() + axis, 1,
                        std::multiplies<size_t>{});
    auto indices_count = compute_size(indices_shape);
    size_t block_size =
        std::accumulate(in_shape.begin() + axis + 1, in_shape.end(), 1,
                        std::multiplies<size_t>{});
        printf("---------------outer_count: %d, block_size:%d\n", (int)outer_count, (int)block_size);

    auto *in_ptr = input;
    auto *out_ptr = output;
    for (size_t o = 0; o < outer_count; ++o) {
#ifdef NNCASE_OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
        #if(0)
        for (int i = 0; i < indices_count; ++i) {
            auto *o_ptr = out_ptr + i * block_size;
            auto indices_ptr =
                indices[i] >= 0 ? indices[i] : indices[i] + in_shape[axis];
            memcpy(o_ptr, in_ptr + (indices_ptr * block_size),
                   block_size * sizeof(T));
        }
        #else
        //printf("!!!!!!!!!!!!%f,%f,%f,%f\n", (float)in_ptr[0], (float)in_ptr[1], (float)in_ptr[2], (float)in_ptr[3]);
        //   print_vector_by_type(in_ptr, 4, 16, "!!!!!!in_ptr... ", 1);
            kvx(in_ptr, out_ptr, indices, indices_count, block_size);
            // for(int i = 0; i < indices_count; ++i)
            // {
            //     memcpy(out_ptr + i * block_size, in_ptr + (indices[i] >= 0 ? indices[i] : indices[i] + in_shape[axis]) * block_size, block_size * sizeof(T));
            // }
        #endif
        in_ptr += in_shape[axis] * block_size;
        out_ptr += indices_count * block_size;
    }
    print_vector_by_type(output, len_b, 16, "output... ", 1);
    return ok();
}
} // namespace

#define GATHER_IMPL(size, type)                                                \
    case size:                                                                 \
        return integer_cast(indices_type, indices, [&](auto &&indices_value) { \
            return gather_impl(reinterpret_cast<const type *>(input),          \
                               reinterpret_cast<type *>(output), in_shape,     \
                               out_shape, in_strides, out_strides,             \
                               indices_value, indices_shape, axis, context);   \
        });


result<void> nncase::kernels::stackvm::optimized::gather(
    datatype_t type, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> out_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    datatype_t indices_type, const gsl::byte *indices,
    gsl::span<const size_t> indices_shape, size_t axis,
    kernel_context &context) noexcept {
        
    TYPE_IMPL_SELECT(type, GATHER_IMPL);
}
