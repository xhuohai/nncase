// llama 65B
// seq-len = 384, w-len = 8192, heads = 64, head-len = 128
// 1 chips, 8 blocks per chip, 4 threads per block

#include "thread_context.h"
using namespace shared;

static bool w_loaded;
static tensor<float> v2_w({8, 2048, 128});
static tensor<float> v16_w({8, 2048, 128});
static tensor<float> v31_w({8, 2048, 128});
static tensor<float> v35_w({1024, 2048});
static tensor<float> v3_data({384, 128});
static tensor<float> v11_data({384, 128});
static tensor<int64_t> position_ids({1, 384});

void stage1_kernel(tensor<float, loc_t::device> &Hidden_in, /* [1, 384, 8192] */
                   tensor<float, loc_t::device> &V0_gamma,  /* [8192] */
                   tensor<float, loc_t::device> &V0_beta,   /* [8192] */
                   tensor<float, loc_t::device> &V2_w,     /* [64, 8192, 128] */
                   tensor<float, loc_t::device> &V16_w,    /* [64, 8192, 128] */
                   tensor<float, loc_t::device> &V31_w,    /* [64, 8192, 128] */
                   tensor<float, loc_t::device> &V35_w,    /* [8192, 8192] */
                   tensor<float, loc_t::device> &V3_data,  /* [384, 128] */
                   tensor<float, loc_t::device> &V11_data, /* [384, 128] */
                   tensor<float, loc_t::device> &Attn_mask, /* [1,1,384,384] */
                   tensor<int64_t, loc_t::device> &Position_ids /* [1,384] */
) {
    thread_context ctx(bid, tid);
    tensor<float> v0_gamma({2048});
    tensor<float> v0_beta({2048});
    tensor<float> v0({1, 48, 2048}); /* [1, 384, 8192] [1, 48@b, 2048@t]  */

    tensor<float> attn_mask(
        {1, 1, 384, 384}); /* [1,1,384,384] [1,1,96@t,96@t] */

    if (!w_loaded) {
        tdma_load_async(v0_gamma, V0_gamma({tid * 2048}, {2048}), ctx);
        tdma_load_async(v0_beta, V0_beta({tid * 2048}, {2048}), ctx);
        tdma_load_async(v2_w, V2_w({8 * bid, 2048 * tid, 0}, {8, 2048, 128}),
                        ctx);
        tdma_load_async(v16_w, V16_w({8 * bid, 2048 * tid, 0}, {8, 2048, 128}),
                        ctx);
        tdma_load_async(v31_w, V31_w({8 * bid, 2048 * tid, 0}, {8, 2048, 128}),
                        ctx);
        tdma_load_async(v35_w, V35_w({1024 * bid, 2048 * tid}, {1024, 2048}),
                        ctx);
        tdma_load_async(v3_data, std::move(V3_data), ctx);
        tdma_load_async(v11_data, std::move(V11_data), ctx);
        tdma_load_async(position_ids, std::move(Position_ids), ctx);
        tdma_load_async(attn_mask,
                        Attn_mask({0, 0, tid * 96, tid * 96}, {1, 1, 96, 96}),
                        ctx);

        tdma_wait(ctx);
    }

    tdma_load_async(v0, Hidden_in({0, bid * 48, tid * 2048}, {1, 48, 2048}),
                    ctx);
    {
        tensor<float> v0_sum({1, 48});
        tensor<float> v0_sum_sqr({1, 48});
        reduce_sum_sqr(v0, v0_sum, v0_sum_sqr);
        tdma_reduce_async(v0_sum, v0_sum, reduce_op_t::sum, ctx);
        tdma_reduce_async(v0_sum_sqr, v0_sum_sqr, reduce_op_t::sum, ctx);
        layernorm(v0, v0_sum, v0_sum_sqr, 2, 8192);
    } // v0 [1, 384, 8192] [1, 48@b, 8192]

    auto v1 = unsqueeze(v0); // [1, 1, 384, 8192] [1, 1, 48@b, 2048@t]
    /* 这里如果V2不shared的话可以只做thread间的reduce */
    tensor_block_mma_sync(v1, v2_w, V2, false, ctx);

    tensor<float> v3({1, 384, 128}); //
    gather(v3_data, position_ids, v3, 0);

    auto v4 = unsqueeze(v3); /* 1, 1, 384, 128 */

    tensor<float> v5({1, 8, 384, 128}); // [1, 64, 384, 128] [1, 8@b, 384, 128]
    binary(V2, v4, v5, binary_op_t::mul);

    auto v6 =
        V2({0, 0, 0, 64}, {1, 8, 384, 64}); //[1, 64, 384, 64] [1, 8@b, 384, 64]
    auto v7 = v6;
    if (tid == 0) {
        /* V2 is shared, so don't need perfrom on each thread. */
        unary(v6, v7, unary_op_t::neg);
    }
    tdma_wait(ctx);

    // auto v8 = v2({0, 0, 0, 0}, {1, 8, 384, 64}); //[1, 64, 384, 64] [1, 8@b,
    // 384, 64]@shared
    /* v10 为concat(v7,v8), 实际来源于V2，因此无需实际动作 */
    // [1, 64, 384, 128] [1, 8@b, 384, 128]@shared ->
    // [1, 8@b, 96@t, 128]@shared here can resplit .
    auto v10 = V2({0, 0, tid * 96, 0}, {1, 8, 96, 128});

    tensor<float> v11({1, 384, 128}); //[1, 384, 128] [1, 384, 128]
    gather(v11_data, position_ids, v11, 0);
    auto v12 = unsqueeze(v11); // [1, 1, 384, 128] [1, 1, 384, 128]

    tensor<float> v13({1, 8, 96, 128}); // [1, 64, 384, 128] [1, 8@b, 96@t, 128]
    binary(v10, v12, v13, binary_op_t::mul);

    tensor<float> v14({1, 8, 96, 128}); // [1, 64, 384, 128] [1, 8@b, 96@t, 128]
    binary(v5, v13, v14, binary_op_t::add);

    // [1, 1, 384, 8192] [1, 1, 48@b, 2048@t]
    tensor<float> v15 = v0({0, bid * 48, tid * 2048}, {1, 48, 2048});

    tensor_block_mma_sync(v15, v16_w, V16, false, ctx);
    tensor<float> v17({1, 8, 384, 128}); // [1, 64, 384, 128] [1, 8@b, 384, 128]
    binary(V16, v4, v17, binary_op_t ::mul);

    // [1, 64, 384, 64] [1, 8@b, 384, 64]@shared
    auto v18 = V16({0, 0, 0, 64}, {1, 8, 384, 64});
    if (tid == 0) {
        unary(v18, v18, unary_op_t::neg);
    }
    tdma_wait(ctx);

    //[1, 64, 384, 128] [1, 8@b, 96@t, 128]@shared
    auto v22 = V16({0, 0, tid * 96, 0}, {1, 8, 96, 128});

    // [1, 64, 384, 128] [1, 8@b, 96@t, 128]@shared
    tensor<float> v23({1, 8, 96, 128});
    binary(v22, v12, v23, binary_op_t::mul);

    // [1, 64, 384, 128] [1, 8@b, 96@t, 128]
    tensor<float> v24({1, 8, 96, 128});
    binary(v17, v23, v24, binary_op_t::add);

    // [1, 64, 128, 384] [1, 8@b, 128, 96@t]
    tensor<float> v25({1, 8, 128, 96});
    transpose(v24, v25, dims_t({0, 1, 3, 2}));

    // [1, 8@b, 96@t, 128] @  [1, 8@b, 128, 96@t] => [1, 8@b, 96@t, 96@t]
    auto v26 = V26({0, 0, tid * 96, tid * 96}, {1, 8, 96, 96});
    matmul(v14, v25, v26);

    // [1, 64, 384, 384] [1, 8@b, 96@t, 96@t] @shared
    auto v27 = v26;
    auto v26_c = tensor<float>({1, 1, 1, 1});
    tdma_fill_async(v26_c, 11.313708f);
    binary(v26, v26_c, v27, binary_op_t::div);

    // [1, 64, 384, 384] [1, 8@b, 96@t, 96@t] @shared
    auto v28 = v27;
    binary(v27, attn_mask, v28, binary_op_t::div);

    // resplit need sync.
    tdma_wait(ctx);
    auto v28_1 = V26({0, 0, tid * 96, 0}, {1, 8, 96, 384});
    tensor<float> v29({1, 8, 96, 384}); //[1, 64, 384, 384] [1, 8@b, 96@t, 384]
    softmax(v28_1, v29, 3);

    //  [1, 1, 384, 8192] [1, 1, 48@b, 2048@t]
    auto v30 = unsqueeze(v0);

    tensor_block_mma_sync(v30, v31_w, V31, false, ctx);

    // [1, 8@b, 384, 128]@local @ [1, 8@b, 384, 128]@shared
    // [1, 64, 384, 128] [1, 8@b, 384, 128]
    if (tid == 0) {
        matmul(v29, V31, V32);
        // V33 [1, 384, 64, 128] [1, 384, 8@b, 128]@shared
        transpose(V32, V33, dims_t({0, 2, 1, 3}));
    }
    tdma_wait(ctx);
    // auto v34 =
}
