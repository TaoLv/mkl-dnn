/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
*******************************************************************************/

#include <stdlib.h>

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"
#include "rnn/rnn_cells.hpp"

namespace rnn {

/******************************************************************************/
/******************************* Copy Routines ********************************/
/******************************************************************************/
void prepare_bias(const prb_t &p, float *bias_with_compensation_,
        const float *bias_, const float *weights_layer_,
        const float *weights_iter_) {
    AOC<const float> weights_layer(
            weights_layer_, p.n_layer, p.n_dir(), p.slc, p.n_gates(), p.dhc);
    AOC<const float> weights_iter(
            weights_iter_, p.n_layer, p.n_dir(), p.sic, p.n_gates(), p.dhc);

    AOC<const float> bias(bias_, p.n_layer, p.n_dir(), p.n_gates(), p.dhc);
    AOC<float> bias_with_compensation(
            bias_with_compensation_, p.n_layer, p.n_dir(), p.n_gates(), p.dhc);

    for (int layer = 0; layer < p.n_layer; ++layer)
        for (int dir = 0; dir < p.n_dir(); ++dir)
            for (int gate = 0; gate < p.n_gates(); ++gate)
                for (int dhc = 0; dhc < p.dhc; ++dhc) {
                    float weights_compensation = 0;
                    for (int sic = 0; sic < p.sic; ++sic)
                        weights_compensation
                                += weights_iter(layer, dir, sic, gate, dhc);
                    for (int slc = 0; slc < p.slc; ++slc)
                        weights_compensation
                                += weights_layer(layer, dir, slc, gate, dhc);

                    float scale = p.data_scale;
                    if (p.scale_policy == policy_t::PER_OC)
                        scale *= p.wei_oc_scales[gate * p.dhc + dhc];
                    else if (p.scale_policy == policy_t::COMMON)
                        scale *= p.wei_scale;

                    bias_with_compensation(layer, dir, gate, dhc)
                            = bias(layer, dir, gate, dhc)
                            - weights_compensation * p.data_shift / scale;
                }
}

void copy_init_fwd(const prb_t &p, float *ws_, const float *src_layer_,
        const float *src_iter_, const float *src_iter_c_,
        rnn_iter_direction_t iter_dir, rnn_layer_direction_t lay_dir,
        int64_t dir_val) {
    AOC<float> ws(ws_, p.n_layer + 2, p.n_dir(), p.n_iter + 2, p.n_states(),
            p.mb * p.wc);
    AOC<const float> src_layer(src_layer_, p.n_iter, p.mb * p.slc);
    AOC<const float> src_iter(src_iter_, p.n_layer, p.n_dir(), p.mb * p.sic);
    AOC<const float> src_iter_c(
            src_iter_c_, p.n_layer, p.n_dir(), p.mb * p.dhc);

    int64_t lay_dest = (lay_dir == bottom2top) ? 0 : p.n_layer + 1;
    int64_t it_dest = (iter_dir == left2right) ? 0 : p.n_iter + 1;

    // Copy src_layer
    for (int64_t it = 0; it < p.n_iter; it++) {
        copy(p.mb, p.slc, p.slc, p.wc, &src_layer(it, 0),
                &ws(lay_dest, dir_val, it + 1, H, 0));
        if (p.is_int8())
            data_q10n(p.mb, p.slc, p.wc, &ws(lay_dest, dir_val, it + 1, H, 0),
                    p.data_scale, p.data_shift);
    }

    // Copy src_iter (and src_iter_c)
    for (int64_t lay = 0; lay < p.n_layer; lay++) {
        copy(p.mb, p.sic, p.sic, p.wc, &src_iter(lay, dir_val, 0),
                &ws(lay + 1, dir_val, it_dest, H, 0));
        if (p.is_int8())
            data_q10n(p.mb, p.sic, p.wc, &ws(lay + 1, dir_val, it_dest, H, 0),
                    p.data_scale, p.data_shift);

        if (p.alg == VANILLA_LSTM)
            copy(p.mb, p.dhc, p.dhc, p.wc, &src_iter_c(lay, dir_val, 0),
                    &ws(lay + 1, dir_val, it_dest, C, 0));
    }
}

void copy_res_fwd(const prb_t &p, float *dst_layer_, float *dst_iter_,
        float *dst_iter_c_, const float *ws_, rnn_iter_direction_t iter_dir,
        rnn_layer_direction_t lay_dir, int64_t dir_val, rnn_action_t action,
        bool is_concat = false) {
    const int64_t dlc = is_concat ? 2 * p.dlc : p.dlc;
    AOC<float> dst_iter(dst_iter_, p.n_layer, p.n_dir(), p.mb, p.dhc);
    AOC<float> dst_iter_c(dst_iter_c_, p.n_layer, p.n_dir(), p.mb, p.dhc);
    AOC<float> dst_layer(dst_layer_, p.n_iter, p.mb, dlc);
    AOC<const float> ws(ws_, p.n_layer + 2, p.n_dir(), p.n_iter + 2,
            p.n_states(), p.mb, p.wc);

    // Copy dst_layer
    for (int64_t it = 0; it < p.n_iter; it++) {
        for (int64_t nb = 0; nb < p.mb; nb++) {
            auto from = &ws(p.n_layer, dir_val, it + 1, H, nb, 0);
            auto to = &dst_layer(it, nb, action == action_concat ? p.dlc : 0);
            copy(1, p.dlc, p.wc, dlc, from, to, action, p.is_int8());

            if (p.is_int8() && p.cfg[DST_LAYER].dt != dnnl_u8) {
                float data_shift = p.data_shift;
                bool do_deq10n = true;

                if (p.direction == dnnl_bidirectional_sum) {
                    // In `bidir_sum` case, we need to dequantize data only
                    // after the final summation. Also, since we sum two shifted
                    // tensors, we need to enlarge the shift by 2x.
                    do_deq10n = action == action_sum;
                    data_shift *= 2;
                }

                if (do_deq10n)
                    data_deq10n(1, p.dlc, dlc, to, p.data_scale, data_shift);
            }
        }
    }

    int64_t it_source = (iter_dir == left2right) ? p.n_iter : 1;

    // Copy dst_iter (and dst_iter_c)
    for (int64_t lay = 0; lay < p.n_layer; lay++) {
        if (p.alg == VANILLA_LSTM) {
            copy(p.mb, p.dhc, p.wc, p.dhc,
                    &ws(lay + 1, dir_val, it_source, C, 0, 0),
                    &dst_iter_c(lay, dir_val, 0, 0));
        }

        copy(p.mb, p.dhc, p.wc, p.dhc,
                &ws(lay + 1, dir_val, it_source, H, 0, 0),
                &dst_iter(lay, dir_val, 0, 0));
        if (p.is_int8() && p.cfg[DST_ITER].dt != dnnl_u8)
            data_deq10n(p.mb, p.dhc, p.dhc, &dst_iter(lay, dir_val, 0, 0),
                    p.data_scale, p.data_shift);
    }
}

/******************************************************************************/
/*************************** Computation Routines *****************************/
/******************************************************************************/

void rnn_cell_fwd(const prb_t &p, float *dst_iter, float *dst_iter_c,
        float *gates, const float *weights_layer, const float *weights_iter,
        const float *weights_peephole, const float *bias,
        const float *src_layer, const float *src_iter, const float *src_iter_c,
        float *ws_local_) {
    switch (p.alg) {
        case VANILLA_GRU:
            gru_fwd(p, dst_iter, gates, weights_layer, weights_iter, bias,
                    src_layer, src_iter);
            break;
        case LBR_GRU:
            lbr_gru_fwd(p, dst_iter, gates, weights_layer, weights_iter, bias,
                    src_layer, src_iter, ws_local_);
            break;
        case VANILLA_LSTM:
            lstm_fwd(p, dst_iter, dst_iter_c, gates, weights_layer,
                    weights_iter, weights_peephole, bias, src_layer, src_iter,
                    src_iter_c);
            break;
        case VANILLA_RNN:
            rnn_fwd(p, dst_iter, gates, weights_layer, weights_iter, bias,
                    src_layer, src_iter);
            break;
        default: break;
    }
}

void rnn_linear_fwd(const prb_t &p, const float *src_iter_,
        const float *src_iter_c_, const float *src_layer_,
        const float *weights_layer_, const float *weights_iter_,
        const float *weights_peephole_, const float *bias_, float *dst_iter_,
        float *dst_iter_c_, float *dst_layer_, float *ws_, float *gates_) {

    assert(p.wc == MAX2(p.sic, MAX2(p.slc, p.dhc)));
    bool is_lbr = p.alg == LBR_GRU;
    bool is_concat = p.direction == dnnl_bidirectional_concat;

    float *bias_with_compensation = nullptr;
    if (p.is_int8()) {
        bias_with_compensation = new float[p.n_layer * p.n_dir()
                * (p.n_gates() + is_lbr) * p.dhc];
        prepare_bias(p, bias_with_compensation, bias_, weights_layer_,
                weights_iter_);
        bias_ = bias_with_compensation;
    }

    AOC<const float> weights_peephole(
            weights_peephole_, p.n_layer, p.n_dir(), 3 * p.dhc);
    AOC<const float> bias(
            bias_, p.n_layer, p.n_dir(), (p.n_gates() + is_lbr) * p.dhc);
    AOC<const float> weights_layer(
            weights_layer_, p.n_layer, p.n_dir(), p.n_gates() * p.dhc, p.slc);
    AOC<const float> weights_iter(
            weights_iter_, p.n_layer, p.n_dir(), p.n_gates() * p.dhc, p.sic);
    AOC<float> ws(ws_, p.n_layer + 2, p.n_dir(), p.n_iter + 2, p.n_states(),
            p.mb, p.wc);
    AOC<float> gates(
            gates_, p.n_layer, p.n_dir(), p.n_iter, p.mb, p.n_gates(), p.dhc);

    int64_t ws_local_size = is_lbr * p.mb * p.n_gates() * p.dhc;
    float *ws_local_ = new float[ws_local_size];

    auto process_direction = [&](rnn_iter_direction_t iter_dir,
                                     rnn_layer_direction_t lay_dir,
                                     int64_t dir_val, rnn_action_t action) {
        // we first need to copy the initial src_layer and src_iter{,_c} into
        // ws to simplify the logic of the code
        BENCHDNN_PRINT(80,
                "rnn_linear_fwd: call copy_init dir_val = " IFMT "\n", dir_val);
        copy_init_fwd(p, ws_, src_layer_, src_iter_, src_iter_c_, iter_dir,
                lay_dir, dir_val);

        // We run the grid of computation
        for (int64_t il = 0; il < p.n_layer; il++) {
            for (int64_t it = 0; it < p.n_iter; it++) {
                BENCHDNN_PRINT(80,
                        "==== layer = " IFMT " iter = " IFMT " ===\n", il, it);
                int64_t iter
                        = (iter_dir == left2right) ? it + 1 : p.n_iter - it;
                int64_t prev_iter
                        = (iter_dir == left2right) ? iter - 1 : iter + 1;
                int64_t lay = il + 1;
                rnn_cell_fwd(p, &ws(lay, dir_val, iter, H, 0, 0),
                        &ws(lay, dir_val, iter, C, 0, 0),
                        &gates(lay - 1, dir_val, iter - 1, 0, 0, 0),
                        &weights_layer(lay - 1, dir_val, 0, 0),
                        &weights_iter(lay - 1, dir_val, 0, 0),
                        &weights_peephole(lay - 1, dir_val, 0),
                        &bias(lay - 1, dir_val, 0),
                        &ws(lay - 1, dir_val, iter, H, 0, 0),
                        &ws(lay, dir_val, prev_iter, H, 0, 0),
                        &ws(lay, dir_val, prev_iter, C, 0, 0), ws_local_);
            }
        }

        // Finally we copy the results to the result buffers
        copy_res_fwd(p, dst_layer_, dst_iter_, dst_iter_c_, ws_, iter_dir,
                lay_dir, dir_val, action, is_concat);
    };

    switch (p.direction) {
        case dnnl_unidirectional_left2right:
            process_direction(left2right, bottom2top, 0, action_copy);
            break;
        case dnnl_unidirectional_right2left:
            process_direction(right2left, bottom2top, 0, action_copy);
            break;
        case dnnl_bidirectional_sum:
            process_direction(left2right, bottom2top, 0, action_copy);
            process_direction(right2left, bottom2top, 1, action_sum);
            break;
        case dnnl_bidirectional_concat:
            process_direction(left2right, bottom2top, 0, action_copy);
            process_direction(right2left, bottom2top, 1, action_concat);
            break;
        default: assert(!"unknown direction"); break;
    }

    delete[] ws_local_;
    if (bias_with_compensation) delete[] bias_with_compensation;
}

void compute_ref_fwd(const prb_t &p, dnn_mem_t &src_layer_m,
        dnn_mem_t &src_iter_m, dnn_mem_t &src_iter_c_m,
        dnn_mem_t &weights_src_layer_m, dnn_mem_t &weights_src_iter_m,
        dnn_mem_t &weights_peephole_m, dnn_mem_t &bias_m,
        dnn_mem_t &dst_layer_m, dnn_mem_t &dst_iter_m,
        dnn_mem_t &dst_iter_c_m) {

    assert(p.direction == dnnl_unidirectional_left2right
            || p.direction == dnnl_unidirectional_right2left
            || p.direction == dnnl_bidirectional_sum
            || p.direction == dnnl_bidirectional_concat);

    assert(p.wc == MAX2(p.sic, MAX2(p.slc, p.dhc)));
    int64_t ws_size = (p.n_layer + 2) * p.n_dir() * (p.n_iter + 2)
            * p.n_states() * p.mb * p.wc;
    auto *ws = new float[ws_size];
    int64_t gates_size
            = p.n_layer * p.n_dir() * p.n_iter * p.mb * p.n_gates() * p.dhc;
    auto *gates = new float[gates_size];

    rnn_linear_fwd(p, (float *)src_iter_m, (float *)src_iter_c_m,
            (float *)src_layer_m, (float *)weights_src_layer_m,
            (float *)weights_src_iter_m, (float *)weights_peephole_m,
            (float *)bias_m, (float *)dst_iter_m, (float *)dst_iter_c_m,
            (float *)dst_layer_m, ws, gates);

    delete[] ws;
    delete[] gates;
}

} // namespace rnn
