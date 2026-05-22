/*******************************************************************************
* Copyright 2025 Intel Corporation
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

/// @example sycl_matmul_dropout.cpp
/// > Annotated version: @ref sycl_matmul_dropout_cpp

/// @page sycl_matmul_dropout_cpp_brief
/// @brief This C++ API example demonstrates a matrix multiplication with
/// dropout attribute on a SYCL GPU device, using device memory for seed
/// and offset.

/// @page sycl_matmul_dropout_cpp Matrix Multiplication with Dropout (SYCL GPU)
/// \copybrief sycl_matmul_dropout_cpp_brief
///
/// The workflow includes the following steps:
///   - Create a GPU engine using SYCL runtime.
///   - Allocate source, weights, and destination matrices using SYCL USM.
///   - Configure a matmul primitive with the dropout attribute, passing
///     seed and offset via device memory buffers.
///   - Execute the matmul primitive with dropout on the GPU.
///   - Read back and validate the results.
///
/// @include sycl_matmul_dropout.cpp

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace dnnl;

void sycl_matmul_dropout_example(engine::kind engine_kind) {
    // Matrix dimensions: C[M x N] = A[M x K] * B[K x N]
    const memory::dim M = 64;
    const memory::dim K = 128;
    const memory::dim N = 256;

    // Create engine and stream
    engine eng(engine_kind, 0);
    dnnl::stream strm(eng);

    // Create memory descriptors for matmul
    memory::desc src_md({M, K}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc wei_md({K, N}, memory::data_type::f32, memory::format_tag::ab);
    memory::desc dst_md({M, N}, memory::data_type::f32, memory::format_tag::ab);

    // Allocate USM memory on the device
    auto sycl_dev = sycl_interop::get_device(eng);
    auto sycl_ctx = sycl_interop::get_context(eng);

    const size_t src_size = M * K;
    const size_t wei_size = K * N;
    const size_t dst_size = M * N;

    float *src_data = (float *)sycl::malloc_shared(
            src_size * sizeof(float), sycl_dev, sycl_ctx);
    float *wei_data = (float *)sycl::malloc_shared(
            wei_size * sizeof(float), sycl_dev, sycl_ctx);
    float *dst_data = (float *)sycl::malloc_shared(
            dst_size * sizeof(float), sycl_dev, sycl_ctx);

    // Initialize source and weights data
    for (size_t i = 0; i < src_size; i++)
        src_data[i] = static_cast<float>((i % 7) - 3) * 0.1f;
    for (size_t i = 0; i < wei_size; i++)
        wei_data[i] = static_cast<float>((i % 5) - 2) * 0.1f;
    for (size_t i = 0; i < dst_size; i++)
        dst_data[i] = 0.0f;

    // Create oneDNN memory objects wrapping USM pointers
    memory src_mem = sycl_interop::make_memory(
            src_md, eng, sycl_interop::memory_kind::usm, src_data);
    memory wei_mem = sycl_interop::make_memory(
            wei_md, eng, sycl_interop::memory_kind::usm, wei_data);
    memory dst_mem = sycl_interop::make_memory(
            dst_md, eng, sycl_interop::memory_kind::usm, dst_data);

    // Configure dropout attribute with device memory for seed and offset.
    // set_dropout_v2 parameters:
    //   - mask_desc: memory descriptor for dropout mask output (empty = no mask output)
    //   - seed_dt: data type of the seed (s64)
    //   - use_offset: true - an offset buffer will be passed at execution
    //   - use_host_scalars: false - seed/offset/probability are device memory
    primitive_attr matmul_attr;
    memory::desc mask_desc; // default (empty) means no mask output
    matmul_attr.set_dropout(mask_desc,
            memory::data_type::s64, // seed data type
            true, // use_offset = true
            false); // use_host_scalars = false (device memory)

    // Create matmul primitive descriptor and primitive
    auto matmul_pd
            = matmul::primitive_desc(eng, src_md, wei_md, dst_md, matmul_attr);
    auto matmul_prim = matmul(matmul_pd);

    // Allocate device memory for dropout probability, seed, and offset
    memory::desc prob_md({1}, memory::data_type::f32, memory::format_tag::a);
    memory::desc seed_md({1}, memory::data_type::s64, memory::format_tag::a);
    memory::desc offset_md({1}, memory::data_type::s64, memory::format_tag::a);

    memory prob_mem(prob_md, eng);
    memory seed_mem(seed_md, eng);
    memory offset_mem(offset_md, eng);

    // Write dropout parameters to device memory
    float probability = 0.5f; // 50% dropout
    int64_t seed = 12345678;
    int64_t offset = 100;

    write_to_dnnl_memory(&probability, prob_mem);
    write_to_dnnl_memory(&seed, seed_mem);
    write_to_dnnl_memory(&offset, offset_mem);

    // Execute matmul with dropout
    matmul_prim.execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                    {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_DROPOUT_PROBABILITY, prob_mem},
                    {DNNL_ARG_ATTR_DROPOUT_SEED, seed_mem},
                    {DNNL_ARG_ATTR_DROPOUT_OFFSET, offset_mem}});
    strm.wait();

    // Validate: check that dropout was applied (some outputs should be zero)
    int zero_count = 0;
    int nonzero_count = 0;
    for (size_t i = 0; i < dst_size; i++) {
        if (dst_data[i] == 0.0f)
            zero_count++;
        else
            nonzero_count++;
    }

    std::cout << "Matmul with dropout completed successfully." << std::endl;
    std::cout << "  Output size: " << dst_size << std::endl;
    std::cout << "  Zero elements (dropped): " << zero_count << std::endl;
    std::cout << "  Non-zero elements: " << nonzero_count << std::endl;

    // With 50% dropout, we expect roughly half the outputs to be zero.
    // Allow a wide tolerance since it's random.
    float drop_ratio
            = static_cast<float>(zero_count) / static_cast<float>(dst_size);
    std::cout << "  Observed drop ratio: " << drop_ratio << " (expected ~0.5)"
              << std::endl;

    if (drop_ratio < 0.1f || drop_ratio > 0.9f) {
        throw std::string("Dropout ratio is outside expected range [0.1, 0.9]");
    }

    // Free USM memory
    sycl::free(src_data, sycl_ctx);
    sycl::free(wei_data, sycl_ctx);
    sycl::free(dst_data, sycl_ctx);
}

int main(int argc, char **argv) {
    int exit_code = 0;

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        sycl_matmul_dropout_example(engine_kind);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::string &e) {
        std::cout << "Error in the example: " << e << "." << std::endl;
        exit_code = 2;
    } catch (sycl::exception &e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        exit_code = 3;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind2str_upper(engine_kind) << "." << std::endl;
    finalize();
    return exit_code;
}
