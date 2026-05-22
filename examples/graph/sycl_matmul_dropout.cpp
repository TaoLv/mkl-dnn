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

/// @example sycl_matmul_dropout_graph.cpp
/// > Annotated version: @ref graph_sycl_matmul_dropout_cpp

/// @page graph_sycl_matmul_dropout_cpp_brief
/// @brief This C++ API example demonstrates a matmul + dropout pattern using
/// the oneDNN Graph API on a SYCL GPU device, with device memory for seed and
/// offset.

/// @page graph_sycl_matmul_dropout_cpp MatMul + Dropout Graph (SYCL GPU)
/// \copybrief graph_sycl_matmul_dropout_cpp_brief
///
/// The workflow includes the following steps:
///   - Build a graph containing MatMul followed by Dropout ops.
///   - Get partitions from the graph (expecting them to be fused).
///   - Compile partitions and allocate SYCL USM device memory.
///   - Pass dropout seed and offset via device memory tensors.
///   - Execute the compiled partition on the GPU.
///
/// @include sycl_matmul_dropout_graph.cpp

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "example_utils.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using property_type = logical_tensor::property_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

void sycl_matmul_dropout_graph_tutorial() {
    // Matrix dimensions: C[M x N] = A[M x K] * B[K x N]
    const dim M = 64, K = 128, N = 256;

    dims src_dims {M, K};
    dims wei_dims {K, N};
    dims dst_dims {M, N};

    /// Build the graph: MatMul -> Dropout
    ///
    /// MatMul inputs:  src(id=0), weights(id=1)
    /// MatMul output:  matmul_dst(id=2)
    /// Dropout inputs: matmul_dst(id=2), seed(id=3), offset(id=4), prob(id=5)
    /// Dropout output: dropout_dst(id=6)

    // MatMul logical tensors
    logical_tensor matmul_src {
            0, data_type::f32, src_dims, layout_type::strided};
    logical_tensor matmul_wei {
            1, data_type::f32, wei_dims, layout_type::strided};
    logical_tensor matmul_dst {
            2, data_type::f32, dst_dims, layout_type::strided};

    // Dropout logical tensors
    // seed, offset, and probability: 1-element tensors on device memory
    logical_tensor dropout_seed {3, data_type::s64, dims {1},
            layout_type::strided, property_type::undef};
    logical_tensor dropout_offset {4, data_type::s64, dims {1},
            layout_type::strided, property_type::undef};
    logical_tensor dropout_prob {5, data_type::f32, dims {1},
            layout_type::strided, property_type::undef};
    logical_tensor dropout_dst {
            6, data_type::f32, dst_dims, layout_type::strided};

    // Create MatMul op
    op matmul_op(0, op::kind::MatMul, {matmul_src, matmul_wei}, {matmul_dst},
            "matmul");
    matmul_op.set_attr<bool>(op::attr::transpose_a, false);
    matmul_op.set_attr<bool>(op::attr::transpose_b, false);

    // Create Dropout op
    op dropout_op(1, op::kind::Dropout,
            {matmul_dst, dropout_seed, dropout_offset, dropout_prob},
            {dropout_dst}, "dropout");

    // Build the graph
    dnnl::graph::graph g(dnnl::engine::kind::gpu);
    g.add_op(matmul_op);
    g.add_op(dropout_op);
    g.finalize();

    // Get partitions - expect a single fused partition
    auto partitions = g.get_partitions();
    std::cout << "Number of partitions: " << partitions.size() << std::endl;
    if (partitions.size() != 1) {
        std::cout << "Expected a single fused partition" << std::endl;
        return;
    }
    auto &part = partitions[0];
    if (!part.is_supported()) {
        std::cout << "Partition is not supported" << std::endl;
        return;
    }

    // Set up SYCL engine and stream
    allocator alloc = dnnl::graph::sycl_interop::make_allocator(
            sycl_malloc_wrapper, sycl_free_wrapper);

    sycl::queue q = sycl::queue(
            sycl::gpu_selector_v, sycl::property::queue::in_order {});

    dnnl::engine eng = dnnl::graph::sycl_interop::make_engine_with_allocator(
            q.get_device(), q.get_context(), alloc);

    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);

    // Get input/output ports and compile the partition
    auto inputs = part.get_input_ports();
    auto outputs = part.get_output_ports();

    for (auto &output : outputs) {
        const auto id = output.get_id();
        output = logical_tensor {id, output.get_data_type(),
                DNNL_GRAPH_UNKNOWN_NDIMS, layout_type::strided};
    }

    compiled_partition cp = part.compile(inputs, outputs, eng);

    for (auto &output : outputs) {
        output = cp.query_logical_tensor(output.get_id());
    }

    // Allocate input tensors
    std::vector<std::shared_ptr<void>> data_buffers;

    // src: device memory
    std::vector<float> src_host(M * K);
    for (size_t i = 0; i < src_host.size(); i++)
        src_host[i] = static_cast<float>((i % 7) - 3) * 0.1f;
    auto src_size = matmul_src.get_mem_size();
    data_buffers.emplace_back(
            sycl::malloc_device(src_size, q.get_device(), q.get_context()),
            sycl_deletor_t {q.get_context()});
    q.memcpy(data_buffers.back().get(), src_host.data(), src_size).wait();
    tensor src_ts(matmul_src, eng, data_buffers.back().get());

    // weights: device memory
    std::vector<float> wei_host(K * N);
    for (size_t i = 0; i < wei_host.size(); i++)
        wei_host[i] = static_cast<float>((i % 5) - 2) * 0.1f;
    auto wei_size = matmul_wei.get_mem_size();
    data_buffers.emplace_back(
            sycl::malloc_device(wei_size, q.get_device(), q.get_context()),
            sycl_deletor_t {q.get_context()});
    q.memcpy(data_buffers.back().get(), wei_host.data(), wei_size).wait();
    tensor wei_ts(matmul_wei, eng, data_buffers.back().get());

    // seed: device memory scalar (s64)
    int64_t seed_val = 12345678;
    data_buffers.emplace_back(sycl::malloc_device(sizeof(int64_t),
                                      q.get_device(), q.get_context()),
            sycl_deletor_t {q.get_context()});
    q.memcpy(data_buffers.back().get(), &seed_val, sizeof(int64_t)).wait();
    tensor seed_ts(dropout_seed, eng, data_buffers.back().get());

    // offset: device memory scalar (s64)
    int64_t offset_val = 100;
    data_buffers.emplace_back(sycl::malloc_device(sizeof(int64_t),
                                      q.get_device(), q.get_context()),
            sycl_deletor_t {q.get_context()});
    q.memcpy(data_buffers.back().get(), &offset_val, sizeof(int64_t)).wait();
    tensor offset_ts(dropout_offset, eng, data_buffers.back().get());

    // probability: device memory (f32)
    float prob_val = 0.5f;
    data_buffers.emplace_back(
            sycl::malloc_device(sizeof(float), q.get_device(), q.get_context()),
            sycl_deletor_t {q.get_context()});
    q.memcpy(data_buffers.back().get(), &prob_val, sizeof(float)).wait();
    tensor prob_ts(dropout_prob, eng, data_buffers.back().get());

    // Allocate output tensor
    auto out_lt = outputs[0];
    auto out_size = out_lt.get_mem_size();
    data_buffers.emplace_back(
            sycl::malloc_device(out_size, q.get_device(), q.get_context()),
            sycl_deletor_t {q.get_context()});
    tensor out_ts(out_lt, eng, data_buffers.back().get());

    // Build input tensor list in the order expected by the compiled partition
    std::unordered_map<size_t, tensor> id_to_tensor;
    id_to_tensor[0] = src_ts;
    id_to_tensor[1] = wei_ts;
    id_to_tensor[3] = seed_ts;
    id_to_tensor[4] = offset_ts;
    id_to_tensor[5] = prob_ts;

    std::vector<tensor> inputs_ts;
    for (const auto &in : inputs) {
        inputs_ts.push_back(id_to_tensor.at(in.get_id()));
    }

    // Debug: print input/output tensor addresses and values
    std::cout << "=== Graph example: input tensors ===" << std::endl;
    for (size_t i = 0; i < inputs_ts.size(); i++) {
        auto id = inputs[i].get_id();
        auto prop = inputs[i].get_property_type();
        std::cout << "  input[" << i << "] id=" << id
                  << " property=" << static_cast<int>(prop);
        if (prop == property_type::host_scalar) {
            std::cout << " (host_scalar)";
            if (id == 5) { // probability - we know it's float
                std::cout << " value=" << prob_val;
            }
        } else {
            std::cout << " device_ptr=" << inputs_ts[i].get_data_handle();
        }
        std::cout << std::endl;
    }
    std::cout << "=== Graph example: output tensor ===" << std::endl;
    std::cout << "  output id=" << out_lt.get_id()
              << " device_ptr=" << out_ts.get_data_handle() << std::endl;

    // Execute the compiled partition
    cp.execute(strm, inputs_ts, {out_ts});
    strm.wait();

    // Read back the output and validate
    size_t output_size = M * N;
    std::vector<float> host_output(output_size);
    q.memcpy(host_output.data(), out_ts.get_data_handle(),
             output_size * sizeof(float))
            .wait();

    int zero_count = 0;
    int nonzero_count = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (host_output[i] == 0.0f)
            zero_count++;
        else
            nonzero_count++;
    }

    std::cout << "Graph matmul+dropout completed successfully." << std::endl;
    std::cout << "  Output size: " << output_size << std::endl;
    std::cout << "  Zero elements (dropped): " << zero_count << std::endl;
    std::cout << "  Non-zero elements: " << nonzero_count << std::endl;

    float drop_ratio
            = static_cast<float>(zero_count) / static_cast<float>(output_size);
    std::cout << "  Observed drop ratio: " << drop_ratio << " (expected ~0.5)"
              << std::endl;

    std::cout << "Graph:" << std::endl
              << " [matmul_src] [matmul_wei]  [seed] [offset] [prob]"
              << std::endl
              << "       \\       /               \\      |      /" << std::endl
              << "        matmul                   dropout" << std::endl
              << "            \\                    /" << std::endl
              << "             -----> [dst] ------" << std::endl
              << "                       |" << std::endl
              << "                 [dropout_dst]" << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            {dnnl::engine::kind::gpu}, sycl_matmul_dropout_graph_tutorial);
}
