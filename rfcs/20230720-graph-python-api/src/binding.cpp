/*******************************************************************************
 * Copyright 2021-2024 Intel Corporation
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

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <sstream>

#include "oneapi/dnnl/dnnl_graph.hpp"

using cpartition = dnnl::graph::compiled_partition;
using engine = dnnl::graph::engine;
using graph = dnnl::graph::graph;
using logical_tensor = dnnl::graph::logical_tensor;
using op = dnnl::graph::op;
using partition = dnnl::graph::partition;
using stream = dnnl::graph::stream;
using tensor = dnnl::graph::tensor;

namespace dnnl {
namespace graph {
namespace binding {

void bind_cpartition(pybind11::module &m) {
    pybind11::class_<cpartition> cp(m, "compiled_partition");

    cp.def(pybind11::init<>());
    cp.def("query_logical_tensor", &cpartition::query_logical_tensor);
    cp.def("get_inplace_ports", &cpartition::get_inplace_ports);
    cp.def("execute", &cpartition::execute);
}

const std::string engine_kind2str(engine::kind v) {
    if (v == engine::kind::any) return "any";
    if (v == engine::kind::cpu) return "cpu";
    if (v == engine::kind::gpu) return "gpu";
    return "unknown engine_kind";
}

auto eng2string = [](const engine &eng) {
    std::stringstream ss;
    ss << "engine(kind = " << engine_kind2str(eng.get_kind()) << ")";
    return ss.str();
};

void bind_engine(pybind11::module &m) {
    pybind11::class_<engine> eng(m, "engine");

    eng.def(pybind11::init<engine::kind, size_t>());
    eng.def("get_kind", &engine::get_kind);
    eng.def("get_count", &engine::get_count);
    eng.def("__repr__", eng2string);

    pybind11::enum_<engine::kind>(eng, "kind")
            .value("any", engine::kind::any)
            .value("cpu", engine::kind::cpu)
            .value("gpu", engine::kind::gpu)
            .export_values();
}

void bind_graph(pybind11::module &m) {
    pybind11::class_<graph> g(m, "graph");

    g.def(pybind11::init<engine::kind>());
    g.def(pybind11::init<engine::kind, dnnl::graph::fpmath_mode>());
    g.def("add_op", &graph::add_op);
    g.def("finalize", &graph::finalize);
    g.def("is_finalized", &graph::is_finalized);
    g.def("get_partitions", &graph::get_partitions);

    pybind11::enum_<dnnl::graph::fpmath_mode>(g, "fpmath_mode")
            .value("strict", dnnl::graph::fpmath_mode::strict)
            .value("bf16", dnnl::graph::fpmath_mode::bf16)
            .value("f16", dnnl::graph::fpmath_mode::f16)
            .value("any", dnnl::graph::fpmath_mode::any)
            .value("tf32", dnnl::graph::fpmath_mode::tf32)
            .export_values();
}

const std::string data_type2str(logical_tensor::data_type v) {
#define CASE(x) \
    case (logical_tensor::data_type::x): return #x

    switch (v) {
        CASE(undef);
        CASE(f16);
        CASE(bf16);
        CASE(f32);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(boolean);
        default: return "unknown data type";
    }

#undef CASE
}

const std::string layout_type2str(logical_tensor::layout_type v) {
    if (v == logical_tensor::layout_type::undef) return "undef";
    if (v == logical_tensor::layout_type::any) return "any";
    if (v == logical_tensor::layout_type::strided) return "strided";
    if (v == logical_tensor::layout_type::opaque) return "opaque";
    return "unknown layout_type";
}

const std::string dims2string(const std::vector<int64_t> &dims) {
    std::stringstream ss;
    ss << "(";
    const char *delimer = "";
    for (const auto &d : dims) {
        ss << delimer << d;
        delimer = ", ";
    }
    ss << ")";
    return ss.str();
};

auto lt2string = [](const logical_tensor &lt) {
    std::stringstream ss;
    ss << "logical_tensor(id = " << lt.get_id()
       << ", dtype = " << data_type2str(lt.get_data_type())
       << ", layout = " << layout_type2str(lt.get_layout_type())
       << ", shape = " << dims2string(lt.get_dims());
    if (lt.get_layout_type() == logical_tensor::layout_type::opaque) {
        ss << ", layout_id = " << lt.get_layout_id();
    } else if (lt.get_layout_type() == logical_tensor::layout_type::strided) {
      ss << ", stride = " << dims2string(lt.get_strides());
    } else {
    }
    ss << ")";
    return ss.str();
};

void bind_logical_tensor(pybind11::module &m) {
    pybind11::class_<logical_tensor> lt(m, "logical_tensor");

    lt.def(pybind11::init<size_t, logical_tensor::data_type,
                          logical_tensor::layout_type>());
    lt.def(pybind11::init<size_t, logical_tensor::data_type, int32_t,
                          logical_tensor::layout_type,
                          logical_tensor::property_type>());
    lt.def(pybind11::init<size_t, logical_tensor::data_type,
                          logical_tensor::dims, logical_tensor::layout_type,
                          logical_tensor::property_type>());
    lt.def(
        pybind11::init<size_t, logical_tensor::data_type, logical_tensor::dims,
                       logical_tensor::dims, logical_tensor::property_type>());
    lt.def(
        pybind11::init<size_t, logical_tensor::data_type, logical_tensor::dims,
                       size_t, logical_tensor::property_type>());
    lt.def("get_id", &logical_tensor::get_id);
    lt.def("get_data_type", &logical_tensor::get_data_type);
    lt.def("get_layout_type", &logical_tensor::get_layout_type);
    lt.def("get_property_type", &logical_tensor::get_property_type);
    lt.def("get_layout_id", &logical_tensor::get_layout_id);
    lt.def("get_mem_size", &logical_tensor::get_mem_size);
    lt.def("get_dims", &logical_tensor::get_dims);
    lt.def("get_strides", &logical_tensor::get_strides);
    lt.def("is_equal", &logical_tensor::is_equal);
    lt.def("__repr__", lt2string);

    pybind11::enum_<logical_tensor::data_type>(lt, "data_type")
            .value("dt_undef", logical_tensor::data_type::undef)
            .value("f16", logical_tensor::data_type::f16)
            .value("bf16", logical_tensor::data_type::bf16)
            .value("f32", logical_tensor::data_type::f32)
            .value("s32", logical_tensor::data_type::s32)
            .value("s8", logical_tensor::data_type::s8)
            .value("u8", logical_tensor::data_type::u8)
            .value("boolean", logical_tensor::data_type::boolean)
            .export_values();

    pybind11::enum_<logical_tensor::layout_type>(lt, "layout_type")
            .value("lt_undef", logical_tensor::layout_type::undef)
            .value("any", logical_tensor::layout_type::any)
            .value("strided", logical_tensor::layout_type::strided)
            .value("opaque", logical_tensor::layout_type::opaque)
            .export_values();

    pybind11::enum_<logical_tensor::property_type>(lt, "property_type")
            .value("pt_undef", logical_tensor::property_type::undef)
            .value("variable", logical_tensor::property_type::variable)
            .value("constant", logical_tensor::property_type::constant)
            .export_values();
}

template <class T>
void set_op_attribute(op &aop, T x, op::attr attr) {
    if (pybind11::isinstance<pybind11::list>(x)) {
        if (pybind11::isinstance<pybind11::int_>(
                    x.template cast<pybind11::list>()[0])) {
            std::vector<int64_t> int_attr = {};
            for (auto val : x.template cast<pybind11::list>()) {
                int_attr.push_back(val.template cast<int64_t>());
            }
            aop.set_attr<std::vector<int64_t>>(attr, int_attr);
        } else if (pybind11::isinstance<pybind11::float_>(
                           x.template cast<pybind11::list>()[0])) {
            std::vector<float> int_attr = {};
            for (auto val : x.template cast<pybind11::list>()) {
                int_attr.push_back(val.template cast<float>());
            }
            aop.set_attr<std::vector<float>>(attr, int_attr);
        } else {
            assert(!"unknown vector type");
        }
    } else if (pybind11::isinstance<pybind11::bool_>(x)) {
        aop.set_attr<bool>(attr, x.template cast<bool>());
    } else if (pybind11::isinstance<pybind11::int_>(x)) {
        aop.set_attr<int64_t>(attr, x.template cast<int64_t>());
    } else if (pybind11::isinstance<pybind11::float_>(x)) {
        aop.set_attr<float>(attr, x.template cast<float>());
    } else if (pybind11::isinstance<pybind11::str>(x)) {
        aop.set_attr<std::string>(attr, x.template cast<std::string>());
    } else {
        assert(!"unknown attribute type");
    }
}

void bind_op(pybind11::module &m) {
    pybind11::class_<op> opr(m, "op");

    opr.def(pybind11::init<size_t, op::kind, std::string>());
    opr.def(pybind11::init(
        [](size_t id, op::kind kind, const std::vector<logical_tensor> &inputs,
           const std::vector<logical_tensor> &outputs, std::string name) {
          auto aop = op(id, kind, inputs, outputs, name);
          return aop;
        }));
    opr.def("set_attr", [](op &aop, op::attr key, pybind11::object val) {
        set_op_attribute(aop, val, key);
    });
    opr.def("add_input", &op::add_input);
    opr.def("add_inputs", &op::add_inputs);
    opr.def("add_output", &op::add_output);
    opr.def("add_outputs", &op::add_outputs);

    pybind11::enum_<op::kind>(opr, "kind")
            .value("Abs", op::kind::Abs)
            .value("AbsBackward", op::kind::AbsBackward)
            .value("Add", op::kind::Add)
            .value("AvgPool", op::kind::AvgPool)
            .value("AvgPoolBackward", op::kind::AvgPoolBackward)
            .value("BatchNormForwardTraining",
                    op::kind::BatchNormForwardTraining)
            .value("BatchNormInference", op::kind::BatchNormInference)
            .value("BatchNormTrainingBackward",
                    op::kind::BatchNormTrainingBackward)
            .value("BiasAdd", op::kind::BiasAdd)
            .value("BiasAddBackward", op::kind::BiasAddBackward)
            .value("Clamp", op::kind::Clamp)
            .value("ClampBackward", op::kind::ClampBackward)
            .value("Concat", op::kind::Concat)
            .value("Convolution", op::kind::Convolution)
            .value("ConvolutionBackwardData", op::kind::ConvolutionBackwardData)
            .value("ConvolutionBackwardWeights",
                    op::kind::ConvolutionBackwardWeights)
            .value("ConvTranspose", op::kind::ConvTranspose)
            .value("ConvTransposeBackwardData",
                    op::kind::ConvTransposeBackwardData)
            .value("ConvTransposeBackwardWeights",
                    op::kind::ConvTransposeBackwardWeights)
            .value("Dequantize", op::kind::Dequantize)
            .value("Divide", op::kind::Divide)
            .value("DynamicDequantize", op::kind::DynamicDequantize)
            .value("DynamicQuantize", op::kind::DynamicQuantize)
            .value("Elu", op::kind::Elu)
            .value("EluBackward", op::kind::EluBackward)
            .value("End", op::kind::End)
            .value("Exp", op::kind::Exp)
            .value("GELU", op::kind::GELU)
            .value("GELUBackward", op::kind::GELUBackward)
            .value("HardSigmoid", op::kind::HardSigmoid)
            .value("HardSigmoidBackward", op::kind::HardSigmoidBackward)
            .value("HardSwish", op::kind::HardSwish)
            .value("HardSwishBackward", op::kind::HardSwishBackward)
            .value("Interpolate", op::kind::Interpolate)
            .value("InterpolateBackward", op::kind::InterpolateBackward)
            .value("LayerNorm", op::kind::LayerNorm)
            .value("LayerNormBackward", op::kind::LayerNormBackward)
            .value("LeakyReLU", op::kind::LeakyReLU)
            .value("Log", op::kind::Log)
            .value("LogSoftmax", op::kind::LogSoftmax)
            .value("LogSoftmaxBackward", op::kind::LogSoftmaxBackward)
            .value("MatMul", op::kind::MatMul)
            .value("Maximum", op::kind::Maximum)
            .value("MaxPool", op::kind::MaxPool)
            .value("MaxPoolBackward", op::kind::MaxPoolBackward)
            .value("Minimum", op::kind::Minimum)
            .value("Mish", op::kind::Mish)
            .value("MishBackward", op::kind::MishBackward)
            .value("Multiply", op::kind::Multiply)
            .value("Pow", op::kind::Pow)
            .value("PReLU", op::kind::PReLU)
            .value("PReLUBackward", op::kind::PReLUBackward)
            .value("Quantize", op::kind::Quantize)
            .value("Reciprocal", op::kind::Reciprocal)
            .value("ReduceL1", op::kind::ReduceL1)
            .value("ReduceL2", op::kind::ReduceL2)
            .value("ReduceMax", op::kind::ReduceMax)
            .value("ReduceMean", op::kind::ReduceMean)
            .value("ReduceMin", op::kind::ReduceMin)
            .value("ReduceProd", op::kind::ReduceProd)
            .value("ReduceSum", op::kind::ReduceSum)
            .value("ReLU", op::kind::ReLU)
            .value("ReLUBackward", op::kind::ReLUBackward)
            .value("Reorder", op::kind::Reorder)
            .value("Round", op::kind::Round)
            .value("Select", op::kind::Select)
            .value("Sigmoid", op::kind::Sigmoid)
            .value("SigmoidBackward", op::kind::SigmoidBackward)
            .value("SoftMax", op::kind::SoftMax)
            .value("SoftMaxBackward", op::kind::SoftMaxBackward)
            .value("SoftPlus", op::kind::SoftPlus)
            .value("SoftPlusBackward", op::kind::SoftPlusBackward)
            .value("Sqrt", op::kind::Sqrt)
            .value("SqrtBackward", op::kind::SqrtBackward)
            .value("Square", op::kind::Square)
            .value("SquaredDifference", op::kind::SquaredDifference)
            .value("StaticReshape", op::kind::StaticReshape)
            .value("StaticTranspose", op::kind::StaticTranspose)
            .value("Subtract", op::kind::Subtract)
            .value("Tanh", op::kind::Tanh)
            .value("TanhBackward", op::kind::TanhBackward)
            .value("TypeCast", op::kind::TypeCast)
            .value("Wildcard", op::kind::Wildcard)
            .export_values();

    pybind11::enum_<op::attr>(opr, "attr")
            .value("undef", op::attr::undef)
            .value("alpha", op::attr::alpha)
            .value("beta", op::attr::beta)
            .value("epsilon", op::attr::epsilon)
            .value("max", op::attr::max)
            .value("min", op::attr::min)
            .value("momentum", op::attr::momentum)
            .value("scales", op::attr::scales)
            .value("axis", op::attr::axis)
            .value("begin_norm_axis", op::attr::begin_norm_axis)
            .value("groups", op::attr::groups)
            .value("axes", op::attr::axes)
            .value("dilations", op::attr::dilations)
            .value("dst_shape", op::attr::dst_shape)
            .value("kernel", op::attr::kernel)
            .value("order", op::attr::order)
            .value("output_padding", op::attr::output_padding)
            .value("pads_begin", op::attr::pads_begin)
            .value("pads_end", op::attr::pads_end)
            .value("shape", op::attr::shape)
            .value("sizes", op::attr::sizes)
            .value("src_shape", op::attr::src_shape)
            .value("strides", op::attr::strides)
            .value("weights_shape", op::attr::weights_shape)
            .value("zps", op::attr::zps)
            .value("exclude_pad", op::attr::exclude_pad)
            .value("keep_dims", op::attr::keep_dims)
            .value("keep_stats", op::attr::keep_stats)
            .value("per_channel_broadcast", op::attr::per_channel_broadcast)
            .value("special_zero", op::attr::special_zero)
            .value("transpose_a", op::attr::transpose_a)
            .value("transpose_b", op::attr::transpose_b)
            .value("use_affine", op::attr::use_affine)
            .value("use_dst", op::attr::use_dst)
            .value("auto_broadcast", op::attr::auto_broadcast)
            .value("auto_pad", op::attr::auto_pad)
            .value("coordinate_transformation_mode",
                    op::attr::coordinate_transformation_mode)
            .value("data_format", op::attr::data_format)
            .value("mode", op::attr::mode)
            .value("qtype", op::attr::qtype)
            .value("rounding_type", op::attr::rounding_type)
            .value("weights_format", op::attr::weights_format)
            .export_values();
}

void bind_partition(pybind11::module &m) {
    pybind11::class_<partition> p(m, "partition");

    p.def(pybind11::init<>());
    p.def(pybind11::init(
        [](const op &op, engine::kind ekind) { return partition(op, ekind); }));
    p.def("get_ops_num", &partition::get_ops_num);
    p.def("get_ops", &partition::get_ops);
    p.def("get_id", &partition::get_id);
    p.def("is_supported", &partition::is_supported);
    p.def("get_input_ports", &partition::get_input_ports);
    p.def("get_output_ports", &partition::get_output_ports);
    p.def("get_engine_kind", &partition::get_engine_kind);
    p.def("compile", &partition::compile);

    pybind11::enum_<partition::policy>(p, "policy")
            .value("fusion", partition::policy::fusion)
            .value("debug", partition::policy::debug)
            .export_values();
}

void bind_stream(pybind11::module &m) {
    pybind11::class_<stream> strm(m, "stream");

    strm.def(pybind11::init<engine &>());
    strm.def("get_engine", &stream::get_engine);
    strm.def("wait", &stream::wait);
}

static size_t size_of(logical_tensor::data_type dtype) {
    switch (dtype) {
        case logical_tensor::data_type::f32:
        case logical_tensor::data_type::s32: return 4U;
        case logical_tensor::data_type::s8:
        case logical_tensor::data_type::u8: return 1U;
        case logical_tensor::data_type::f16:
        case logical_tensor::data_type::bf16: return 2U;
        case logical_tensor::data_type::boolean: return sizeof(bool);
        default: return 0;
    }
}

static std::string format_string(logical_tensor::data_type dtype) {
    switch (dtype) {
        case logical_tensor::data_type::f32:
        case logical_tensor::data_type::f16:
        case logical_tensor::data_type::bf16:
            return pybind11::format_descriptor<float>::format();
            break;
        case logical_tensor::data_type::u8:
            return pybind11::format_descriptor<uint8_t>::format();
            break;
        case logical_tensor::data_type::s8:
            return pybind11::format_descriptor<int8_t>::format();
            break;
        case logical_tensor::data_type::boolean:
            return pybind11::format_descriptor<bool>::format();
            break;
        case logical_tensor::data_type::s32:
            return pybind11::format_descriptor<int32_t>::format();
            break;
        default: throw std::runtime_error("unknown data type");
    }
}

pybind11::buffer_info to_buffer_info(tensor &t, logical_tensor &lt) {
    auto strides = lt.get_strides();
    auto shapes = lt.get_dims();
    auto dtype = lt.get_data_type();
    std::transform(strides.begin(), strides.end(), strides.begin(),
            [&](int64_t i) { return i * size_of(dtype); });
    return pybind11::buffer_info(t.get_data_handle(), /* Pointer to buffer */
            size_of(dtype), /* Size of one scalar */
            format_string(dtype), /* Python struct-style format descriptor */
            shapes.size(), /* Number of dimensions */
            shapes, /* Buffer dimensions */
            strides);
}

logical_tensor::data_type convert_from_array_dtype(const pybind11::array &a) {
  auto tgt_dtype = a.dtype();
  if (tgt_dtype.is(pybind11::dtype::of<float>())) {
    return logical_tensor::data_type::f32;
  } else if (tgt_dtype.is(pybind11::dtype::of<int8_t>())) {
    return logical_tensor::data_type::s8;
  } else if (tgt_dtype.is(pybind11::dtype::of<uint8_t>())) {
    return logical_tensor::data_type::u8;
  } else if (tgt_dtype.is(pybind11::dtype::of<int32_t>())) {
    return logical_tensor::data_type::s32;
  } else if (tgt_dtype.is(pybind11::dtype::of<bool>())) {
    return logical_tensor::data_type::boolean;
  } else {
    // not support fp16 and bf16 yet
    assert(!"unsupported data type.");
  }
  return logical_tensor::data_type::undef;
}

void bind_tensor(pybind11::module &m) {
    pybind11::class_<tensor> t(m, "tensor", pybind11::buffer_protocol());

    t.def(pybind11::init(
            [](logical_tensor &lt, engine &eng, pybind11::buffer b) {
                auto bufinfo = b.request();
                return tensor(lt, eng, bufinfo.ptr);
            }));
    t.def(pybind11::init([](logical_tensor &lt, engine &eng) {
        return tensor(lt, eng, nullptr);
    }));
    t.def(pybind11::init([](logical_tensor &lt, engine &eng, int64_t data_ptr) {
      return tensor(lt, eng, reinterpret_cast<void *>(data_ptr));
    }));
    t.def(
        "set_data_handle",
        [](tensor &self, int64_t data_ptr) {
          self.set_data_handle(reinterpret_cast<void *>(data_ptr));
        },
        pybind11::arg("data_ptr"));
    t.def("get_data_handle", [](tensor &self) {
      return reinterpret_cast<int64_t>(self.get_data_handle());
    });
    t.def("get_engine", &tensor::get_engine);
    t.def("from_numpy", [](pybind11::array &b, const engine &eng) {
      // create a logical tensor with id `0`
      logical_tensor::dims shape{b.shape(), b.shape() + b.ndim()};
      logical_tensor::dims strides{b.strides(), b.strides() + b.ndim()};
      logical_tensor lt{0, convert_from_array_dtype(b), shape, strides};
      // get mutable pointer from array
      return tensor{lt, eng, b.mutable_data()};
    });
    t.def(
        "to_numpy",
        [](tensor &self, logical_tensor &lt) {
          auto bufinfo = to_buffer_info(self, lt);
          // pass any valid pybind object to `base` to make sure there is
          // no memory copy between returned array and self tensor, see
          // details at https://github.com/pybind/pybind11/issues/323
          return pybind11::array(pybind11::dtype(bufinfo), bufinfo.shape,
                                 bufinfo.strides, bufinfo.ptr,
                                 /* base = */ pybind11::str{});
        },
        pybind11::arg("lt"));
}

void bind_status(pybind11::module &m) {
    pybind11::enum_<dnnl::graph::status>(m, "status")
            .value("success", dnnl::graph::status::success)
            .value("out_of_memory", dnnl::graph::status::out_of_memory)
            .value("invalid_arguments", dnnl::graph::status::invalid_arguments)
            .value("unimplemented", dnnl::graph::status::unimplemented)
            .value("last_impl_reached", dnnl::graph::status::last_impl_reached)
            .value("runtime_error", dnnl::graph::status::runtime_error)
            .value("not_required", dnnl::graph::status::not_required)
            .value("invalid_graph", dnnl::graph::status::invalid_graph)
            .value("invalid_graph_op", dnnl::graph::status::invalid_graph_op)
            .value("invalid_shape", dnnl::graph::status::invalid_shape)
            .value("invalid_data_type", dnnl::graph::status::invalid_data_type)
            .export_values();
}

void bind(pybind11::module &m) {
  m.doc() = R"pbdoc(
        oneDNN Graph API Python binding
        -------------------------------
        .. currentmodule:: dnnl_graph
        .. autosummary::
           :toctree: _generate
    )pbdoc";

  // the version macros are defined in dnnl_version.h.
  std::ostringstream oss;
  oss << "v" << DNNL_VERSION_MAJOR << "." << DNNL_VERSION_MINOR << "."
      << DNNL_VERSION_PATCH << "+" << DNNL_VERSION_HASH;
  m.attr("__version__") = oss.str();

  bind_status(m);
  bind_graph(m);
  bind_logical_tensor(m);
  bind_engine(m);
  bind_op(m);
  bind_tensor(m);
  bind_partition(m);
  bind_cpartition(m);
  bind_stream(m);
}

PYBIND11_MODULE(dnnl_graph, m) { bind(m); }

} // namespace binding
} // namespace graph
} // namespace dnnl