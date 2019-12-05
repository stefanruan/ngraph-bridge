# ==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
licenses(["notice"])  
exports_files(["LICENSE"])

load("@ngraph_bridge//:cxx_abi_option.bzl", "CXX_ABI")

cc_library(
    name = "ngraph_headers",
    hdrs = glob([
        "src/ngraph/**/*.hpp",
        "src/ngraph/*.hpp"
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ngraph_core",
    srcs = glob([
        "src/ngraph/*.cpp",
        "src/ngraph/autodiff/*.cpp",
        "src/ngraph/builder/*.cpp",
        "src/ngraph/descriptor/*.cpp",
        "src/ngraph/descriptor/layout/*.cpp",
        "src/ngraph/op/*.cpp",
        "src/ngraph/op/fused/*.cpp",
        "src/ngraph/op/experimental/batch_mat_mul.cpp",
        "src/ngraph/op/experimental/compiled_kernel.cpp",
        "src/ngraph/op/experimental/dyn_broadcast.cpp",
        "src/ngraph/op/experimental/dyn_replace_slice.cpp",
        "src/ngraph/op/experimental/dyn_pad.cpp",
        "src/ngraph/op/experimental/dyn_reshape.cpp",
        "src/ngraph/op/experimental/dyn_slice.cpp",
        "src/ngraph/op/experimental/generate_mask.cpp",
        "src/ngraph/op/experimental/quantized_avg_pool.cpp",
        "src/ngraph/op/experimental/quantized_concat.cpp",
        "src/ngraph/op/experimental/quantized_conv.cpp",
        "src/ngraph/op/experimental/quantized_conv_bias.cpp",
        "src/ngraph/op/experimental/quantized_conv_relu.cpp",
        "src/ngraph/op/experimental/quantized_max_pool.cpp",
        "src/ngraph/op/experimental/shape_of.cpp",
        "src/ngraph/op/experimental/range.cpp",
        "src/ngraph/op/experimental/quantized_dot.cpp",
        "src/ngraph/op/experimental/quantized_dot_bias.cpp",
        "src/ngraph/op/experimental/tile.cpp",
        "src/ngraph/op/experimental/transpose.cpp",
        "src/ngraph/op/experimental/layers/interpolate.cpp",
        "src/ngraph/op/util/*.cpp",
        "src/ngraph/pattern/*.cpp",
        "src/ngraph/pattern/*.hpp",
        "src/ngraph/pass/*.cpp",
        "src/ngraph/pass/*.hpp",
        "src/ngraph/runtime/*.cpp",
        "src/ngraph/runtime/dynamic/dynamic_backend.cpp",
        "src/ngraph/type/*.cpp",
        ],
        exclude = [
        "src/ngraph/ngraph.cpp",
        "src/ngraph/serializer_stub.cpp"
    ]),
    deps = [
        ":ngraph_headers",
        "@nlohmann_json_lib",
    ],
    copts = [
        "-I external/ngraph/src",
        "-I external/ngraph/src/ngraph",
        "-I external/nlohmann_json_lib/include/",
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-fstack-protector-all",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_VERSION=\\"v0.25.1-rc.10\\"',
        "-D NGRAPH_DEX_ONLY",
        '-D PROJECT_ROOT_DIR=\\"\\"',
        '-D NGRAPH_STATIC_LIB_ENABLE'
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = 'ngraph_version',
    srcs = glob([
        "src/ngraph/ngraph.cpp"
    ]),
    deps = [
        ":ngraph_headers",
    ],
    copts = [
        "-I external/ngraph/src",
        "-I external/ngraph/src/ngraph",
        "-I external/nlohmann_json_lib/include/",
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-fstack-protector-all",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_VERSION=\\"df4d896be578fc706061929dec1fad31fe37da24\\"',
        "-D NGRAPH_DEX_ONLY",
        '-D PROJECT_ROOT_DIR=\\"\\"',
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = 'cpu_backend',
    hdrs = glob([
        "src/ngraph/runtime/cpu/*.hpp",
        "src/ngraph/runtime/cpu/*.h",
        "src/ngraph/runtime/cpu/kernel/*.hpp",
        "src/ngraph/state/rng_state.hpp",
    ]),
    srcs = glob([
        "src/ngraph/runtime/cpu/cpu_backend.cpp",
        "src/ngraph/runtime/cpu/cpu_builder.cpp",
        "src/ngraph/runtime/cpu/cpu_builder_registry.cpp",
        "src/ngraph/runtime/cpu/cpu_call_frame.cpp",
        "src/ngraph/runtime/cpu/cpu_debug_tracer.cpp",
        "src/ngraph/runtime/cpu/cpu_executor.cpp",
        "src/ngraph/runtime/cpu/cpu_external_function.cpp",
        "src/ngraph/runtime/cpu/cpu_kernels.cpp",
        "src/ngraph/runtime/cpu/cpu_layout_descriptor.cpp",
        "src/ngraph/runtime/cpu/cpu_op_annotations.cpp",
        "src/ngraph/runtime/cpu/cpu_tensor_view_wrapper.cpp",
        "src/ngraph/runtime/cpu/cpu_tensor_view.cpp",
        "src/ngraph/runtime/cpu/cpu_tracing.cpp",
        "src/ngraph/runtime/cpu/cpu_visualize_tree.cpp",
        "src/ngraph/runtime/cpu/cpu_cse.cpp",
        "src/ngraph/runtime/cpu/cpu_debugger.cpp",
        "src/ngraph/runtime/cpu/builder/add.cpp",
        "src/ngraph/runtime/cpu/builder/allreduce.cpp",
        "src/ngraph/runtime/cpu/builder/avg_pool.cpp",
        "src/ngraph/runtime/cpu/builder/argmin.cpp",
        "src/ngraph/runtime/cpu/builder/argmax.cpp",
        "src/ngraph/runtime/cpu/builder/batch_norm.cpp",
        "src/ngraph/runtime/cpu/builder/broadcast.cpp",
        "src/ngraph/runtime/cpu/builder/broadcast_distributed.cpp",
        "src/ngraph/runtime/cpu/builder/bounded_relu.cpp",
        "src/ngraph/runtime/cpu/builder/concat.cpp",
        "src/ngraph/runtime/cpu/builder/convert.cpp",
        "src/ngraph/runtime/cpu/builder/convert_layout.cpp",
        "src/ngraph/runtime/cpu/builder/convolution.cpp",
        "src/ngraph/runtime/cpu/builder/cum_sum.cpp",
        "src/ngraph/runtime/cpu/builder/dot.cpp",
        "src/ngraph/runtime/cpu/builder/dropout.cpp",
        "src/ngraph/runtime/cpu/builder/embedding_lookup.cpp",
        "src/ngraph/runtime/cpu/builder/erf.cpp",
        "src/ngraph/runtime/cpu/builder/gather.cpp",
        "src/ngraph/runtime/cpu/builder/gather_nd.cpp",
        "src/ngraph/runtime/cpu/builder/leaky_relu.cpp",
        "src/ngraph/runtime/cpu/builder/lstm.cpp",
        "src/ngraph/runtime/cpu/builder/lrn.cpp",
        "src/ngraph/runtime/cpu/builder/matmul_bias.cpp",
        "src/ngraph/runtime/cpu/builder/max.cpp",
        "src/ngraph/runtime/cpu/builder/max_pool.cpp",
        "src/ngraph/runtime/cpu/builder/min.cpp",
        "src/ngraph/runtime/cpu/builder/one_hot.cpp",
        "src/ngraph/runtime/cpu/builder/relu.cpp",
        "src/ngraph/runtime/cpu/builder/pad.cpp",
        "src/ngraph/runtime/cpu/builder/product.cpp",
        "src/ngraph/runtime/cpu/builder/reduce_function.cpp",
        "src/ngraph/runtime/cpu/builder/replace_slice.cpp",
        "src/ngraph/runtime/cpu/builder/quantization.cpp",
        "src/ngraph/runtime/cpu/builder/quantized_avg_pool.cpp",
        "src/ngraph/runtime/cpu/builder/quantized_conv.cpp",
        "src/ngraph/runtime/cpu/builder/quantized_concat.cpp",
        "src/ngraph/runtime/cpu/builder/quantized_dot.cpp",
        "src/ngraph/runtime/cpu/builder/quantized_matmul.cpp",
        "src/ngraph/runtime/cpu/builder/quantized_max_pool.cpp",
        "src/ngraph/runtime/cpu/builder/reshape.cpp",
        "src/ngraph/runtime/cpu/builder/reverse.cpp",
        "src/ngraph/runtime/cpu/builder/reverse_sequence.cpp",
        "src/ngraph/runtime/cpu/builder/rnn.cpp",
        "src/ngraph/runtime/cpu/builder/scatter_add.cpp",
        "src/ngraph/runtime/cpu/builder/scatter_nd_add.cpp",
        "src/ngraph/runtime/cpu/builder/select.cpp",
        "src/ngraph/runtime/cpu/builder/sigmoid.cpp",
        "src/ngraph/runtime/cpu/builder/slice.cpp",
        "src/ngraph/runtime/cpu/builder/state.cpp",
        "src/ngraph/runtime/cpu/builder/softmax.cpp",
        "src/ngraph/runtime/cpu/builder/get_output_element.cpp",
        "src/ngraph/runtime/cpu/builder/sum.cpp",
        "src/ngraph/runtime/cpu/builder/topk.cpp",
        "src/ngraph/runtime/cpu/builder/tile.cpp",
        "src/ngraph/runtime/cpu/builder/update_slice.cpp",
        "src/ngraph/runtime/cpu/kernel/pad.cpp",
        "src/ngraph/runtime/cpu/kernel/reduce_max.cpp",
        "src/ngraph/runtime/cpu/kernel/reduce_sum.cpp",
        "src/ngraph/runtime/cpu/kernel/reshape.cpp",
        "src/ngraph/runtime/cpu/mkldnn_emitter.cpp",
        "src/ngraph/runtime/cpu/mkldnn_invoke.cpp",
        "src/ngraph/runtime/cpu/mkldnn_utils.cpp",
        "src/ngraph/runtime/cpu/op/batch_mat_mul_transpose.cpp",
        "src/ngraph/runtime/cpu/op/batch_norm_relu.cpp",
        "src/ngraph/runtime/cpu/op/bounded_relu.cpp",
        "src/ngraph/runtime/cpu/op/conv_add.cpp",
        "src/ngraph/runtime/cpu/op/conv_relu.cpp",
        "src/ngraph/runtime/cpu/op/convert_layout.cpp",
        "src/ngraph/runtime/cpu/op/deconv.cpp",
        "src/ngraph/runtime/cpu/op/dropout.cpp",
        "src/ngraph/runtime/cpu/op/group_conv_bias.cpp",
        "src/ngraph/runtime/cpu/op/halide_op.cpp",
        "src/ngraph/runtime/cpu/op/leaky_relu.cpp",
        "src/ngraph/runtime/cpu/op/loop_kernel.cpp",
        "src/ngraph/runtime/cpu/op/lstm.cpp",
        "src/ngraph/runtime/cpu/op/matmul_bias.cpp",
        "src/ngraph/runtime/cpu/op/max_pool_with_indices.cpp",
        "src/ngraph/runtime/cpu/op/quantized_matmul.cpp",
        "src/ngraph/runtime/cpu/op/rnn.cpp",
        "src/ngraph/runtime/cpu/op/sigmoid_mul.cpp",
        "src/ngraph/runtime/cpu/op/update_slice.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_assignment.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_collapse_dims.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_fusion.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_horizontal_fusion.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_layout.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_loop_kernel_fusion.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_mat_fusion.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_memory_assignment.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_memory_optimization.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_mkldnn_primitive_build.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_rnn_fusion.cpp",
        "src/ngraph/runtime/cpu/pass/cpu_workspace_insertion.cpp",
        "src/ngraph/runtime/cpu/ngraph_version.cpp",
        "src/ngraph/state/rng_state.cpp", 
    ]),
    deps = [
        ":ngraph_headers",
        ":ngraph_core",
        "@eigen",
        "@mkl_dnn",
    ],
    copts = [
        "-I external/ngraph/src",
        "-I external/ngraph/src/ngraph",
        "-I external/nlohmann_json_lib/include/",
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-fstack-protector-all",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_VERSION=\\"0.25.1-rc.10\\"',
        "-D NGRAPH_DEX_ONLY",
        '-D PROJECT_ROOT_DIR=\\"\\"',
        '-D NGRAPH_CPU_STATIC_LIB_ENABLE'
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
        "-Wl,-Bsymbolic-functions",
        "-Wl,--exclude-libs=ALL",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

