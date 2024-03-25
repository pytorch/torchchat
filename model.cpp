#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_RUNTIME_FAILURE;                             \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_RUNTIME_FAILURE;                             \
  }                                                          \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() : prev_mode(aoti_torch_grad_mode_is_enabled()) {
    aoti_torch_grad_mode_set_enabled(false);
  }
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  bool prev_mode;
};

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
  return AOTInductorModelContainerCreateWithDevice(
      container_handle, num_models, is_cpu ? "cpu" : "cuda", cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0" << std::endl;
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map =
      reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
          constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(
      container_handle,
      constant_map_handle,
      /*use_inactive*/ true,
      /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ container->swap_constant_buffer(); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array =
          std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map =
          reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
              constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only
                 // use for CPU models
          "");

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs){CONVERT_EXCEPTION_TO_ERROR_CODE({
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  *ret_num_outputs = model->num_outputs();
})}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

} // extern "C"
// NOTE: Like interface.cpp, this file will be copied into AOTInductor
// generated output. This file is intended to keep implementation
// details separate from the implementation of the AOTI public
// interface. Note also that #includes should go into interface.cpp
// for simplicity of maintenance.

namespace torch {
namespace aot_inductor {
template <typename T>
void convert_output_to_handle(
    const ArrayRefTensor<T>& output,
    AtenTensorHandle& handle) {
  handle = output.expensiveCopyToTensor();
}

template <typename... Ts, std::size_t... Is>
void convert_outputs_to_handles_helper(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles,
    std::index_sequence<Is...>) {
  (convert_output_to_handle(std::get<Is>(outputs), output_handles[Is]), ...);
}
template <typename... Ts>
void convert_outputs_to_handles(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles) {
  convert_outputs_to_handles_helper(
      outputs, output_handles, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void convert_handle_to_arrayref_tensor(
    AtenTensorHandle handle,
    ArrayRefTensor<T>& input) {
  void* data_ptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &data_ptr));
  int64_t dim;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &dim));
  int64_t numel;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(handle, &numel));
  int64_t* sizes;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  int64_t* strides;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  int32_t dtype;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &dtype));
  int32_t device_type;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(handle, &device_type));
  int32_t device_index;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &device_index));

  input = ArrayRefTensor<T>(
      MiniArrayRef<T>(reinterpret_cast<T*>(data_ptr), numel),
      MiniArrayRef<const int64_t>(sizes, dim),
      MiniArrayRef<const int64_t>(strides, dim),
      device_type,
      device_index);
}

template <typename... Ts, std::size_t... Is>
void convert_handles_to_inputs_helper(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs,
    std::index_sequence<Is...>) {
  (convert_handle_to_arrayref_tensor(input_handles[Is], std::get<Is>(inputs)),
   ...);
}

template <typename... Ts>
void convert_handles_to_inputs(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs) {
  convert_handles_to_inputs_helper(
      input_handles, inputs, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void assert_numel(const ArrayRefTensor<T>& tensor, int64_t numel) {
  if (tensor.numel() != numel) {
    std::stringstream err;
    err << "incorrect numel for input tensor. expected " << numel << ", got "
        << tensor.numel();
    throw std::runtime_error(err.str());
  }
}
} // namespace aot_inductor
} // namespace torch
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/model.h>

[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}

#include "cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void
cpp_fused__to_copy_embedding_index_index_put_logical_not_masked_fill_mean_mm_mul_stack_zeros_like_0(
    const long* in_ptr0,
    const bfloat16* in_ptr1,
    const bfloat16* in_ptr2,
    const bfloat16* in_ptr3,
    const long* in_ptr4,
    const bfloat16* in_ptr5,
    const float* in_ptr6,
    const float* in_ptr7,
    const bool* in_ptr8,
    float* out_ptr0,
    float* out_ptr1,
    float* out_ptr2,
    float* out_ptr3,
    float* out_ptr4,
    float* out_ptr5,
    bfloat16* out_ptr6,
    bfloat16* out_ptr7,
    bfloat16* out_ptr8,
    bfloat16* out_ptr9,
    bfloat16* out_ptr10) {
  {{float tmp_acc0 = 0;
  at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = in_ptr0[static_cast<long>(0L)];
    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
    auto tmp2 = tmp0 < 0;
    auto tmp3 = tmp2 ? tmp1 : tmp0;
    AOTI_TORCH_CHECK(
        (0 <= tmp3) & (tmp3 < 32000L),
        "index out of bounds: 0 <= tmp3 < 32000L")
    auto tmp4 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr1 + static_cast<long>(x0 + (288L * tmp3)), 8);
    auto tmp5 = cvt_lowp_fp_to_fp32<bfloat16>(tmp4);
    auto tmp6 = tmp5 * tmp5;
    tmp_acc0_vec = tmp_acc0_vec + tmp6;
  }
  tmp_acc0 = tmp_acc0 +
      at::vec::vec_reduce_all<float>(
                 [](at::vec::Vectorized<float>& x,
                    at::vec::Vectorized<float>& y) { return x + y; },
                 tmp_acc0_vec);
  out_ptr0[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(864L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = in_ptr0[static_cast<long>(0L)];
        auto tmp6 = out_ptr0[static_cast<long>(0L)];
        auto tmp15 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr2 + static_cast<long>(x1), 8);
        auto tmp19 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr3 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = decltype(tmp0)(tmp0 + 32000);
        auto tmp2 = tmp0 < 0;
        auto tmp3 = tmp2 ? tmp1 : tmp0;
        AOTI_TORCH_CHECK(
            (0 <= tmp3) & (tmp3 < 32000L),
            "index out of bounds: 0 <= tmp3 < 32000L")
        auto tmp4 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr1 + static_cast<long>(x1 + (288L * tmp3)), 8);
        auto tmp5 = cvt_lowp_fp_to_fp32<bfloat16>(tmp4);
        auto tmp7 = static_cast<float>(288.0);
        auto tmp8 = tmp6 / tmp7;
        auto tmp9 = static_cast<float>(1e-05);
        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
        auto tmp11 = 1 / std::sqrt(tmp10);
        auto tmp12 = at::vec::Vectorized<float>(tmp11);
        auto tmp13 = tmp5 * tmp12;
        auto tmp14 = (tmp13);
        auto tmp16 = cvt_lowp_fp_to_fp32<bfloat16>(tmp15);
        auto tmp17 = tmp14 * tmp16;
        auto tmp18 = (tmp17);
        auto tmp20 = cvt_lowp_fp_to_fp32<bfloat16>(tmp19);
        auto tmp21 = tmp18 * tmp20;
        tmp_acc0_vec = tmp_acc0_vec + tmp21;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(24L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr1[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp2 = in_ptr4[static_cast<long>(0L)];
      auto tmp9 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr1[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp18 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr1[static_cast<long>(
              288L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp21 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr1[static_cast<long>(
              289L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp1 = (tmp0);
      auto tmp3 = decltype(tmp2)(tmp2 + 2048);
      auto tmp4 = tmp2 < 0;
      auto tmp5 = tmp4 ? tmp3 : tmp2;
      AOTI_TORCH_CHECK(
          (0 <= tmp5) & (tmp5 < 2048L),
          "index out of bounds: 0 <= tmp5 < 2048L")
      auto tmp6 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr5[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp7 = cvt_lowp_fp_to_fp32<bfloat16>(tmp6);
      auto tmp8 = tmp1 * tmp7;
      auto tmp10 = (tmp9);
      auto tmp11 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr5[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
      auto tmp13 = tmp10 * tmp12;
      auto tmp14 = tmp8 - tmp13;
      auto tmp15 = tmp10 * tmp7;
      auto tmp16 = tmp1 * tmp12;
      auto tmp17 = tmp15 + tmp16;
      auto tmp19 = (tmp18);
      auto tmp20 = tmp19 * tmp7;
      auto tmp22 = (tmp21);
      auto tmp23 = tmp22 * tmp12;
      auto tmp24 = tmp20 - tmp23;
      auto tmp25 = tmp22 * tmp7;
      auto tmp26 = tmp19 * tmp12;
      auto tmp27 = tmp25 + tmp26;
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp14.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr2[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp17.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr3[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp24.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr4[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp27.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr5[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr4[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          in_ptr6 + static_cast<long>(x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr6 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr4[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          out_ptr1 + static_cast<long>(576L + x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr6 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 =
        at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
    auto tmp1 = cvt_fp32_to_lowp_fp<bfloat16>(tmp0);
    tmp1.store(out_ptr7 + static_cast<long>(x0), 8);
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(352L);
       x0 += static_cast<long>(1L)) {
    auto tmp0 = in_ptr4[static_cast<long>(0L)];
    auto tmp1 = decltype(tmp0)(tmp0 + 352);
    auto tmp2 = tmp0 < 0;
    auto tmp3 = tmp2 ? tmp1 : tmp0;
    AOTI_TORCH_CHECK(
        (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
    auto tmp4 = in_ptr8[static_cast<long>(x0 + (352L * tmp3))];
    auto tmp5 = !tmp4;
    auto tmp6 = -std::numeric_limits<float>::infinity();
    auto tmp7 = static_cast<float>(0.0);
    auto tmp8 = tmp5 ? tmp6 : tmp7;
    auto tmp9 = c10::convert<bfloat16>(tmp8);
    out_ptr8[static_cast<long>(x0)] = tmp9;
    out_ptr9[static_cast<long>(x0)] = tmp9;
    out_ptr10[static_cast<long>(x0)] = tmp9;
  }
}
}

#include "cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void
cpp_fused__to_copy_add_embedding_index_put_mean_mm_mul_rsqrt_stack_1(
    const bfloat16* in_ptr0,
    const bfloat16* in_ptr1,
    const long* in_ptr2,
    const bfloat16* in_ptr3,
    const bfloat16* in_ptr4,
    const bfloat16* in_ptr5,
    const bfloat16* in_ptr6,
    const bfloat16* in_ptr7,
    const bfloat16* in_ptr8,
    const bfloat16* in_ptr9,
    const long* in_ptr10,
    const bfloat16* in_ptr11,
    const float* in_ptr12,
    const float* in_ptr13,
    float* out_ptr0,
    float* out_ptr1,
    bfloat16* out_ptr2,
    float* out_ptr3,
    float* out_ptr4,
    float* out_ptr5,
    float* out_ptr6,
    float* out_ptr7,
    float* out_ptr8,
    float* out_ptr9,
    float* out_ptr10,
    float* out_ptr11,
    float* out_ptr12,
    bfloat16* out_ptr13,
    bfloat16* out_ptr14) {
  {
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
         x0 += static_cast<long>(1L)) {
      {
        float tmp_acc0 = 0;
        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
        for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
             x1 += static_cast<long>(8L)) {
          auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr0 + static_cast<long>(x1), 8);
          auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr1 + static_cast<long>(x1 + (288L * x0)), 8);
          auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
          auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
          auto tmp4 = tmp1 * tmp3;
          tmp_acc0_vec = tmp_acc0_vec + tmp4;
        }
        tmp_acc0 = tmp_acc0 +
            at::vec::vec_reduce_all<float>(
                       [](at::vec::Vectorized<float>& x,
                          at::vec::Vectorized<float>& y) { return x + y; },
                       tmp_acc0_vec);
        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      }
    }
  }
  {{float tmp_acc0 = 0;
  at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = in_ptr2[static_cast<long>(0L)];
    auto tmp6 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
    auto tmp2 = tmp0 < 0;
    auto tmp3 = tmp2 ? tmp1 : tmp0;
    AOTI_TORCH_CHECK(
        (0 <= tmp3) & (tmp3 < 32000L),
        "index out of bounds: 0 <= tmp3 < 32000L")
    auto tmp4 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr3 + static_cast<long>(x0 + (288L * tmp3)), 8);
    auto tmp5 = cvt_lowp_fp_to_fp32<bfloat16>(tmp4);
    auto tmp7 = (tmp6);
    auto tmp8 = tmp5 + tmp7;
    auto tmp9 = (tmp8);
    auto tmp10 = tmp9 * tmp9;
    tmp_acc0_vec = tmp_acc0_vec + tmp10;
  }
  tmp_acc0 = tmp_acc0 +
      at::vec::vec_reduce_all<float>(
                 [](at::vec::Vectorized<float>& x,
                    at::vec::Vectorized<float>& y) { return x + y; },
                 tmp_acc0_vec);
  out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = in_ptr2[static_cast<long>(0L)];
    auto tmp6 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp10 = out_ptr1[static_cast<long>(0L)];
    auto tmp19 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr4 + static_cast<long>(x0), 8);
    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
    auto tmp2 = tmp0 < 0;
    auto tmp3 = tmp2 ? tmp1 : tmp0;
    AOTI_TORCH_CHECK(
        (0 <= tmp3) & (tmp3 < 32000L),
        "index out of bounds: 0 <= tmp3 < 32000L")
    auto tmp4 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr3 + static_cast<long>(x0 + (288L * tmp3)), 8);
    auto tmp5 = cvt_lowp_fp_to_fp32<bfloat16>(tmp4);
    auto tmp7 = (tmp6);
    auto tmp8 = tmp5 + tmp7;
    auto tmp9 = (tmp8);
    auto tmp11 = static_cast<float>(288.0);
    auto tmp12 = tmp10 / tmp11;
    auto tmp13 = static_cast<float>(1e-05);
    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
    auto tmp15 = 1 / std::sqrt(tmp14);
    auto tmp16 = at::vec::Vectorized<float>(tmp15);
    auto tmp17 = tmp9 * tmp16;
    auto tmp18 = (tmp17);
    auto tmp20 = cvt_lowp_fp_to_fp32<bfloat16>(tmp19);
    auto tmp21 = tmp18 * tmp20;
    auto tmp22 = cvt_fp32_to_lowp_fp<bfloat16>(tmp21);
    tmp22.store(out_ptr2 + static_cast<long>(x0), 8);
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(768L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      float tmp_acc1 = 0;
      at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            out_ptr2 + static_cast<long>(x1), 8);
        auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr5 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp5 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr6 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
        auto tmp4 = tmp1 * tmp3;
        auto tmp6 = cvt_lowp_fp_to_fp32<bfloat16>(tmp5);
        auto tmp7 = tmp1 * tmp6;
        tmp_acc0_vec = tmp_acc0_vec + tmp4;
        tmp_acc1_vec = tmp_acc1_vec + tmp7;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      tmp_acc1 = tmp_acc1 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc1_vec);
      out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(768L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
        auto tmp5 =
            at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
        auto tmp9 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr7 + static_cast<long>(x1 + (768L * x0)), 8);
        auto tmp1 = (tmp0);
        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + tmp1.neg().exp());
        auto tmp3 = tmp1 * tmp2;
        auto tmp4 = (tmp3);
        auto tmp6 = (tmp5);
        auto tmp7 = tmp4 * tmp6;
        auto tmp8 = (tmp7);
        auto tmp10 = cvt_lowp_fp_to_fp32<bfloat16>(tmp9);
        auto tmp11 = tmp8 * tmp10;
        tmp_acc0_vec = tmp_acc0_vec + tmp11;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{{float tmp_acc0 = 0;
at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
     x0 += static_cast<long>(8L)) {
  auto tmp0 = in_ptr2[static_cast<long>(0L)];
  auto tmp6 =
      at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
  auto tmp9 =
      at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
  auto tmp1 = decltype(tmp0)(tmp0 + 32000);
  auto tmp2 = tmp0 < 0;
  auto tmp3 = tmp2 ? tmp1 : tmp0;
  AOTI_TORCH_CHECK(
      (0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
  auto tmp4 = at::vec::Vectorized<bfloat16>::loadu(
      in_ptr3 + static_cast<long>(x0 + (288L * tmp3)), 8);
  auto tmp5 = cvt_lowp_fp_to_fp32<bfloat16>(tmp4);
  auto tmp7 = (tmp6);
  auto tmp8 = tmp5 + tmp7;
  auto tmp10 = (tmp9);
  auto tmp11 = tmp8 + tmp10;
  auto tmp12 = (tmp11);
  auto tmp13 = tmp12 * tmp12;
  tmp_acc0_vec = tmp_acc0_vec + tmp13;
}
tmp_acc0 = tmp_acc0 +
    at::vec::vec_reduce_all<float>(
               [](at::vec::Vectorized<float>& x,
                  at::vec::Vectorized<float>& y) { return x + y; },
               tmp_acc0_vec);
out_ptr6[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = in_ptr2[static_cast<long>(0L)];
    auto tmp6 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp9 =
        at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
    auto tmp13 = out_ptr6[static_cast<long>(0L)];
    auto tmp22 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr8 + static_cast<long>(x0), 8);
    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
    auto tmp2 = tmp0 < 0;
    auto tmp3 = tmp2 ? tmp1 : tmp0;
    AOTI_TORCH_CHECK(
        (0 <= tmp3) & (tmp3 < 32000L),
        "index out of bounds: 0 <= tmp3 < 32000L")
    auto tmp4 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr3 + static_cast<long>(x0 + (288L * tmp3)), 8);
    auto tmp5 = cvt_lowp_fp_to_fp32<bfloat16>(tmp4);
    auto tmp7 = (tmp6);
    auto tmp8 = tmp5 + tmp7;
    auto tmp10 = (tmp9);
    auto tmp11 = tmp8 + tmp10;
    auto tmp12 = (tmp11);
    auto tmp14 = static_cast<float>(288.0);
    auto tmp15 = tmp13 / tmp14;
    auto tmp16 = static_cast<float>(1e-05);
    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
    auto tmp18 = 1 / std::sqrt(tmp17);
    auto tmp19 = at::vec::Vectorized<float>(tmp18);
    auto tmp20 = tmp12 * tmp19;
    auto tmp21 = (tmp20);
    auto tmp23 = cvt_lowp_fp_to_fp32<bfloat16>(tmp22);
    auto tmp24 = tmp21 * tmp23;
    auto tmp25 = (tmp24);
    tmp25.store(out_ptr7 + static_cast<long>(x0));
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(864L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x1));
        auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr9 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp2 = cvt_lowp_fp_to_fp32<bfloat16>(tmp1);
        auto tmp3 = tmp0 * tmp2;
        tmp_acc0_vec = tmp_acc0_vec + tmp3;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr8[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(24L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp2 = in_ptr10[static_cast<long>(0L)];
      auto tmp9 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp18 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              288L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp21 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              289L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp1 = (tmp0);
      auto tmp3 = decltype(tmp2)(tmp2 + 2048);
      auto tmp4 = tmp2 < 0;
      auto tmp5 = tmp4 ? tmp3 : tmp2;
      AOTI_TORCH_CHECK(
          (0 <= tmp5) & (tmp5 < 2048L),
          "index out of bounds: 0 <= tmp5 < 2048L")
      auto tmp6 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr11[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp7 = cvt_lowp_fp_to_fp32<bfloat16>(tmp6);
      auto tmp8 = tmp1 * tmp7;
      auto tmp10 = (tmp9);
      auto tmp11 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr11[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
      auto tmp13 = tmp10 * tmp12;
      auto tmp14 = tmp8 - tmp13;
      auto tmp15 = tmp10 * tmp7;
      auto tmp16 = tmp1 * tmp12;
      auto tmp17 = tmp15 + tmp16;
      auto tmp19 = (tmp18);
      auto tmp20 = tmp19 * tmp7;
      auto tmp22 = (tmp21);
      auto tmp23 = tmp22 * tmp12;
      auto tmp24 = tmp20 - tmp23;
      auto tmp25 = tmp22 * tmp7;
      auto tmp26 = tmp19 * tmp12;
      auto tmp27 = tmp25 + tmp26;
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp14.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr9[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp17.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr10[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp24.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr11[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp27.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr12[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr10[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          in_ptr12 + static_cast<long>(x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr13 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr10[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          out_ptr8 + static_cast<long>(576L + x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr13 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 =
        at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
    auto tmp1 = cvt_fp32_to_lowp_fp<bfloat16>(tmp0);
    tmp1.store(out_ptr14 + static_cast<long>(x0), 8);
  }
}
}

#include "cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void cpp_fused__to_copy_add_embedding_index_put_mean_mm_mul_stack_2(
    const bfloat16* in_ptr0,
    const bfloat16* in_ptr1,
    const long* in_ptr2,
    const bfloat16* in_ptr3,
    const float* in_ptr4,
    const float* in_ptr5,
    const bfloat16* in_ptr6,
    const bfloat16* in_ptr7,
    const bfloat16* in_ptr8,
    const bfloat16* in_ptr9,
    const bfloat16* in_ptr10,
    const bfloat16* in_ptr11,
    const long* in_ptr12,
    const bfloat16* in_ptr13,
    const float* in_ptr14,
    const float* in_ptr15,
    float* out_ptr0,
    bfloat16* out_ptr1,
    float* out_ptr2,
    float* out_ptr3,
    float* out_ptr4,
    float* out_ptr5,
    float* out_ptr6,
    float* out_ptr7,
    float* out_ptr8,
    float* out_ptr9,
    float* out_ptr10,
    float* out_ptr11,
    bfloat16* out_ptr12,
    bfloat16* out_ptr13) {
  {
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
         x0 += static_cast<long>(1L)) {
      {
        float tmp_acc0 = 0;
        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
        for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
             x1 += static_cast<long>(8L)) {
          auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr0 + static_cast<long>(x1), 8);
          auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr1 + static_cast<long>(x1 + (288L * x0)), 8);
          auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
          auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
          auto tmp4 = tmp1 * tmp3;
          tmp_acc0_vec = tmp_acc0_vec + tmp4;
        }
        tmp_acc0 = tmp_acc0 +
            at::vec::vec_reduce_all<float>(
                       [](at::vec::Vectorized<float>& x,
                          at::vec::Vectorized<float>& y) { return x + y; },
                       tmp_acc0_vec);
        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      }
    }
  }
  {{float tmp_acc0 = 0;
  at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = in_ptr2[static_cast<long>(0L)];
    auto tmp6 =
        at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
    auto tmp9 =
        at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
    auto tmp12 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
    auto tmp2 = tmp0 < 0;
    auto tmp3 = tmp2 ? tmp1 : tmp0;
    AOTI_TORCH_CHECK(
        (0 <= tmp3) & (tmp3 < 32000L),
        "index out of bounds: 0 <= tmp3 < 32000L")
    auto tmp4 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr3 + static_cast<long>(x0 + (288L * tmp3)), 8);
    auto tmp5 = cvt_lowp_fp_to_fp32<bfloat16>(tmp4);
    auto tmp7 = (tmp6);
    auto tmp8 = tmp5 + tmp7;
    auto tmp10 = (tmp9);
    auto tmp11 = tmp8 + tmp10;
    auto tmp13 = (tmp12);
    auto tmp14 = tmp11 + tmp13;
    auto tmp15 = cvt_fp32_to_lowp_fp<bfloat16>(tmp14);
    auto tmp16 = tmp14 * tmp14;
    tmp15.store(out_ptr1 + static_cast<long>(x0), 8);
    tmp_acc0_vec = tmp_acc0_vec + tmp16;
  }
  tmp_acc0 = tmp_acc0 +
      at::vec::vec_reduce_all<float>(
                 [](at::vec::Vectorized<float>& x,
                    at::vec::Vectorized<float>& y) { return x + y; },
                 tmp_acc0_vec);
  out_ptr2[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(768L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      float tmp_acc1 = 0;
      at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            out_ptr1 + static_cast<long>(x1), 8);
        auto tmp2 = out_ptr2[static_cast<long>(0L)];
        auto tmp11 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr6 + static_cast<long>(x1), 8);
        auto tmp15 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr7 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp18 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr8 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = static_cast<float>(288.0);
        auto tmp4 = tmp2 / tmp3;
        auto tmp5 = static_cast<float>(1e-05);
        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
        auto tmp7 = 1 / std::sqrt(tmp6);
        auto tmp8 = at::vec::Vectorized<float>(tmp7);
        auto tmp9 = tmp1 * tmp8;
        auto tmp10 = (tmp9);
        auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
        auto tmp13 = tmp10 * tmp12;
        auto tmp14 = (tmp13);
        auto tmp16 = cvt_lowp_fp_to_fp32<bfloat16>(tmp15);
        auto tmp17 = tmp14 * tmp16;
        auto tmp19 = cvt_lowp_fp_to_fp32<bfloat16>(tmp18);
        auto tmp20 = tmp14 * tmp19;
        tmp_acc0_vec = tmp_acc0_vec + tmp17;
        tmp_acc1_vec = tmp_acc1_vec + tmp20;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      tmp_acc1 = tmp_acc1 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc1_vec);
      out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(768L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
        auto tmp5 =
            at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
        auto tmp9 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr9 + static_cast<long>(x1 + (768L * x0)), 8);
        auto tmp1 = (tmp0);
        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + tmp1.neg().exp());
        auto tmp3 = tmp1 * tmp2;
        auto tmp4 = (tmp3);
        auto tmp6 = (tmp5);
        auto tmp7 = tmp4 * tmp6;
        auto tmp8 = (tmp7);
        auto tmp10 = cvt_lowp_fp_to_fp32<bfloat16>(tmp9);
        auto tmp11 = tmp8 * tmp10;
        tmp_acc0_vec = tmp_acc0_vec + tmp11;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{{float tmp_acc0 = 0;
at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
     x0 += static_cast<long>(8L)) {
  auto tmp0 =
      at::vec::Vectorized<bfloat16>::loadu(out_ptr1 + static_cast<long>(x0), 8);
  auto tmp2 =
      at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
  auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
  auto tmp3 = (tmp2);
  auto tmp4 = tmp1 + tmp3;
  auto tmp5 = (tmp4);
  auto tmp6 = tmp5 * tmp5;
  tmp_acc0_vec = tmp_acc0_vec + tmp6;
}
tmp_acc0 = tmp_acc0 +
    at::vec::vec_reduce_all<float>(
               [](at::vec::Vectorized<float>& x,
                  at::vec::Vectorized<float>& y) { return x + y; },
               tmp_acc0_vec);
out_ptr6[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(864L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            out_ptr1 + static_cast<long>(x1), 8);
        auto tmp2 =
            at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1));
        auto tmp6 = out_ptr6[static_cast<long>(0L)];
        auto tmp15 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr10 + static_cast<long>(x1), 8);
        auto tmp19 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr11 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = (tmp2);
        auto tmp4 = tmp1 + tmp3;
        auto tmp5 = (tmp4);
        auto tmp7 = static_cast<float>(288.0);
        auto tmp8 = tmp6 / tmp7;
        auto tmp9 = static_cast<float>(1e-05);
        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
        auto tmp11 = 1 / std::sqrt(tmp10);
        auto tmp12 = at::vec::Vectorized<float>(tmp11);
        auto tmp13 = tmp5 * tmp12;
        auto tmp14 = (tmp13);
        auto tmp16 = cvt_lowp_fp_to_fp32<bfloat16>(tmp15);
        auto tmp17 = tmp14 * tmp16;
        auto tmp18 = (tmp17);
        auto tmp20 = cvt_lowp_fp_to_fp32<bfloat16>(tmp19);
        auto tmp21 = tmp18 * tmp20;
        tmp_acc0_vec = tmp_acc0_vec + tmp21;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(24L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr7[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp2 = in_ptr12[static_cast<long>(0L)];
      auto tmp9 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr7[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp18 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr7[static_cast<long>(
              288L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp21 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr7[static_cast<long>(
              289L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp1 = (tmp0);
      auto tmp3 = decltype(tmp2)(tmp2 + 2048);
      auto tmp4 = tmp2 < 0;
      auto tmp5 = tmp4 ? tmp3 : tmp2;
      AOTI_TORCH_CHECK(
          (0 <= tmp5) & (tmp5 < 2048L),
          "index out of bounds: 0 <= tmp5 < 2048L")
      auto tmp6 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr13[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp7 = cvt_lowp_fp_to_fp32<bfloat16>(tmp6);
      auto tmp8 = tmp1 * tmp7;
      auto tmp10 = (tmp9);
      auto tmp11 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr13[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
      auto tmp13 = tmp10 * tmp12;
      auto tmp14 = tmp8 - tmp13;
      auto tmp15 = tmp10 * tmp7;
      auto tmp16 = tmp1 * tmp12;
      auto tmp17 = tmp15 + tmp16;
      auto tmp19 = (tmp18);
      auto tmp20 = tmp19 * tmp7;
      auto tmp22 = (tmp21);
      auto tmp23 = tmp22 * tmp12;
      auto tmp24 = tmp20 - tmp23;
      auto tmp25 = tmp22 * tmp7;
      auto tmp26 = tmp19 * tmp12;
      auto tmp27 = tmp25 + tmp26;
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp14.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr8[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp17.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr9[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp24.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr10[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp27.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr11[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr12[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          in_ptr14 + static_cast<long>(x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr12 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr12[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          out_ptr7 + static_cast<long>(576L + x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr12 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 =
        at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
    auto tmp1 = cvt_fp32_to_lowp_fp<bfloat16>(tmp0);
    tmp1.store(out_ptr13 + static_cast<long>(x0), 8);
  }
}
}

#include "cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void
cpp_fused__to_copy_add_index_index_put_logical_not_masked_fill_mean_mm_mul_rsqrt_stack_zeros_like_3(
    const bfloat16* in_ptr0,
    const bfloat16* in_ptr1,
    const bfloat16* in_ptr2,
    const float* in_ptr3,
    const bfloat16* in_ptr4,
    const bfloat16* in_ptr5,
    const bfloat16* in_ptr6,
    const bfloat16* in_ptr7,
    const bfloat16* in_ptr8,
    const bfloat16* in_ptr9,
    const long* in_ptr10,
    const bfloat16* in_ptr11,
    const float* in_ptr12,
    const float* in_ptr13,
    const bool* in_ptr14,
    float* out_ptr0,
    float* out_ptr1,
    bfloat16* out_ptr2,
    float* out_ptr3,
    float* out_ptr4,
    float* out_ptr5,
    float* out_ptr6,
    float* out_ptr7,
    float* out_ptr8,
    float* out_ptr9,
    float* out_ptr10,
    float* out_ptr11,
    float* out_ptr12,
    bfloat16* out_ptr13,
    bfloat16* out_ptr14,
    bfloat16* out_ptr15,
    bfloat16* out_ptr16,
    bfloat16* out_ptr17) {
  {
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
         x0 += static_cast<long>(1L)) {
      {
        float tmp_acc0 = 0;
        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
        for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
             x1 += static_cast<long>(8L)) {
          auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr0 + static_cast<long>(x1), 8);
          auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr1 + static_cast<long>(x1 + (288L * x0)), 8);
          auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
          auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
          auto tmp4 = tmp1 * tmp3;
          tmp_acc0_vec = tmp_acc0_vec + tmp4;
        }
        tmp_acc0 = tmp_acc0 +
            at::vec::vec_reduce_all<float>(
                       [](at::vec::Vectorized<float>& x,
                          at::vec::Vectorized<float>& y) { return x + y; },
                       tmp_acc0_vec);
        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      }
    }
  }
  {{float tmp_acc0 = 0;
  at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr2 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp8 = (tmp7);
    auto tmp9 = tmp8 * tmp8;
    tmp_acc0_vec = tmp_acc0_vec + tmp9;
  }
  tmp_acc0 = tmp_acc0 +
      at::vec::vec_reduce_all<float>(
                 [](at::vec::Vectorized<float>& x,
                    at::vec::Vectorized<float>& y) { return x + y; },
                 tmp_acc0_vec);
  out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr2 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp9 = out_ptr1[static_cast<long>(0L)];
    auto tmp18 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr4 + static_cast<long>(x0), 8);
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp8 = (tmp7);
    auto tmp10 = static_cast<float>(288.0);
    auto tmp11 = tmp9 / tmp10;
    auto tmp12 = static_cast<float>(1e-05);
    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
    auto tmp14 = 1 / std::sqrt(tmp13);
    auto tmp15 = at::vec::Vectorized<float>(tmp14);
    auto tmp16 = tmp8 * tmp15;
    auto tmp17 = (tmp16);
    auto tmp19 = cvt_lowp_fp_to_fp32<bfloat16>(tmp18);
    auto tmp20 = tmp17 * tmp19;
    auto tmp21 = cvt_fp32_to_lowp_fp<bfloat16>(tmp20);
    tmp21.store(out_ptr2 + static_cast<long>(x0), 8);
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(768L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      float tmp_acc1 = 0;
      at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            out_ptr2 + static_cast<long>(x1), 8);
        auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr5 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp5 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr6 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
        auto tmp4 = tmp1 * tmp3;
        auto tmp6 = cvt_lowp_fp_to_fp32<bfloat16>(tmp5);
        auto tmp7 = tmp1 * tmp6;
        tmp_acc0_vec = tmp_acc0_vec + tmp4;
        tmp_acc1_vec = tmp_acc1_vec + tmp7;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      tmp_acc1 = tmp_acc1 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc1_vec);
      out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(768L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
        auto tmp5 =
            at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
        auto tmp9 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr7 + static_cast<long>(x1 + (768L * x0)), 8);
        auto tmp1 = (tmp0);
        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + tmp1.neg().exp());
        auto tmp3 = tmp1 * tmp2;
        auto tmp4 = (tmp3);
        auto tmp6 = (tmp5);
        auto tmp7 = tmp4 * tmp6;
        auto tmp8 = (tmp7);
        auto tmp10 = cvt_lowp_fp_to_fp32<bfloat16>(tmp9);
        auto tmp11 = tmp8 * tmp10;
        tmp_acc0_vec = tmp_acc0_vec + tmp11;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{{float tmp_acc0 = 0;
at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
     x0 += static_cast<long>(8L)) {
  auto tmp0 =
      at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x0), 8);
  auto tmp2 =
      at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
  auto tmp5 =
      at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
  auto tmp8 =
      at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
  auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
  auto tmp3 = (tmp2);
  auto tmp4 = tmp1 + tmp3;
  auto tmp6 = (tmp5);
  auto tmp7 = tmp4 + tmp6;
  auto tmp9 = (tmp8);
  auto tmp10 = tmp7 + tmp9;
  auto tmp11 = (tmp10);
  auto tmp12 = tmp11 * tmp11;
  tmp_acc0_vec = tmp_acc0_vec + tmp12;
}
tmp_acc0 = tmp_acc0 +
    at::vec::vec_reduce_all<float>(
               [](at::vec::Vectorized<float>& x,
                  at::vec::Vectorized<float>& y) { return x + y; },
               tmp_acc0_vec);
out_ptr6[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr2 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp8 =
        at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
    auto tmp12 = out_ptr6[static_cast<long>(0L)];
    auto tmp21 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr8 + static_cast<long>(x0), 8);
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp9 = (tmp8);
    auto tmp10 = tmp7 + tmp9;
    auto tmp11 = (tmp10);
    auto tmp13 = static_cast<float>(288.0);
    auto tmp14 = tmp12 / tmp13;
    auto tmp15 = static_cast<float>(1e-05);
    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
    auto tmp17 = 1 / std::sqrt(tmp16);
    auto tmp18 = at::vec::Vectorized<float>(tmp17);
    auto tmp19 = tmp11 * tmp18;
    auto tmp20 = (tmp19);
    auto tmp22 = cvt_lowp_fp_to_fp32<bfloat16>(tmp21);
    auto tmp23 = tmp20 * tmp22;
    auto tmp24 = (tmp23);
    tmp24.store(out_ptr7 + static_cast<long>(x0));
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(864L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x1));
        auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr9 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp2 = cvt_lowp_fp_to_fp32<bfloat16>(tmp1);
        auto tmp3 = tmp0 * tmp2;
        tmp_acc0_vec = tmp_acc0_vec + tmp3;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr8[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(24L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp2 = in_ptr10[static_cast<long>(0L)];
      auto tmp9 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp18 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              288L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp21 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              289L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp1 = (tmp0);
      auto tmp3 = decltype(tmp2)(tmp2 + 2048);
      auto tmp4 = tmp2 < 0;
      auto tmp5 = tmp4 ? tmp3 : tmp2;
      AOTI_TORCH_CHECK(
          (0 <= tmp5) & (tmp5 < 2048L),
          "index out of bounds: 0 <= tmp5 < 2048L")
      auto tmp6 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr11[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp7 = cvt_lowp_fp_to_fp32<bfloat16>(tmp6);
      auto tmp8 = tmp1 * tmp7;
      auto tmp10 = (tmp9);
      auto tmp11 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr11[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
      auto tmp13 = tmp10 * tmp12;
      auto tmp14 = tmp8 - tmp13;
      auto tmp15 = tmp10 * tmp7;
      auto tmp16 = tmp1 * tmp12;
      auto tmp17 = tmp15 + tmp16;
      auto tmp19 = (tmp18);
      auto tmp20 = tmp19 * tmp7;
      auto tmp22 = (tmp21);
      auto tmp23 = tmp22 * tmp12;
      auto tmp24 = tmp20 - tmp23;
      auto tmp25 = tmp22 * tmp7;
      auto tmp26 = tmp19 * tmp12;
      auto tmp27 = tmp25 + tmp26;
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp14.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr9[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp17.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr10[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp24.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr11[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp27.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr12[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr10[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          in_ptr12 + static_cast<long>(x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr13 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr10[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          out_ptr8 + static_cast<long>(576L + x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr13 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 =
        at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
    auto tmp1 = cvt_fp32_to_lowp_fp<bfloat16>(tmp0);
    tmp1.store(out_ptr14 + static_cast<long>(x0), 8);
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(352L);
       x0 += static_cast<long>(1L)) {
    auto tmp0 = in_ptr10[static_cast<long>(0L)];
    auto tmp1 = decltype(tmp0)(tmp0 + 352);
    auto tmp2 = tmp0 < 0;
    auto tmp3 = tmp2 ? tmp1 : tmp0;
    AOTI_TORCH_CHECK(
        (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
    auto tmp4 = in_ptr14[static_cast<long>(x0 + (352L * tmp3))];
    auto tmp5 = !tmp4;
    auto tmp6 = -std::numeric_limits<float>::infinity();
    auto tmp7 = static_cast<float>(0.0);
    auto tmp8 = tmp5 ? tmp6 : tmp7;
    auto tmp9 = c10::convert<bfloat16>(tmp8);
    out_ptr15[static_cast<long>(x0)] = tmp9;
    out_ptr16[static_cast<long>(x0)] = tmp9;
    out_ptr17[static_cast<long>(x0)] = tmp9;
  }
}
}

#include "cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void cpp_fused__to_copy_add_index_put_mean_mm_mul_stack_4(
    bfloat16* in_out_ptr0,
    const bfloat16* in_ptr0,
    const bfloat16* in_ptr1,
    const float* in_ptr2,
    const float* in_ptr3,
    const float* in_ptr4,
    const bfloat16* in_ptr5,
    const bfloat16* in_ptr6,
    const bfloat16* in_ptr7,
    const bfloat16* in_ptr8,
    const bfloat16* in_ptr9,
    const bfloat16* in_ptr10,
    const long* in_ptr11,
    const bfloat16* in_ptr12,
    const float* in_ptr13,
    const float* in_ptr14,
    float* out_ptr0,
    float* out_ptr1,
    float* out_ptr2,
    float* out_ptr3,
    float* out_ptr4,
    float* out_ptr5,
    float* out_ptr6,
    float* out_ptr7,
    float* out_ptr8,
    float* out_ptr9,
    float* out_ptr10,
    bfloat16* out_ptr11,
    bfloat16* out_ptr12) {
  {
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
         x0 += static_cast<long>(1L)) {
      {
        float tmp_acc0 = 0;
        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
        for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
             x1 += static_cast<long>(8L)) {
          auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr0 + static_cast<long>(x1), 8);
          auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr1 + static_cast<long>(x1 + (288L * x0)), 8);
          auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
          auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
          auto tmp4 = tmp1 * tmp3;
          tmp_acc0_vec = tmp_acc0_vec + tmp4;
        }
        tmp_acc0 = tmp_acc0 +
            at::vec::vec_reduce_all<float>(
                       [](at::vec::Vectorized<float>& x,
                          at::vec::Vectorized<float>& y) { return x + y; },
                       tmp_acc0_vec);
        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      }
    }
  }
  {{float tmp_acc0 = 0;
  at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_out_ptr0 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp8 =
        at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
    auto tmp11 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp9 = (tmp8);
    auto tmp10 = tmp7 + tmp9;
    auto tmp12 = (tmp11);
    auto tmp13 = tmp10 + tmp12;
    auto tmp14 = cvt_fp32_to_lowp_fp<bfloat16>(tmp13);
    auto tmp15 = tmp13 * tmp13;
    tmp14.store(in_out_ptr0 + static_cast<long>(x0), 8);
    tmp_acc0_vec = tmp_acc0_vec + tmp15;
  }
  tmp_acc0 = tmp_acc0 +
      at::vec::vec_reduce_all<float>(
                 [](at::vec::Vectorized<float>& x,
                    at::vec::Vectorized<float>& y) { return x + y; },
                 tmp_acc0_vec);
  out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(768L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      float tmp_acc1 = 0;
      at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            in_out_ptr0 + static_cast<long>(x1), 8);
        auto tmp2 = out_ptr1[static_cast<long>(0L)];
        auto tmp11 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr5 + static_cast<long>(x1), 8);
        auto tmp15 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr6 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp18 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr7 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = static_cast<float>(288.0);
        auto tmp4 = tmp2 / tmp3;
        auto tmp5 = static_cast<float>(1e-05);
        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
        auto tmp7 = 1 / std::sqrt(tmp6);
        auto tmp8 = at::vec::Vectorized<float>(tmp7);
        auto tmp9 = tmp1 * tmp8;
        auto tmp10 = (tmp9);
        auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
        auto tmp13 = tmp10 * tmp12;
        auto tmp14 = (tmp13);
        auto tmp16 = cvt_lowp_fp_to_fp32<bfloat16>(tmp15);
        auto tmp17 = tmp14 * tmp16;
        auto tmp19 = cvt_lowp_fp_to_fp32<bfloat16>(tmp18);
        auto tmp20 = tmp14 * tmp19;
        tmp_acc0_vec = tmp_acc0_vec + tmp17;
        tmp_acc1_vec = tmp_acc1_vec + tmp20;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      tmp_acc1 = tmp_acc1 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc1_vec);
      out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(768L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
        auto tmp5 =
            at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
        auto tmp9 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr8 + static_cast<long>(x1 + (768L * x0)), 8);
        auto tmp1 = (tmp0);
        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + tmp1.neg().exp());
        auto tmp3 = tmp1 * tmp2;
        auto tmp4 = (tmp3);
        auto tmp6 = (tmp5);
        auto tmp7 = tmp4 * tmp6;
        auto tmp8 = (tmp7);
        auto tmp10 = cvt_lowp_fp_to_fp32<bfloat16>(tmp9);
        auto tmp11 = tmp8 * tmp10;
        tmp_acc0_vec = tmp_acc0_vec + tmp11;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{{float tmp_acc0 = 0;
at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
     x0 += static_cast<long>(8L)) {
  auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
      in_out_ptr0 + static_cast<long>(x0), 8);
  auto tmp2 =
      at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
  auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
  auto tmp3 = (tmp2);
  auto tmp4 = tmp1 + tmp3;
  auto tmp5 = (tmp4);
  auto tmp6 = tmp5 * tmp5;
  tmp_acc0_vec = tmp_acc0_vec + tmp6;
}
tmp_acc0 = tmp_acc0 +
    at::vec::vec_reduce_all<float>(
               [](at::vec::Vectorized<float>& x,
                  at::vec::Vectorized<float>& y) { return x + y; },
               tmp_acc0_vec);
out_ptr5[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(864L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            in_out_ptr0 + static_cast<long>(x1), 8);
        auto tmp2 =
            at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
        auto tmp6 = out_ptr5[static_cast<long>(0L)];
        auto tmp15 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr9 + static_cast<long>(x1), 8);
        auto tmp19 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr10 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = (tmp2);
        auto tmp4 = tmp1 + tmp3;
        auto tmp5 = (tmp4);
        auto tmp7 = static_cast<float>(288.0);
        auto tmp8 = tmp6 / tmp7;
        auto tmp9 = static_cast<float>(1e-05);
        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
        auto tmp11 = 1 / std::sqrt(tmp10);
        auto tmp12 = at::vec::Vectorized<float>(tmp11);
        auto tmp13 = tmp5 * tmp12;
        auto tmp14 = (tmp13);
        auto tmp16 = cvt_lowp_fp_to_fp32<bfloat16>(tmp15);
        auto tmp17 = tmp14 * tmp16;
        auto tmp18 = (tmp17);
        auto tmp20 = cvt_lowp_fp_to_fp32<bfloat16>(tmp19);
        auto tmp21 = tmp18 * tmp20;
        tmp_acc0_vec = tmp_acc0_vec + tmp21;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(24L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr6[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp2 = in_ptr11[static_cast<long>(0L)];
      auto tmp9 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr6[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp18 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr6[static_cast<long>(
              288L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp21 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr6[static_cast<long>(
              289L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp1 = (tmp0);
      auto tmp3 = decltype(tmp2)(tmp2 + 2048);
      auto tmp4 = tmp2 < 0;
      auto tmp5 = tmp4 ? tmp3 : tmp2;
      AOTI_TORCH_CHECK(
          (0 <= tmp5) & (tmp5 < 2048L),
          "index out of bounds: 0 <= tmp5 < 2048L")
      auto tmp6 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr12[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp7 = cvt_lowp_fp_to_fp32<bfloat16>(tmp6);
      auto tmp8 = tmp1 * tmp7;
      auto tmp10 = (tmp9);
      auto tmp11 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr12[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
      auto tmp13 = tmp10 * tmp12;
      auto tmp14 = tmp8 - tmp13;
      auto tmp15 = tmp10 * tmp7;
      auto tmp16 = tmp1 * tmp12;
      auto tmp17 = tmp15 + tmp16;
      auto tmp19 = (tmp18);
      auto tmp20 = tmp19 * tmp7;
      auto tmp22 = (tmp21);
      auto tmp23 = tmp22 * tmp12;
      auto tmp24 = tmp20 - tmp23;
      auto tmp25 = tmp22 * tmp7;
      auto tmp26 = tmp19 * tmp12;
      auto tmp27 = tmp25 + tmp26;
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp14.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr7[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp17.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr8[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp24.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr9[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp27.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr10[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr11[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          in_ptr13 + static_cast<long>(x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr11 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr11[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          out_ptr6 + static_cast<long>(576L + x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr11 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 =
        at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
    auto tmp1 = cvt_fp32_to_lowp_fp<bfloat16>(tmp0);
    tmp1.store(out_ptr12 + static_cast<long>(x0), 8);
  }
}
}

#include "cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void cpp_fused__to_copy_add_index_put_mean_mm_mul_rsqrt_stack_5(
    const bfloat16* in_ptr0,
    const bfloat16* in_ptr1,
    const bfloat16* in_ptr2,
    const float* in_ptr3,
    const bfloat16* in_ptr4,
    const bfloat16* in_ptr5,
    const bfloat16* in_ptr6,
    const bfloat16* in_ptr7,
    const bfloat16* in_ptr8,
    const bfloat16* in_ptr9,
    const long* in_ptr10,
    const bfloat16* in_ptr11,
    const float* in_ptr12,
    const float* in_ptr13,
    float* out_ptr0,
    float* out_ptr1,
    bfloat16* out_ptr2,
    float* out_ptr3,
    float* out_ptr4,
    float* out_ptr5,
    float* out_ptr6,
    float* out_ptr7,
    float* out_ptr8,
    float* out_ptr9,
    float* out_ptr10,
    float* out_ptr11,
    float* out_ptr12,
    bfloat16* out_ptr13,
    bfloat16* out_ptr14) {
  {
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
         x0 += static_cast<long>(1L)) {
      {
        float tmp_acc0 = 0;
        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
        for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
             x1 += static_cast<long>(8L)) {
          auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr0 + static_cast<long>(x1), 8);
          auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr1 + static_cast<long>(x1 + (288L * x0)), 8);
          auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
          auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
          auto tmp4 = tmp1 * tmp3;
          tmp_acc0_vec = tmp_acc0_vec + tmp4;
        }
        tmp_acc0 = tmp_acc0 +
            at::vec::vec_reduce_all<float>(
                       [](at::vec::Vectorized<float>& x,
                          at::vec::Vectorized<float>& y) { return x + y; },
                       tmp_acc0_vec);
        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      }
    }
  }
  {{float tmp_acc0 = 0;
  at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr2 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp8 = (tmp7);
    auto tmp9 = tmp8 * tmp8;
    tmp_acc0_vec = tmp_acc0_vec + tmp9;
  }
  tmp_acc0 = tmp_acc0 +
      at::vec::vec_reduce_all<float>(
                 [](at::vec::Vectorized<float>& x,
                    at::vec::Vectorized<float>& y) { return x + y; },
                 tmp_acc0_vec);
  out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr2 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp9 = out_ptr1[static_cast<long>(0L)];
    auto tmp18 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr4 + static_cast<long>(x0), 8);
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp8 = (tmp7);
    auto tmp10 = static_cast<float>(288.0);
    auto tmp11 = tmp9 / tmp10;
    auto tmp12 = static_cast<float>(1e-05);
    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
    auto tmp14 = 1 / std::sqrt(tmp13);
    auto tmp15 = at::vec::Vectorized<float>(tmp14);
    auto tmp16 = tmp8 * tmp15;
    auto tmp17 = (tmp16);
    auto tmp19 = cvt_lowp_fp_to_fp32<bfloat16>(tmp18);
    auto tmp20 = tmp17 * tmp19;
    auto tmp21 = cvt_fp32_to_lowp_fp<bfloat16>(tmp20);
    tmp21.store(out_ptr2 + static_cast<long>(x0), 8);
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(768L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      float tmp_acc1 = 0;
      at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            out_ptr2 + static_cast<long>(x1), 8);
        auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr5 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp5 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr6 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
        auto tmp4 = tmp1 * tmp3;
        auto tmp6 = cvt_lowp_fp_to_fp32<bfloat16>(tmp5);
        auto tmp7 = tmp1 * tmp6;
        tmp_acc0_vec = tmp_acc0_vec + tmp4;
        tmp_acc1_vec = tmp_acc1_vec + tmp7;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      tmp_acc1 = tmp_acc1 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc1_vec);
      out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(768L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
        auto tmp5 =
            at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
        auto tmp9 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr7 + static_cast<long>(x1 + (768L * x0)), 8);
        auto tmp1 = (tmp0);
        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + tmp1.neg().exp());
        auto tmp3 = tmp1 * tmp2;
        auto tmp4 = (tmp3);
        auto tmp6 = (tmp5);
        auto tmp7 = tmp4 * tmp6;
        auto tmp8 = (tmp7);
        auto tmp10 = cvt_lowp_fp_to_fp32<bfloat16>(tmp9);
        auto tmp11 = tmp8 * tmp10;
        tmp_acc0_vec = tmp_acc0_vec + tmp11;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{{float tmp_acc0 = 0;
at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
     x0 += static_cast<long>(8L)) {
  auto tmp0 =
      at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x0), 8);
  auto tmp2 =
      at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
  auto tmp5 =
      at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
  auto tmp8 =
      at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
  auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
  auto tmp3 = (tmp2);
  auto tmp4 = tmp1 + tmp3;
  auto tmp6 = (tmp5);
  auto tmp7 = tmp4 + tmp6;
  auto tmp9 = (tmp8);
  auto tmp10 = tmp7 + tmp9;
  auto tmp11 = (tmp10);
  auto tmp12 = tmp11 * tmp11;
  tmp_acc0_vec = tmp_acc0_vec + tmp12;
}
tmp_acc0 = tmp_acc0 +
    at::vec::vec_reduce_all<float>(
               [](at::vec::Vectorized<float>& x,
                  at::vec::Vectorized<float>& y) { return x + y; },
               tmp_acc0_vec);
out_ptr6[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr2 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp8 =
        at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
    auto tmp12 = out_ptr6[static_cast<long>(0L)];
    auto tmp21 = at::vec::Vectorized<bfloat16>::loadu(
        in_ptr8 + static_cast<long>(x0), 8);
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp9 = (tmp8);
    auto tmp10 = tmp7 + tmp9;
    auto tmp11 = (tmp10);
    auto tmp13 = static_cast<float>(288.0);
    auto tmp14 = tmp12 / tmp13;
    auto tmp15 = static_cast<float>(1e-05);
    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
    auto tmp17 = 1 / std::sqrt(tmp16);
    auto tmp18 = at::vec::Vectorized<float>(tmp17);
    auto tmp19 = tmp11 * tmp18;
    auto tmp20 = (tmp19);
    auto tmp22 = cvt_lowp_fp_to_fp32<bfloat16>(tmp21);
    auto tmp23 = tmp20 * tmp22;
    auto tmp24 = (tmp23);
    tmp24.store(out_ptr7 + static_cast<long>(x0));
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(864L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x1));
        auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr9 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp2 = cvt_lowp_fp_to_fp32<bfloat16>(tmp1);
        auto tmp3 = tmp0 * tmp2;
        tmp_acc0_vec = tmp_acc0_vec + tmp3;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr8[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(24L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp2 = in_ptr10[static_cast<long>(0L)];
      auto tmp9 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp18 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              288L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp21 = [&] {
        __at_align__ std::array<float, 8> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = out_ptr8[static_cast<long>(
              289L + (2L * x1) + (2L * x1_inner) + (48L * x0))];
        }
        return at::vec::Vectorized<float>::loadu(tmpbuf.data());
      }();
      auto tmp1 = (tmp0);
      auto tmp3 = decltype(tmp2)(tmp2 + 2048);
      auto tmp4 = tmp2 < 0;
      auto tmp5 = tmp4 ? tmp3 : tmp2;
      AOTI_TORCH_CHECK(
          (0 <= tmp5) & (tmp5 < 2048L),
          "index out of bounds: 0 <= tmp5 < 2048L")
      auto tmp6 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr11[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp7 = cvt_lowp_fp_to_fp32<bfloat16>(tmp6);
      auto tmp8 = tmp1 * tmp7;
      auto tmp10 = (tmp9);
      auto tmp11 = [&] {
        __at_align__ std::array<bfloat16, 16> tmpbuf;
#pragma unroll 8
        for (long x1_inner = 0; x1_inner < 8; x1_inner++) {
          tmpbuf[x1_inner] = in_ptr11[static_cast<long>(
              1L + (2L * x1) + (2L * x1_inner) + (48L * tmp5))];
        }
        return at::vec::Vectorized<bfloat16>::loadu(tmpbuf.data(), 8);
      }();
      auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
      auto tmp13 = tmp10 * tmp12;
      auto tmp14 = tmp8 - tmp13;
      auto tmp15 = tmp10 * tmp7;
      auto tmp16 = tmp1 * tmp12;
      auto tmp17 = tmp15 + tmp16;
      auto tmp19 = (tmp18);
      auto tmp20 = tmp19 * tmp7;
      auto tmp22 = (tmp21);
      auto tmp23 = tmp22 * tmp12;
      auto tmp24 = tmp20 - tmp23;
      auto tmp25 = tmp22 * tmp7;
      auto tmp26 = tmp19 * tmp12;
      auto tmp27 = tmp25 + tmp26;
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp14.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr9[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp17.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr10[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp24.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr11[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
      {
        __at_align__ float tmpbuf[8 * sizeof(float) / sizeof(float)];
        tmp27.store(tmpbuf);
        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
          out_ptr12[static_cast<long>(
              (2L * x1) + (2L * x1_inner) + (48L * x0))] = tmpbuf[x1_inner];
      }
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr10[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          in_ptr12 + static_cast<long>(x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr13 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(6L);
       x0 += static_cast<long>(1L)) {
    for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(48L);
         x1 += static_cast<long>(8L)) {
      auto tmp0 = in_ptr10[static_cast<long>(0L)];
      auto tmp4 = at::vec::Vectorized<float>::loadu(
          out_ptr8 + static_cast<long>(576L + x1 + (48L * x0)));
      auto tmp1 = decltype(tmp0)(tmp0 + 352);
      auto tmp2 = tmp0 < 0;
      auto tmp3 = tmp2 ? tmp1 : tmp0;
      AOTI_TORCH_CHECK(
          (0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
      auto tmp5 = cvt_fp32_to_lowp_fp<bfloat16>(tmp4);
      tmp5.store(
          out_ptr13 + static_cast<long>(x1 + (48L * tmp3) + (16896L * x0)), 8);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 =
        at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
    auto tmp1 = cvt_fp32_to_lowp_fp<bfloat16>(tmp0);
    tmp1.store(out_ptr14 + static_cast<long>(x0), 8);
  }
}
}

#include "cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void cpp_fused__to_copy_add_mean_mm_mul_6(
    bfloat16* in_out_ptr0,
    const bfloat16* in_ptr0,
    const bfloat16* in_ptr1,
    const float* in_ptr2,
    const float* in_ptr3,
    const float* in_ptr4,
    const bfloat16* in_ptr5,
    const bfloat16* in_ptr6,
    const bfloat16* in_ptr7,
    const bfloat16* in_ptr8,
    const bfloat16* in_ptr9,
    const bfloat16* in_ptr10,
    float* out_ptr0,
    float* out_ptr1,
    float* out_ptr2,
    float* out_ptr3,
    float* out_ptr4,
    float* out_ptr5,
    float* out_ptr6,
    bfloat16* out_ptr7) {
  {
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
         x0 += static_cast<long>(1L)) {
      {
        float tmp_acc0 = 0;
        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
        for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
             x1 += static_cast<long>(8L)) {
          auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr0 + static_cast<long>(x1), 8);
          auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr1 + static_cast<long>(x1 + (288L * x0)), 8);
          auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
          auto tmp3 = cvt_lowp_fp_to_fp32<bfloat16>(tmp2);
          auto tmp4 = tmp1 * tmp3;
          tmp_acc0_vec = tmp_acc0_vec + tmp4;
        }
        tmp_acc0 = tmp_acc0 +
            at::vec::vec_reduce_all<float>(
                       [](at::vec::Vectorized<float>& x,
                          at::vec::Vectorized<float>& y) { return x + y; },
                       tmp_acc0_vec);
        out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      }
    }
  }
  {{float tmp_acc0 = 0;
  at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(8L)) {
    auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
        in_out_ptr0 + static_cast<long>(x0), 8);
    auto tmp2 =
        at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
    auto tmp5 =
        at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
    auto tmp8 =
        at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
    auto tmp11 =
        at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
    auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
    auto tmp3 = (tmp2);
    auto tmp4 = tmp1 + tmp3;
    auto tmp6 = (tmp5);
    auto tmp7 = tmp4 + tmp6;
    auto tmp9 = (tmp8);
    auto tmp10 = tmp7 + tmp9;
    auto tmp12 = (tmp11);
    auto tmp13 = tmp10 + tmp12;
    auto tmp14 = cvt_fp32_to_lowp_fp<bfloat16>(tmp13);
    auto tmp15 = tmp13 * tmp13;
    tmp14.store(in_out_ptr0 + static_cast<long>(x0), 8);
    tmp_acc0_vec = tmp_acc0_vec + tmp15;
  }
  tmp_acc0 = tmp_acc0 +
      at::vec::vec_reduce_all<float>(
                 [](at::vec::Vectorized<float>& x,
                    at::vec::Vectorized<float>& y) { return x + y; },
                 tmp_acc0_vec);
  out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
}
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(768L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      float tmp_acc1 = 0;
      at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
            in_out_ptr0 + static_cast<long>(x1), 8);
        auto tmp2 = out_ptr1[static_cast<long>(0L)];
        auto tmp11 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr5 + static_cast<long>(x1), 8);
        auto tmp15 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr6 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp18 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr7 + static_cast<long>(x1 + (288L * x0)), 8);
        auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
        auto tmp3 = static_cast<float>(288.0);
        auto tmp4 = tmp2 / tmp3;
        auto tmp5 = static_cast<float>(1e-05);
        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
        auto tmp7 = 1 / std::sqrt(tmp6);
        auto tmp8 = at::vec::Vectorized<float>(tmp7);
        auto tmp9 = tmp1 * tmp8;
        auto tmp10 = (tmp9);
        auto tmp12 = cvt_lowp_fp_to_fp32<bfloat16>(tmp11);
        auto tmp13 = tmp10 * tmp12;
        auto tmp14 = (tmp13);
        auto tmp16 = cvt_lowp_fp_to_fp32<bfloat16>(tmp15);
        auto tmp17 = tmp14 * tmp16;
        auto tmp19 = cvt_lowp_fp_to_fp32<bfloat16>(tmp18);
        auto tmp20 = tmp14 * tmp19;
        tmp_acc0_vec = tmp_acc0_vec + tmp17;
        tmp_acc1_vec = tmp_acc1_vec + tmp20;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      tmp_acc1 = tmp_acc1 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc1_vec);
      out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
    }
  }
}
{
  for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
       x0 += static_cast<long>(1L)) {
    {
      float tmp_acc0 = 0;
      at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
      for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(768L);
           x1 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
        auto tmp5 =
            at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
        auto tmp9 = at::vec::Vectorized<bfloat16>::loadu(
            in_ptr8 + static_cast<long>(x1 + (768L * x0)), 8);
        auto tmp1 = (tmp0);
        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + tmp1.neg().exp());
        auto tmp3 = tmp1 * tmp2;
        auto tmp4 = (tmp3);
        auto tmp6 = (tmp5);
        auto tmp7 = tmp4 * tmp6;
        auto tmp8 = (tmp7);
        auto tmp10 = cvt_lowp_fp_to_fp32<bfloat16>(tmp9);
        auto tmp11 = tmp8 * tmp10;
        tmp_acc0_vec = tmp_acc0_vec + tmp11;
      }
      tmp_acc0 = tmp_acc0 +
          at::vec::vec_reduce_all<float>(
                     [](at::vec::Vectorized<float>& x,
                        at::vec::Vectorized<float>& y) { return x + y; },
                     tmp_acc0_vec);
      out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
    }
  }
}
{
  {
    float tmp_acc0 = 0;
    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(288L);
         x0 += static_cast<long>(8L)) {
      auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
          in_out_ptr0 + static_cast<long>(x0), 8);
      auto tmp2 =
          at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
      auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
      auto tmp3 = (tmp2);
      auto tmp4 = tmp1 + tmp3;
      auto tmp5 = (tmp4);
      auto tmp6 = tmp5 * tmp5;
      tmp_acc0_vec = tmp_acc0_vec + tmp6;
    }
    tmp_acc0 = tmp_acc0 +
        at::vec::vec_reduce_all<float>(
                   [](at::vec::Vectorized<float>& x,
                      at::vec::Vectorized<float>& y) { return x + y; },
                   tmp_acc0_vec);
    out_ptr5[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
  }
}
#pragma omp parallel num_threads(96)
{
  int tid = omp_get_thread_num();
  {
#pragma omp for
    for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(32000L);
         x0 += static_cast<long>(1L)) {
      {
        float tmp_acc0 = 0;
        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
        for (long x1 = static_cast<long>(0L); x1 < static_cast<long>(288L);
             x1 += static_cast<long>(8L)) {
          auto tmp0 = at::vec::Vectorized<bfloat16>::loadu(
              in_out_ptr0 + static_cast<long>(x1), 8);
          auto tmp2 = at::vec::Vectorized<float>::loadu(
              out_ptr4 + static_cast<long>(x1));
          auto tmp6 = out_ptr5[static_cast<long>(0L)];
          auto tmp15 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr9 + static_cast<long>(x1), 8);
          auto tmp19 = at::vec::Vectorized<bfloat16>::loadu(
              in_ptr10 + static_cast<long>(x1 + (288L * x0)), 8);
          auto tmp1 = cvt_lowp_fp_to_fp32<bfloat16>(tmp0);
          auto tmp3 = (tmp2);
          auto tmp4 = tmp1 + tmp3;
          auto tmp5 = (tmp4);
          auto tmp7 = static_cast<float>(288.0);
          auto tmp8 = tmp6 / tmp7;
          auto tmp9 = static_cast<float>(1e-05);
          auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
          auto tmp11 = 1 / std::sqrt(tmp10);
          auto tmp12 = at::vec::Vectorized<float>(tmp11);
          auto tmp13 = tmp5 * tmp12;
          auto tmp14 = (tmp13);
          auto tmp16 = cvt_lowp_fp_to_fp32<bfloat16>(tmp15);
          auto tmp17 = tmp14 * tmp16;
          auto tmp18 = (tmp17);
          auto tmp20 = cvt_lowp_fp_to_fp32<bfloat16>(tmp19);
          auto tmp21 = tmp18 * tmp20;
          tmp_acc0_vec = tmp_acc0_vec + tmp21;
        }
        tmp_acc0 = tmp_acc0 +
            at::vec::vec_reduce_all<float>(
                       [](at::vec::Vectorized<float>& x,
                          at::vec::Vectorized<float>& y) { return x + y; },
                       tmp_acc0_vec);
        out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
      }
    }
  }
#pragma omp single
  {
    {
      for (long x0 = static_cast<long>(0L); x0 < static_cast<long>(32000L);
           x0 += static_cast<long>(8L)) {
        auto tmp0 =
            at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x0));
        auto tmp1 = cvt_fp32_to_lowp_fp<bfloat16>(tmp0);
        tmp1.store(out_ptr7 + static_cast<long>(x0), 8);
      }
    }
  }
}
}
CACHE_TORCH_DTYPE(bfloat16);
CACHE_TORCH_DTYPE(bool);
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cpu);
namespace torch {
namespace aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
 public:
};
} // namespace

AOTInductorModel::AOTInductorModel(
    std::shared_ptr<ConstantMap> constants_map,
    std::shared_ptr<std::vector<ConstantHandle>> constants_array,
    const std::string& device_str,
    std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(2, 13, 47, device_str, cubin_dir) {
  inputs_info_[0].name = "arg59_1";
  inputs_info_[1].name = "arg60_1";
  constants_info_[0].name = "L__self___model_layers_0_attention_norm_weight";
  constants_info_[0].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[0].offset = 0;
  constants_info_[0].data_size = 576;
  constants_info_[0].from_folded = false;
  constants_info_[0].shape = {288};
  constants_info_[0].stride = {1};
  constants_info_[0].original_fqn =
      "L__self___model_layers_0_attention_norm_weight";
  constants_info_[1].name = "L__self___model_layers_0_ffn_norm_weight";
  constants_info_[1].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[1].offset = 0;
  constants_info_[1].data_size = 576;
  constants_info_[1].from_folded = false;
  constants_info_[1].shape = {288};
  constants_info_[1].stride = {1};
  constants_info_[1].original_fqn = "L__self___model_layers_0_ffn_norm_weight";
  constants_info_[2].name = "L__self___model_layers_1_attention_norm_weight";
  constants_info_[2].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[2].offset = 0;
  constants_info_[2].data_size = 576;
  constants_info_[2].from_folded = false;
  constants_info_[2].shape = {288};
  constants_info_[2].stride = {1};
  constants_info_[2].original_fqn =
      "L__self___model_layers_1_attention_norm_weight";
  constants_info_[3].name = "L__self___model_layers_1_ffn_norm_weight";
  constants_info_[3].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[3].offset = 0;
  constants_info_[3].data_size = 576;
  constants_info_[3].from_folded = false;
  constants_info_[3].shape = {288};
  constants_info_[3].stride = {1};
  constants_info_[3].original_fqn = "L__self___model_layers_1_ffn_norm_weight";
  constants_info_[4].name = "L__self___model_layers_2_attention_norm_weight";
  constants_info_[4].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[4].offset = 0;
  constants_info_[4].data_size = 576;
  constants_info_[4].from_folded = false;
  constants_info_[4].shape = {288};
  constants_info_[4].stride = {1};
  constants_info_[4].original_fqn =
      "L__self___model_layers_2_attention_norm_weight";
  constants_info_[5].name = "L__self___model_layers_2_ffn_norm_weight";
  constants_info_[5].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[5].offset = 0;
  constants_info_[5].data_size = 576;
  constants_info_[5].from_folded = false;
  constants_info_[5].shape = {288};
  constants_info_[5].stride = {1};
  constants_info_[5].original_fqn = "L__self___model_layers_2_ffn_norm_weight";
  constants_info_[6].name = "L__self___model_layers_3_attention_norm_weight";
  constants_info_[6].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[6].offset = 0;
  constants_info_[6].data_size = 576;
  constants_info_[6].from_folded = false;
  constants_info_[6].shape = {288};
  constants_info_[6].stride = {1};
  constants_info_[6].original_fqn =
      "L__self___model_layers_3_attention_norm_weight";
  constants_info_[7].name = "L__self___model_layers_3_ffn_norm_weight";
  constants_info_[7].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[7].offset = 0;
  constants_info_[7].data_size = 576;
  constants_info_[7].from_folded = false;
  constants_info_[7].shape = {288};
  constants_info_[7].stride = {1};
  constants_info_[7].original_fqn = "L__self___model_layers_3_ffn_norm_weight";
  constants_info_[8].name = "L__self___model_layers_4_attention_norm_weight";
  constants_info_[8].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[8].offset = 0;
  constants_info_[8].data_size = 576;
  constants_info_[8].from_folded = false;
  constants_info_[8].shape = {288};
  constants_info_[8].stride = {1};
  constants_info_[8].original_fqn =
      "L__self___model_layers_4_attention_norm_weight";
  constants_info_[9].name = "L__self___model_layers_4_ffn_norm_weight";
  constants_info_[9].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[9].offset = 0;
  constants_info_[9].data_size = 576;
  constants_info_[9].from_folded = false;
  constants_info_[9].shape = {288};
  constants_info_[9].stride = {1};
  constants_info_[9].original_fqn = "L__self___model_layers_4_ffn_norm_weight";
  constants_info_[10].name = "L__self___model_layers_5_attention_norm_weight";
  constants_info_[10].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[10].offset = 0;
  constants_info_[10].data_size = 576;
  constants_info_[10].from_folded = false;
  constants_info_[10].shape = {288};
  constants_info_[10].stride = {1};
  constants_info_[10].original_fqn =
      "L__self___model_layers_5_attention_norm_weight";
  constants_info_[11].name = "L__self___model_layers_5_ffn_norm_weight";
  constants_info_[11].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[11].offset = 0;
  constants_info_[11].data_size = 576;
  constants_info_[11].from_folded = false;
  constants_info_[11].shape = {288};
  constants_info_[11].stride = {1};
  constants_info_[11].original_fqn = "L__self___model_layers_5_ffn_norm_weight";
  constants_info_[12].name = "L__self___model_norm_weight";
  constants_info_[12].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[12].offset = 0;
  constants_info_[12].data_size = 576;
  constants_info_[12].from_folded = false;
  constants_info_[12].shape = {288};
  constants_info_[12].stride = {1};
  constants_info_[12].original_fqn = "model.norm.weight";
  constants_info_[13].name = "L__self___model_tok_embeddings_weight";
  constants_info_[13].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[13].offset = 0;
  constants_info_[13].data_size = 18432000;
  constants_info_[13].from_folded = false;
  constants_info_[13].shape = {32000, 288};
  constants_info_[13].stride = {288, 1};
  constants_info_[13].original_fqn = "model.tok_embeddings.weight";
  constants_info_[14].name = "L__self___model_layers_0_attention_wqkv_weight";
  constants_info_[14].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[14].offset = 0;
  constants_info_[14].data_size = 497664;
  constants_info_[14].from_folded = false;
  constants_info_[14].shape = {864, 288};
  constants_info_[14].stride = {288, 1};
  constants_info_[14].original_fqn =
      "L__self___model_layers_0_attention_wqkv.weight";
  constants_info_[15].name = "L__self___model_layers_0_attention_wo_weight";
  constants_info_[15].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[15].offset = 0;
  constants_info_[15].data_size = 165888;
  constants_info_[15].from_folded = false;
  constants_info_[15].shape = {288, 288};
  constants_info_[15].stride = {288, 1};
  constants_info_[15].original_fqn =
      "L__self___model_layers_0_attention_wo.weight";
  constants_info_[16].name = "L__self___model_layers_0_feed_forward_w1_weight";
  constants_info_[16].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[16].offset = 0;
  constants_info_[16].data_size = 442368;
  constants_info_[16].from_folded = false;
  constants_info_[16].shape = {768, 288};
  constants_info_[16].stride = {288, 1};
  constants_info_[16].original_fqn =
      "L__self___model_layers_0_feed_forward_w1.weight";
  constants_info_[17].name = "L__self___model_layers_0_feed_forward_w3_weight";
  constants_info_[17].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[17].offset = 0;
  constants_info_[17].data_size = 442368;
  constants_info_[17].from_folded = false;
  constants_info_[17].shape = {768, 288};
  constants_info_[17].stride = {288, 1};
  constants_info_[17].original_fqn =
      "L__self___model_layers_0_feed_forward_w3.weight";
  constants_info_[18].name = "L__self___model_layers_0_feed_forward_w2_weight";
  constants_info_[18].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[18].offset = 0;
  constants_info_[18].data_size = 442368;
  constants_info_[18].from_folded = false;
  constants_info_[18].shape = {288, 768};
  constants_info_[18].stride = {768, 1};
  constants_info_[18].original_fqn =
      "L__self___model_layers_0_feed_forward_w2.weight";
  constants_info_[19].name = "L__self___model_layers_1_attention_wqkv_weight";
  constants_info_[19].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[19].offset = 0;
  constants_info_[19].data_size = 497664;
  constants_info_[19].from_folded = false;
  constants_info_[19].shape = {864, 288};
  constants_info_[19].stride = {288, 1};
  constants_info_[19].original_fqn =
      "L__self___model_layers_1_attention_wqkv.weight";
  constants_info_[20].name = "L__self___model_layers_1_attention_wo_weight";
  constants_info_[20].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[20].offset = 0;
  constants_info_[20].data_size = 165888;
  constants_info_[20].from_folded = false;
  constants_info_[20].shape = {288, 288};
  constants_info_[20].stride = {288, 1};
  constants_info_[20].original_fqn =
      "L__self___model_layers_1_attention_wo.weight";
  constants_info_[21].name = "L__self___model_layers_1_feed_forward_w1_weight";
  constants_info_[21].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[21].offset = 0;
  constants_info_[21].data_size = 442368;
  constants_info_[21].from_folded = false;
  constants_info_[21].shape = {768, 288};
  constants_info_[21].stride = {288, 1};
  constants_info_[21].original_fqn =
      "L__self___model_layers_1_feed_forward_w1.weight";
  constants_info_[22].name = "L__self___model_layers_1_feed_forward_w3_weight";
  constants_info_[22].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[22].offset = 0;
  constants_info_[22].data_size = 442368;
  constants_info_[22].from_folded = false;
  constants_info_[22].shape = {768, 288};
  constants_info_[22].stride = {288, 1};
  constants_info_[22].original_fqn =
      "L__self___model_layers_1_feed_forward_w3.weight";
  constants_info_[23].name = "L__self___model_layers_1_feed_forward_w2_weight";
  constants_info_[23].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[23].offset = 0;
  constants_info_[23].data_size = 442368;
  constants_info_[23].from_folded = false;
  constants_info_[23].shape = {288, 768};
  constants_info_[23].stride = {768, 1};
  constants_info_[23].original_fqn =
      "L__self___model_layers_1_feed_forward_w2.weight";
  constants_info_[24].name = "L__self___model_layers_2_attention_wqkv_weight";
  constants_info_[24].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[24].offset = 0;
  constants_info_[24].data_size = 497664;
  constants_info_[24].from_folded = false;
  constants_info_[24].shape = {864, 288};
  constants_info_[24].stride = {288, 1};
  constants_info_[24].original_fqn =
      "L__self___model_layers_2_attention_wqkv.weight";
  constants_info_[25].name = "L__self___model_layers_2_attention_wo_weight";
  constants_info_[25].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[25].offset = 0;
  constants_info_[25].data_size = 165888;
  constants_info_[25].from_folded = false;
  constants_info_[25].shape = {288, 288};
  constants_info_[25].stride = {288, 1};
  constants_info_[25].original_fqn =
      "L__self___model_layers_2_attention_wo.weight";
  constants_info_[26].name = "L__self___model_layers_2_feed_forward_w1_weight";
  constants_info_[26].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[26].offset = 0;
  constants_info_[26].data_size = 442368;
  constants_info_[26].from_folded = false;
  constants_info_[26].shape = {768, 288};
  constants_info_[26].stride = {288, 1};
  constants_info_[26].original_fqn =
      "L__self___model_layers_2_feed_forward_w1.weight";
  constants_info_[27].name = "L__self___model_layers_2_feed_forward_w3_weight";
  constants_info_[27].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[27].offset = 0;
  constants_info_[27].data_size = 442368;
  constants_info_[27].from_folded = false;
  constants_info_[27].shape = {768, 288};
  constants_info_[27].stride = {288, 1};
  constants_info_[27].original_fqn =
      "L__self___model_layers_2_feed_forward_w3.weight";
  constants_info_[28].name = "L__self___model_layers_2_feed_forward_w2_weight";
  constants_info_[28].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[28].offset = 0;
  constants_info_[28].data_size = 442368;
  constants_info_[28].from_folded = false;
  constants_info_[28].shape = {288, 768};
  constants_info_[28].stride = {768, 1};
  constants_info_[28].original_fqn =
      "L__self___model_layers_2_feed_forward_w2.weight";
  constants_info_[29].name = "L__self___model_layers_3_attention_wqkv_weight";
  constants_info_[29].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[29].offset = 0;
  constants_info_[29].data_size = 497664;
  constants_info_[29].from_folded = false;
  constants_info_[29].shape = {864, 288};
  constants_info_[29].stride = {288, 1};
  constants_info_[29].original_fqn =
      "L__self___model_layers_3_attention_wqkv.weight";
  constants_info_[30].name = "L__self___model_layers_3_attention_wo_weight";
  constants_info_[30].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[30].offset = 0;
  constants_info_[30].data_size = 165888;
  constants_info_[30].from_folded = false;
  constants_info_[30].shape = {288, 288};
  constants_info_[30].stride = {288, 1};
  constants_info_[30].original_fqn =
      "L__self___model_layers_3_attention_wo.weight";
  constants_info_[31].name = "L__self___model_layers_3_feed_forward_w1_weight";
  constants_info_[31].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[31].offset = 0;
  constants_info_[31].data_size = 442368;
  constants_info_[31].from_folded = false;
  constants_info_[31].shape = {768, 288};
  constants_info_[31].stride = {288, 1};
  constants_info_[31].original_fqn =
      "L__self___model_layers_3_feed_forward_w1.weight";
  constants_info_[32].name = "L__self___model_layers_3_feed_forward_w3_weight";
  constants_info_[32].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[32].offset = 0;
  constants_info_[32].data_size = 442368;
  constants_info_[32].from_folded = false;
  constants_info_[32].shape = {768, 288};
  constants_info_[32].stride = {288, 1};
  constants_info_[32].original_fqn =
      "L__self___model_layers_3_feed_forward_w3.weight";
  constants_info_[33].name = "L__self___model_layers_3_feed_forward_w2_weight";
  constants_info_[33].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[33].offset = 0;
  constants_info_[33].data_size = 442368;
  constants_info_[33].from_folded = false;
  constants_info_[33].shape = {288, 768};
  constants_info_[33].stride = {768, 1};
  constants_info_[33].original_fqn =
      "L__self___model_layers_3_feed_forward_w2.weight";
  constants_info_[34].name = "L__self___model_layers_4_attention_wqkv_weight";
  constants_info_[34].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[34].offset = 0;
  constants_info_[34].data_size = 497664;
  constants_info_[34].from_folded = false;
  constants_info_[34].shape = {864, 288};
  constants_info_[34].stride = {288, 1};
  constants_info_[34].original_fqn =
      "L__self___model_layers_4_attention_wqkv.weight";
  constants_info_[35].name = "L__self___model_layers_4_attention_wo_weight";
  constants_info_[35].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[35].offset = 0;
  constants_info_[35].data_size = 165888;
  constants_info_[35].from_folded = false;
  constants_info_[35].shape = {288, 288};
  constants_info_[35].stride = {288, 1};
  constants_info_[35].original_fqn =
      "L__self___model_layers_4_attention_wo.weight";
  constants_info_[36].name = "L__self___model_layers_4_feed_forward_w1_weight";
  constants_info_[36].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[36].offset = 0;
  constants_info_[36].data_size = 442368;
  constants_info_[36].from_folded = false;
  constants_info_[36].shape = {768, 288};
  constants_info_[36].stride = {288, 1};
  constants_info_[36].original_fqn =
      "L__self___model_layers_4_feed_forward_w1.weight";
  constants_info_[37].name = "L__self___model_layers_4_feed_forward_w3_weight";
  constants_info_[37].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[37].offset = 0;
  constants_info_[37].data_size = 442368;
  constants_info_[37].from_folded = false;
  constants_info_[37].shape = {768, 288};
  constants_info_[37].stride = {288, 1};
  constants_info_[37].original_fqn =
      "L__self___model_layers_4_feed_forward_w3.weight";
  constants_info_[38].name = "L__self___model_layers_4_feed_forward_w2_weight";
  constants_info_[38].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[38].offset = 0;
  constants_info_[38].data_size = 442368;
  constants_info_[38].from_folded = false;
  constants_info_[38].shape = {288, 768};
  constants_info_[38].stride = {768, 1};
  constants_info_[38].original_fqn =
      "L__self___model_layers_4_feed_forward_w2.weight";
  constants_info_[39].name = "L__self___model_layers_5_attention_wqkv_weight";
  constants_info_[39].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[39].offset = 0;
  constants_info_[39].data_size = 497664;
  constants_info_[39].from_folded = false;
  constants_info_[39].shape = {864, 288};
  constants_info_[39].stride = {288, 1};
  constants_info_[39].original_fqn =
      "L__self___model_layers_5_attention_wqkv.weight";
  constants_info_[40].name = "L__self___model_layers_5_attention_wo_weight";
  constants_info_[40].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[40].offset = 0;
  constants_info_[40].data_size = 165888;
  constants_info_[40].from_folded = false;
  constants_info_[40].shape = {288, 288};
  constants_info_[40].stride = {288, 1};
  constants_info_[40].original_fqn =
      "L__self___model_layers_5_attention_wo.weight";
  constants_info_[41].name = "L__self___model_layers_5_feed_forward_w1_weight";
  constants_info_[41].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[41].offset = 0;
  constants_info_[41].data_size = 442368;
  constants_info_[41].from_folded = false;
  constants_info_[41].shape = {768, 288};
  constants_info_[41].stride = {288, 1};
  constants_info_[41].original_fqn =
      "L__self___model_layers_5_feed_forward_w1.weight";
  constants_info_[42].name = "L__self___model_layers_5_feed_forward_w3_weight";
  constants_info_[42].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[42].offset = 0;
  constants_info_[42].data_size = 442368;
  constants_info_[42].from_folded = false;
  constants_info_[42].shape = {768, 288};
  constants_info_[42].stride = {288, 1};
  constants_info_[42].original_fqn =
      "L__self___model_layers_5_feed_forward_w3.weight";
  constants_info_[43].name = "L__self___model_layers_5_feed_forward_w2_weight";
  constants_info_[43].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[43].offset = 0;
  constants_info_[43].data_size = 442368;
  constants_info_[43].from_folded = false;
  constants_info_[43].shape = {288, 768};
  constants_info_[43].stride = {768, 1};
  constants_info_[43].original_fqn =
      "L__self___model_layers_5_feed_forward_w2.weight";
  constants_info_[44].name = "L__self___model_freqs_cis";
  constants_info_[44].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[44].offset = 0;
  constants_info_[44].data_size = 196608;
  constants_info_[44].from_folded = false;
  constants_info_[44].shape = {2048, 24, 2};
  constants_info_[44].stride = {48, 2, 1};
  constants_info_[44].original_fqn = "model.freqs_cis";
  constants_info_[45].name = "L__self___model_causal_mask";
  constants_info_[45].dtype = static_cast<int32_t>(cached_torch_dtype_bool);
  constants_info_[45].offset = 0;
  constants_info_[45].data_size = 123904;
  constants_info_[45].from_folded = false;
  constants_info_[45].shape = {352, 352};
  constants_info_[45].stride = {352, 1};
  constants_info_[45].original_fqn = "model.causal_mask";
  constants_info_[46].name =
      "L__self___model_layers_0_attention_kv_cache_k_cache";
  constants_info_[46].dtype = static_cast<int32_t>(cached_torch_dtype_bfloat16);
  constants_info_[46].offset = 0;
  constants_info_[46].data_size = 202752;
  constants_info_[46].from_folded = false;
  constants_info_[46].shape = {1, 6, 352, 48};
  constants_info_[46].stride = {101376, 16896, 48, 1};
  constants_info_[46].original_fqn =
      "L__self___model_layers_5_attention_kv_cache_v_cache";
  update_constants_map(std::move(constants_map));
  update_constants_array(std::move(constants_array));
  in_spec_ =
      "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}]}, {\"type\": \"builtins.dict\", \"context\": \"[]\", \"children_spec\": []}]}]";
  out_spec_ = "[1, {\"type\": null, \"context\": null, \"children_spec\": []}]";
  outputs_info_[0].name = "output0";
  outputs_info_[1].name = "output1";
  outputs_info_[2].name = "output2";
  outputs_info_[3].name = "output3";
  outputs_info_[4].name = "output4";
  outputs_info_[5].name = "output5";
  outputs_info_[6].name = "output6";
  outputs_info_[7].name = "output7";
  outputs_info_[8].name = "output8";
  outputs_info_[9].name = "output9";
  outputs_info_[10].name = "output10";
  outputs_info_[11].name = "output11";
  outputs_info_[12].name = "output12";
  this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle>
AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization) {
  if (!initialization) {
    std::cerr
        << "[WARNING] Calling constant_folding in model, but compiled with config: "
        << "aot_inductor.use_runtime_constant_folding=False\n";
  }
  return {};
}

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor) {}

void AOTInductorModel::run_impl(
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor) {
  auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 2);
  auto arg59_1 = std::move(inputs[0]);
  auto arg60_1 = std::move(inputs[1]);
  auto L__self___model_layers_0_attention_norm_weight = constants_->at(0);
  auto L__self___model_layers_0_ffn_norm_weight = constants_->at(1);
  auto L__self___model_layers_1_attention_norm_weight = constants_->at(2);
  auto L__self___model_layers_1_ffn_norm_weight = constants_->at(3);
  auto L__self___model_layers_2_attention_norm_weight = constants_->at(4);
  auto L__self___model_layers_2_ffn_norm_weight = constants_->at(5);
  auto L__self___model_layers_3_attention_norm_weight = constants_->at(6);
  auto L__self___model_layers_3_ffn_norm_weight = constants_->at(7);
  auto L__self___model_layers_4_attention_norm_weight = constants_->at(8);
  auto L__self___model_layers_4_ffn_norm_weight = constants_->at(9);
  auto L__self___model_layers_5_attention_norm_weight = constants_->at(10);
  auto L__self___model_layers_5_ffn_norm_weight = constants_->at(11);
  auto L__self___model_norm_weight = constants_->at(12);
  auto L__self___model_tok_embeddings_weight = constants_->at(13);
  auto L__self___model_layers_0_attention_wqkv_weight = constants_->at(14);
  auto L__self___model_layers_0_attention_wo_weight = constants_->at(15);
  auto L__self___model_layers_0_feed_forward_w1_weight = constants_->at(16);
  auto L__self___model_layers_0_feed_forward_w3_weight = constants_->at(17);
  auto L__self___model_layers_0_feed_forward_w2_weight = constants_->at(18);
  auto L__self___model_layers_1_attention_wqkv_weight = constants_->at(19);
  auto L__self___model_layers_1_attention_wo_weight = constants_->at(20);
  auto L__self___model_layers_1_feed_forward_w1_weight = constants_->at(21);
  auto L__self___model_layers_1_feed_forward_w3_weight = constants_->at(22);
  auto L__self___model_layers_1_feed_forward_w2_weight = constants_->at(23);
  auto L__self___model_layers_2_attention_wqkv_weight = constants_->at(24);
  auto L__self___model_layers_2_attention_wo_weight = constants_->at(25);
  auto L__self___model_layers_2_feed_forward_w1_weight = constants_->at(26);
  auto L__self___model_layers_2_feed_forward_w3_weight = constants_->at(27);
  auto L__self___model_layers_2_feed_forward_w2_weight = constants_->at(28);
  auto L__self___model_layers_3_attention_wqkv_weight = constants_->at(29);
  auto L__self___model_layers_3_attention_wo_weight = constants_->at(30);
  auto L__self___model_layers_3_feed_forward_w1_weight = constants_->at(31);
  auto L__self___model_layers_3_feed_forward_w3_weight = constants_->at(32);
  auto L__self___model_layers_3_feed_forward_w2_weight = constants_->at(33);
  auto L__self___model_layers_4_attention_wqkv_weight = constants_->at(34);
  auto L__self___model_layers_4_attention_wo_weight = constants_->at(35);
  auto L__self___model_layers_4_feed_forward_w1_weight = constants_->at(36);
  auto L__self___model_layers_4_feed_forward_w3_weight = constants_->at(37);
  auto L__self___model_layers_4_feed_forward_w2_weight = constants_->at(38);
  auto L__self___model_layers_5_attention_wqkv_weight = constants_->at(39);
  auto L__self___model_layers_5_attention_wo_weight = constants_->at(40);
  auto L__self___model_layers_5_feed_forward_w1_weight = constants_->at(41);
  auto L__self___model_layers_5_feed_forward_w3_weight = constants_->at(42);
  auto L__self___model_layers_5_feed_forward_w2_weight = constants_->at(43);
  auto L__self___model_freqs_cis = constants_->at(44);
  auto L__self___model_causal_mask = constants_->at(45);
  auto L__self___model_layers_0_attention_kv_cache_k_cache = constants_->at(46);
  inputs.clear();
  auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
  static constexpr int64_t int_array_6[] = {1L, 1L, 1L};
  AtenTensorHandle buf0_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      3,
      int_array_6,
      int_array_6,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf0_handle));
  RAIIAtenTensorHandle buf0(buf0_handle);
  static constexpr int64_t int_array_7[] = {1L, 864L};
  static constexpr int64_t int_array_8[] = {864L, 1L};
  AtenTensorHandle buf1_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      2,
      int_array_7,
      int_array_8,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf1_handle));
  RAIIAtenTensorHandle buf1(buf1_handle);
  static constexpr int64_t int_array_9[] = {1L, 1L, 6L, 24L, 2L};
  static constexpr int64_t int_array_10[] = {288L, 288L, 48L, 2L, 1L};
  AtenTensorHandle buf4_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      5,
      int_array_9,
      int_array_10,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf4_handle));
  RAIIAtenTensorHandle buf4(buf4_handle);
  static constexpr int64_t int_array_0[] = {1L, 1L, 6L, 24L, 1L};
  static constexpr int64_t int_array_1[] = {288L, 288L, 48L, 2L, 1L};
  auto tmp_tensor_handle_0 =
      reinterpret_tensor_wrapper(buf4, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf2 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_0); // alias
  auto tmp_tensor_handle_1 =
      reinterpret_tensor_wrapper(buf4, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf3 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_1); // alias
  AtenTensorHandle buf7_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      5,
      int_array_9,
      int_array_10,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf7_handle));
  RAIIAtenTensorHandle buf7(buf7_handle);
  auto tmp_tensor_handle_2 =
      reinterpret_tensor_wrapper(buf7, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf5 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_2); // alias
  auto tmp_tensor_handle_3 =
      reinterpret_tensor_wrapper(buf7, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf6 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_3); // alias
  static constexpr int64_t int_array_11[] = {1L, 1L, 6L, 48L};
  static constexpr int64_t int_array_12[] = {288L, 288L, 48L, 1L};
  AtenTensorHandle buf10_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      4,
      int_array_11,
      int_array_12,
      cached_torch_dtype_bfloat16,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf10_handle));
  RAIIAtenTensorHandle buf10(buf10_handle);
  static constexpr int64_t int_array_13[] = {1L, 1L, 1L, 352L};
  static constexpr int64_t int_array_14[] = {352L, 352L, 352L, 1L};
  AtenTensorHandle buf11_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      4,
      int_array_13,
      int_array_14,
      cached_torch_dtype_bfloat16,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf11_handle));
  RAIIAtenTensorHandle buf11(buf11_handle);
  AtenTensorHandle buf33_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      4,
      int_array_13,
      int_array_14,
      cached_torch_dtype_bfloat16,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf33_handle));
  RAIIAtenTensorHandle buf33(buf33_handle);
  AtenTensorHandle buf54_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      4,
      int_array_13,
      int_array_14,
      cached_torch_dtype_bfloat16,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf54_handle));
  RAIIAtenTensorHandle buf54(buf54_handle);
  auto* var_0 = get_data_ptr_wrapper(arg59_1);
  auto* var_1 = get_data_ptr_wrapper(L__self___model_tok_embeddings_weight);
  auto* var_2 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_norm_weight);
  auto* var_3 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_wqkv_weight);
  auto* var_4 = get_data_ptr_wrapper(arg60_1);
  auto* var_5 = get_data_ptr_wrapper(L__self___model_freqs_cis);
  auto* var_6 = get_data_ptr_wrapper(buf7);
  auto* var_7 = get_data_ptr_wrapper(buf4);
  auto* var_8 = get_data_ptr_wrapper(L__self___model_causal_mask);
  auto* var_9 = get_data_ptr_wrapper(buf0);
  auto* var_10 = get_data_ptr_wrapper(buf1);
  auto* var_11 = get_data_ptr_wrapper(buf2);
  auto* var_12 = get_data_ptr_wrapper(buf3);
  auto* var_13 = get_data_ptr_wrapper(buf5);
  auto* var_14 = get_data_ptr_wrapper(buf6);
  auto* var_15 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_kv_cache_k_cache);
  auto* var_16 = get_data_ptr_wrapper(buf10);
  auto* var_17 = get_data_ptr_wrapper(buf11);
  auto* var_18 = get_data_ptr_wrapper(buf33);
  auto* var_19 = get_data_ptr_wrapper(buf54);
  cpp_fused__to_copy_embedding_index_index_put_logical_not_masked_fill_mean_mm_mul_stack_zeros_like_0(
      (const long*)(var_0),
      (const bfloat16*)(var_1),
      (const bfloat16*)(var_2),
      (const bfloat16*)(var_3),
      (const long*)(var_4),
      (const bfloat16*)(var_5),
      (const float*)(var_6),
      (const float*)(var_7),
      (const bool*)(var_8),
      (float*)(var_9),
      (float*)(var_10),
      (float*)(var_11),
      (float*)(var_12),
      (float*)(var_13),
      (float*)(var_14),
      (bfloat16*)(var_15),
      (bfloat16*)(var_16),
      (bfloat16*)(var_17),
      (bfloat16*)(var_18),
      (bfloat16*)(var_19));
  buf2.reset();
  buf3.reset();
  buf5.reset();
  buf6.reset();
  // Source Nodes: [mask, y], Original ATen:
  // [aten._scaled_dot_product_flash_attention_for_cpu, aten.index,
  // aten.logical_not, aten.masked_fill, aten.zeros_like]
  static constexpr int64_t int_array_2[] = {1L, 6L, 1L, 48L};
  static constexpr int64_t int_array_3[] = {0L, 48L, 0L, 1L};
  auto tmp_tensor_handle_4 =
      reinterpret_tensor_wrapper(buf10, 4, int_array_2, int_array_3, 0L);
  static constexpr int64_t int_array_4[] = {1L, 6L, 352L, 48L};
  static constexpr int64_t int_array_5[] = {101376L, 16896L, 48L, 1L};
  auto tmp_tensor_handle_5 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  auto tmp_tensor_handle_6 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  AtenTensorHandle buf13_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf13_handle));
  RAIIAtenTensorHandle buf13(buf13_handle);
  AtenTensorHandle buf14_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf14_handle));
  RAIIAtenTensorHandle buf14(buf14_handle);
  aoti_torch_proxy_executor_call_function(
      proxy_executor,
      0,
      0,
      std::vector<int64_t>{}.data(),
      6,
      std::vector<AtenTensorHandle>{
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_4),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_5),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_6),
          buf11,
          buf13,
          buf14}
          .data());
  static constexpr int64_t int_array_15[] = {1L, 288L};
  static constexpr int64_t int_array_16[] = {288L, 1L};
  auto tmp_tensor_handle_42 =
      reinterpret_tensor_wrapper(buf7, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf15 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_42);
  buf7.reset(); // reuse
  auto buf16 = std::move(buf0); // reuse
  static constexpr int64_t int_array_17[] = {1L, 1L, 288L};
  static constexpr int64_t int_array_18[] = {288L, 288L, 1L};
  auto tmp_tensor_handle_43 =
      reinterpret_tensor_wrapper(buf10, 3, int_array_17, int_array_18, 0L);
  decltype(auto) buf17 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_43);
  buf10.reset(); // reuse
  static constexpr int64_t int_array_19[] = {1L, 768L};
  static constexpr int64_t int_array_20[] = {768L, 1L};
  AtenTensorHandle buf18_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      2,
      int_array_19,
      int_array_20,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf18_handle));
  RAIIAtenTensorHandle buf18(buf18_handle);
  AtenTensorHandle buf19_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      2,
      int_array_19,
      int_array_20,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf19_handle));
  RAIIAtenTensorHandle buf19(buf19_handle);
  auto tmp_tensor_handle_44 =
      reinterpret_tensor_wrapper(buf4, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf20 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_44);
  buf4.reset(); // reuse
  AtenTensorHandle buf21_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      3,
      int_array_6,
      int_array_6,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf21_handle));
  RAIIAtenTensorHandle buf21(buf21_handle);
  AtenTensorHandle buf22_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      2,
      int_array_15,
      int_array_16,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf22_handle));
  RAIIAtenTensorHandle buf22(buf22_handle);
  auto buf23 = std::move(buf1); // reuse
  AtenTensorHandle buf26_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      5,
      int_array_9,
      int_array_10,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf26_handle));
  RAIIAtenTensorHandle buf26(buf26_handle);
  auto tmp_tensor_handle_7 =
      reinterpret_tensor_wrapper(buf26, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf24 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_7); // alias
  auto tmp_tensor_handle_8 =
      reinterpret_tensor_wrapper(buf26, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf25 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_8); // alias
  AtenTensorHandle buf29_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      5,
      int_array_9,
      int_array_10,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf29_handle));
  RAIIAtenTensorHandle buf29(buf29_handle);
  auto tmp_tensor_handle_9 =
      reinterpret_tensor_wrapper(buf29, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf27 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_9); // alias
  auto tmp_tensor_handle_10 =
      reinterpret_tensor_wrapper(buf29, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf28 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_10); // alias
  AtenTensorHandle buf32_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      4,
      int_array_11,
      int_array_12,
      cached_torch_dtype_bfloat16,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf32_handle));
  RAIIAtenTensorHandle buf32(buf32_handle);
  auto* var_20 = get_data_ptr_wrapper(buf13);
  auto* var_21 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_wo_weight);
  auto* var_22 = get_data_ptr_wrapper(arg59_1);
  auto* var_23 = get_data_ptr_wrapper(L__self___model_tok_embeddings_weight);
  auto* var_24 = get_data_ptr_wrapper(L__self___model_layers_0_ffn_norm_weight);
  auto* var_25 =
      get_data_ptr_wrapper(L__self___model_layers_0_feed_forward_w1_weight);
  auto* var_26 =
      get_data_ptr_wrapper(L__self___model_layers_0_feed_forward_w3_weight);
  auto* var_27 =
      get_data_ptr_wrapper(L__self___model_layers_0_feed_forward_w2_weight);
  auto* var_28 =
      get_data_ptr_wrapper(L__self___model_layers_1_attention_norm_weight);
  auto* var_29 =
      get_data_ptr_wrapper(L__self___model_layers_1_attention_wqkv_weight);
  auto* var_30 = get_data_ptr_wrapper(arg60_1);
  auto* var_31 = get_data_ptr_wrapper(L__self___model_freqs_cis);
  auto* var_32 = get_data_ptr_wrapper(buf29);
  auto* var_33 = get_data_ptr_wrapper(buf26);
  auto* var_34 = get_data_ptr_wrapper(buf15);
  auto* var_35 = get_data_ptr_wrapper(buf16);
  auto* var_36 = get_data_ptr_wrapper(buf17);
  auto* var_37 = get_data_ptr_wrapper(buf18);
  auto* var_38 = get_data_ptr_wrapper(buf19);
  auto* var_39 = get_data_ptr_wrapper(buf20);
  auto* var_40 = get_data_ptr_wrapper(buf21);
  auto* var_41 = get_data_ptr_wrapper(buf22);
  auto* var_42 = get_data_ptr_wrapper(buf23);
  auto* var_43 = get_data_ptr_wrapper(buf24);
  auto* var_44 = get_data_ptr_wrapper(buf25);
  auto* var_45 = get_data_ptr_wrapper(buf27);
  auto* var_46 = get_data_ptr_wrapper(buf28);
  auto* var_47 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_kv_cache_k_cache);
  auto* var_48 = get_data_ptr_wrapper(buf32);
  cpp_fused__to_copy_add_embedding_index_put_mean_mm_mul_rsqrt_stack_1(
      (const bfloat16*)(var_20),
      (const bfloat16*)(var_21),
      (const long*)(var_22),
      (const bfloat16*)(var_23),
      (const bfloat16*)(var_24),
      (const bfloat16*)(var_25),
      (const bfloat16*)(var_26),
      (const bfloat16*)(var_27),
      (const bfloat16*)(var_28),
      (const bfloat16*)(var_29),
      (const long*)(var_30),
      (const bfloat16*)(var_31),
      (const float*)(var_32),
      (const float*)(var_33),
      (float*)(var_34),
      (float*)(var_35),
      (bfloat16*)(var_36),
      (float*)(var_37),
      (float*)(var_38),
      (float*)(var_39),
      (float*)(var_40),
      (float*)(var_41),
      (float*)(var_42),
      (float*)(var_43),
      (float*)(var_44),
      (float*)(var_45),
      (float*)(var_46),
      (bfloat16*)(var_47),
      (bfloat16*)(var_48));

  buf13.reset();
  buf24.reset();
  buf25.reset();
  buf27.reset();
  buf28.reset();
  // Source Nodes: [mask, y_3], Original ATen:
  // [aten._scaled_dot_product_flash_attention_for_cpu, aten.index,
  // aten.logical_not, aten.masked_fill, aten.zeros_like]
  auto tmp_tensor_handle_11 =
      reinterpret_tensor_wrapper(buf32, 4, int_array_2, int_array_3, 0L);
  auto tmp_tensor_handle_12 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  auto tmp_tensor_handle_13 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  AtenTensorHandle buf35_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf35_handle));
  RAIIAtenTensorHandle buf35(buf35_handle);
  AtenTensorHandle buf36_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf36_handle));
  RAIIAtenTensorHandle buf36(buf36_handle);
  aoti_torch_proxy_executor_call_function(
      proxy_executor,
      1,
      0,
      std::vector<int64_t>{}.data(),
      6,
      std::vector<AtenTensorHandle>{
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_11),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_12),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_13),
          buf33,
          buf35,
          buf36}
          .data());
  auto tmp_tensor_handle_45 =
      reinterpret_tensor_wrapper(buf29, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf37 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_45);
  buf29.reset(); // reuse
  auto tmp_tensor_handle_46 =
      reinterpret_tensor_wrapper(buf32, 3, int_array_17, int_array_18, 0L);
  decltype(auto) buf38 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_46);
  buf32.reset(); // reuse
  auto buf39 = std::move(buf21); // reuse
  auto buf40 = std::move(buf19); // reuse
  auto buf41 = std::move(buf18); // reuse
  auto tmp_tensor_handle_47 =
      reinterpret_tensor_wrapper(buf26, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf42 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_47);
  buf26.reset(); // reuse
  auto buf43 = std::move(buf16); // reuse
  auto buf44 = std::move(buf23); // reuse
  auto tmp_tensor_handle_48 =
      reinterpret_tensor_wrapper(buf22, 5, int_array_9, int_array_10, 0L);
  decltype(auto) buf47 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_48);
  buf22.reset(); // reuse
  auto tmp_tensor_handle_14 =
      reinterpret_tensor_wrapper(buf47, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf45 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_14); // alias
  auto tmp_tensor_handle_15 =
      reinterpret_tensor_wrapper(buf47, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf46 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_15); // alias
  AtenTensorHandle buf50_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      5,
      int_array_9,
      int_array_10,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf50_handle));
  RAIIAtenTensorHandle buf50(buf50_handle);
  auto tmp_tensor_handle_16 =
      reinterpret_tensor_wrapper(buf50, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf48 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_16); // alias
  auto tmp_tensor_handle_17 =
      reinterpret_tensor_wrapper(buf50, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf49 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_17); // alias
  auto tmp_tensor_handle_49 =
      reinterpret_tensor_wrapper(buf17, 4, int_array_11, int_array_12, 0L);
  decltype(auto) buf53 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_49);
  buf17.reset(); // reuse
  auto* var_49 = get_data_ptr_wrapper(buf35);
  auto* var_50 =
      get_data_ptr_wrapper(L__self___model_layers_1_attention_wo_weight);
  auto* var_51 = get_data_ptr_wrapper(arg59_1);
  auto* var_52 = get_data_ptr_wrapper(L__self___model_tok_embeddings_weight);
  auto* var_53 = get_data_ptr_wrapper(buf15);
  auto* var_54 = get_data_ptr_wrapper(buf20);
  auto* var_55 = get_data_ptr_wrapper(L__self___model_layers_1_ffn_norm_weight);
  auto* var_56 =
      get_data_ptr_wrapper(L__self___model_layers_1_feed_forward_w1_weight);
  auto* var_57 =
      get_data_ptr_wrapper(L__self___model_layers_1_feed_forward_w3_weight);
  auto* var_58 =
      get_data_ptr_wrapper(L__self___model_layers_1_feed_forward_w2_weight);
  auto* var_59 =
      get_data_ptr_wrapper(L__self___model_layers_2_attention_norm_weight);
  auto* var_60 =
      get_data_ptr_wrapper(L__self___model_layers_2_attention_wqkv_weight);
  auto* var_61 = get_data_ptr_wrapper(arg60_1);
  auto* var_62 = get_data_ptr_wrapper(L__self___model_freqs_cis);
  auto* var_63 = get_data_ptr_wrapper(buf50);
  auto* var_64 = get_data_ptr_wrapper(buf47);
  auto* var_65 = get_data_ptr_wrapper(buf37);
  auto* var_66 = get_data_ptr_wrapper(buf38);
  auto* var_67 = get_data_ptr_wrapper(buf39);
  auto* var_68 = get_data_ptr_wrapper(buf40);
  auto* var_69 = get_data_ptr_wrapper(buf41);
  auto* var_70 = get_data_ptr_wrapper(buf42);
  auto* var_71 = get_data_ptr_wrapper(buf43);
  auto* var_72 = get_data_ptr_wrapper(buf44);
  auto* var_73 = get_data_ptr_wrapper(buf45);
  auto* var_74 = get_data_ptr_wrapper(buf46);
  auto* var_75 = get_data_ptr_wrapper(buf48);
  auto* var_76 = get_data_ptr_wrapper(buf49);
  auto* var_77 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_kv_cache_k_cache);
  auto* var_78 = get_data_ptr_wrapper(buf53);
  cpp_fused__to_copy_add_embedding_index_put_mean_mm_mul_stack_2(
      (const bfloat16*)(var_49),
      (const bfloat16*)(var_50),
      (const long*)(var_51),
      (const bfloat16*)(var_52),
      (const float*)(var_53),
      (const float*)(var_54),
      (const bfloat16*)(var_55),
      (const bfloat16*)(var_56),
      (const bfloat16*)(var_57),
      (const bfloat16*)(var_58),
      (const bfloat16*)(var_59),
      (const bfloat16*)(var_60),
      (const long*)(var_61),
      (const bfloat16*)(var_62),
      (const float*)(var_63),
      (const float*)(var_64),
      (float*)(var_65),
      (bfloat16*)(var_66),
      (float*)(var_67),
      (float*)(var_68),
      (float*)(var_69),
      (float*)(var_70),
      (float*)(var_71),
      (float*)(var_72),
      (float*)(var_73),
      (float*)(var_74),
      (float*)(var_75),
      (float*)(var_76),
      (bfloat16*)(var_77),
      (bfloat16*)(var_78));
  arg59_1.reset();

  buf45.reset();
  buf46.reset();
  buf48.reset();
  buf49.reset();
  // Source Nodes: [mask, y_6], Original ATen:
  // [aten._scaled_dot_product_flash_attention_for_cpu, aten.index,
  // aten.logical_not, aten.masked_fill, aten.zeros_like]
  auto tmp_tensor_handle_18 =
      reinterpret_tensor_wrapper(buf53, 4, int_array_2, int_array_3, 0L);
  auto tmp_tensor_handle_19 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  auto tmp_tensor_handle_20 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  AtenTensorHandle buf56_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf56_handle));
  RAIIAtenTensorHandle buf56(buf56_handle);
  AtenTensorHandle buf57_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf57_handle));
  RAIIAtenTensorHandle buf57(buf57_handle);
  aoti_torch_proxy_executor_call_function(
      proxy_executor,
      2,
      0,
      std::vector<int64_t>{}.data(),
      6,
      std::vector<AtenTensorHandle>{
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_18),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_19),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_20),
          buf54,
          buf56,
          buf57}
          .data());
  auto tmp_tensor_handle_50 =
      reinterpret_tensor_wrapper(buf50, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf58 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_50);
  buf50.reset(); // reuse
  auto buf59 = std::move(buf43); // reuse
  auto tmp_tensor_handle_51 =
      reinterpret_tensor_wrapper(buf53, 3, int_array_17, int_array_18, 0L);
  decltype(auto) buf60 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_51);
  buf53.reset(); // reuse
  auto buf61 = std::move(buf41); // reuse
  auto buf62 = std::move(buf40); // reuse
  auto tmp_tensor_handle_52 =
      reinterpret_tensor_wrapper(buf47, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf63 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_52);
  buf47.reset(); // reuse
  auto buf64 = std::move(buf39); // reuse
  auto buf65 = std::move(buf37); // reuse
  auto buf66 = std::move(buf44); // reuse
  auto tmp_tensor_handle_53 =
      reinterpret_tensor_wrapper(buf20, 5, int_array_9, int_array_10, 0L);
  decltype(auto) buf69 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_53);
  buf20.reset(); // reuse
  auto tmp_tensor_handle_21 =
      reinterpret_tensor_wrapper(buf69, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf67 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_21); // alias
  auto tmp_tensor_handle_22 =
      reinterpret_tensor_wrapper(buf69, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf68 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_22); // alias
  auto tmp_tensor_handle_54 =
      reinterpret_tensor_wrapper(buf15, 5, int_array_9, int_array_10, 0L);
  decltype(auto) buf72 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_54);
  buf15.reset(); // reuse
  auto tmp_tensor_handle_23 =
      reinterpret_tensor_wrapper(buf72, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf70 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_23); // alias
  auto tmp_tensor_handle_24 =
      reinterpret_tensor_wrapper(buf72, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf71 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_24); // alias
  auto tmp_tensor_handle_55 =
      reinterpret_tensor_wrapper(buf35, 4, int_array_11, int_array_12, 0L);
  decltype(auto) buf75 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_55);
  buf35.reset(); // reuse
  auto buf76 = std::move(buf54); // reuse
  auto buf97 = std::move(buf33); // reuse
  auto buf119 = std::move(buf11); // reuse
  auto* var_79 = get_data_ptr_wrapper(buf56);
  auto* var_80 =
      get_data_ptr_wrapper(L__self___model_layers_2_attention_wo_weight);
  auto* var_81 = get_data_ptr_wrapper(buf38);
  auto* var_82 = get_data_ptr_wrapper(buf42);
  auto* var_83 = get_data_ptr_wrapper(L__self___model_layers_2_ffn_norm_weight);
  auto* var_84 =
      get_data_ptr_wrapper(L__self___model_layers_2_feed_forward_w1_weight);
  auto* var_85 =
      get_data_ptr_wrapper(L__self___model_layers_2_feed_forward_w3_weight);
  auto* var_86 =
      get_data_ptr_wrapper(L__self___model_layers_2_feed_forward_w2_weight);
  auto* var_87 =
      get_data_ptr_wrapper(L__self___model_layers_3_attention_norm_weight);
  auto* var_88 =
      get_data_ptr_wrapper(L__self___model_layers_3_attention_wqkv_weight);
  auto* var_89 = get_data_ptr_wrapper(arg60_1);
  auto* var_90 = get_data_ptr_wrapper(L__self___model_freqs_cis);
  auto* var_91 = get_data_ptr_wrapper(buf72);
  auto* var_92 = get_data_ptr_wrapper(buf69);
  auto* var_93 = get_data_ptr_wrapper(L__self___model_causal_mask);
  auto* var_94 = get_data_ptr_wrapper(buf58);
  auto* var_95 = get_data_ptr_wrapper(buf59);
  auto* var_96 = get_data_ptr_wrapper(buf60);
  auto* var_97 = get_data_ptr_wrapper(buf61);
  auto* var_98 = get_data_ptr_wrapper(buf62);
  auto* var_99 = get_data_ptr_wrapper(buf63);
  auto* var_100 = get_data_ptr_wrapper(buf64);
  auto* var_101 = get_data_ptr_wrapper(buf65);
  auto* var_102 = get_data_ptr_wrapper(buf66);
  auto* var_103 = get_data_ptr_wrapper(buf67);
  auto* var_104 = get_data_ptr_wrapper(buf68);
  auto* var_105 = get_data_ptr_wrapper(buf70);
  auto* var_106 = get_data_ptr_wrapper(buf71);
  auto* var_107 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_kv_cache_k_cache);
  auto* var_108 = get_data_ptr_wrapper(buf75);
  auto* var_109 = get_data_ptr_wrapper(buf76);
  auto* var_110 = get_data_ptr_wrapper(buf97);
  auto* var_111 = get_data_ptr_wrapper(buf119);
  cpp_fused__to_copy_add_index_index_put_logical_not_masked_fill_mean_mm_mul_rsqrt_stack_zeros_like_3(
      (const bfloat16*)(var_79),
      (const bfloat16*)(var_80),
      (const bfloat16*)(var_81),
      (const float*)(var_82),
      (const bfloat16*)(var_83),
      (const bfloat16*)(var_84),
      (const bfloat16*)(var_85),
      (const bfloat16*)(var_86),
      (const bfloat16*)(var_87),
      (const bfloat16*)(var_88),
      (const long*)(var_89),
      (const bfloat16*)(var_90),
      (const float*)(var_91),
      (const float*)(var_92),
      (const bool*)(var_93),
      (float*)(var_94),
      (float*)(var_95),
      (bfloat16*)(var_96),
      (float*)(var_97),
      (float*)(var_98),
      (float*)(var_99),
      (float*)(var_100),
      (float*)(var_101),
      (float*)(var_102),
      (float*)(var_103),
      (float*)(var_104),
      (float*)(var_105),
      (float*)(var_106),
      (bfloat16*)(var_107),
      (bfloat16*)(var_108),
      (bfloat16*)(var_109),
      (bfloat16*)(var_110),
      (bfloat16*)(var_111));

  buf56.reset();
  buf60.reset();
  buf67.reset();
  buf68.reset();
  buf70.reset();
  buf71.reset();
  // Source Nodes: [mask, y_9], Original ATen:
  // [aten._scaled_dot_product_flash_attention_for_cpu, aten.index,
  // aten.logical_not, aten.masked_fill, aten.zeros_like]
  auto tmp_tensor_handle_25 =
      reinterpret_tensor_wrapper(buf75, 4, int_array_2, int_array_3, 0L);
  auto tmp_tensor_handle_26 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  auto tmp_tensor_handle_27 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  AtenTensorHandle buf78_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf78_handle));
  RAIIAtenTensorHandle buf78(buf78_handle);
  AtenTensorHandle buf79_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf79_handle));
  RAIIAtenTensorHandle buf79(buf79_handle);
  aoti_torch_proxy_executor_call_function(
      proxy_executor,
      3,
      0,
      std::vector<int64_t>{}.data(),
      6,
      std::vector<AtenTensorHandle>{
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_25),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_26),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_27),
          buf76,
          buf78,
          buf79}
          .data());
  buf76.reset();
  auto tmp_tensor_handle_56 =
      reinterpret_tensor_wrapper(buf72, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf80 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_56);
  buf72.reset(); // reuse
  auto buf81 = std::move(buf38); // reuse
  auto buf82 = std::move(buf64); // reuse
  auto buf83 = std::move(buf62); // reuse
  auto buf84 = std::move(buf61); // reuse
  auto tmp_tensor_handle_57 =
      reinterpret_tensor_wrapper(buf69, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf85 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_57);
  buf69.reset(); // reuse
  auto buf86 = std::move(buf59); // reuse
  auto buf87 = std::move(buf66); // reuse
  auto tmp_tensor_handle_58 =
      reinterpret_tensor_wrapper(buf65, 5, int_array_9, int_array_10, 0L);
  decltype(auto) buf90 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_58);
  buf65.reset(); // reuse
  auto tmp_tensor_handle_28 =
      reinterpret_tensor_wrapper(buf90, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf88 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_28); // alias
  auto tmp_tensor_handle_29 =
      reinterpret_tensor_wrapper(buf90, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf89 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_29); // alias
  AtenTensorHandle buf93_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      5,
      int_array_9,
      int_array_10,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf93_handle));
  RAIIAtenTensorHandle buf93(buf93_handle);
  auto tmp_tensor_handle_30 =
      reinterpret_tensor_wrapper(buf93, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf91 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_30); // alias
  auto tmp_tensor_handle_31 =
      reinterpret_tensor_wrapper(buf93, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf92 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_31); // alias
  auto buf96 = std::move(buf75); // reuse
  auto* var_112 = get_data_ptr_wrapper(buf81);
  auto* var_113 = get_data_ptr_wrapper(buf78);
  auto* var_114 =
      get_data_ptr_wrapper(L__self___model_layers_3_attention_wo_weight);
  auto* var_115 = get_data_ptr_wrapper(buf42);
  auto* var_116 = get_data_ptr_wrapper(buf58);
  auto* var_117 = get_data_ptr_wrapper(buf63);
  auto* var_118 =
      get_data_ptr_wrapper(L__self___model_layers_3_ffn_norm_weight);
  auto* var_119 =
      get_data_ptr_wrapper(L__self___model_layers_3_feed_forward_w1_weight);
  auto* var_120 =
      get_data_ptr_wrapper(L__self___model_layers_3_feed_forward_w3_weight);
  auto* var_121 =
      get_data_ptr_wrapper(L__self___model_layers_3_feed_forward_w2_weight);
  auto* var_122 =
      get_data_ptr_wrapper(L__self___model_layers_4_attention_norm_weight);
  auto* var_123 =
      get_data_ptr_wrapper(L__self___model_layers_4_attention_wqkv_weight);
  auto* var_124 = get_data_ptr_wrapper(arg60_1);
  auto* var_125 = get_data_ptr_wrapper(L__self___model_freqs_cis);
  auto* var_126 = get_data_ptr_wrapper(buf93);
  auto* var_127 = get_data_ptr_wrapper(buf90);
  auto* var_128 = get_data_ptr_wrapper(buf80);
  auto* var_129 = get_data_ptr_wrapper(buf82);
  auto* var_130 = get_data_ptr_wrapper(buf83);
  auto* var_131 = get_data_ptr_wrapper(buf84);
  auto* var_132 = get_data_ptr_wrapper(buf85);
  auto* var_133 = get_data_ptr_wrapper(buf86);
  auto* var_134 = get_data_ptr_wrapper(buf87);
  auto* var_135 = get_data_ptr_wrapper(buf88);
  auto* var_136 = get_data_ptr_wrapper(buf89);
  auto* var_137 = get_data_ptr_wrapper(buf91);
  auto* var_138 = get_data_ptr_wrapper(buf92);
  auto* var_139 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_kv_cache_k_cache);
  auto* var_140 = get_data_ptr_wrapper(buf96);
  cpp_fused__to_copy_add_index_put_mean_mm_mul_stack_4(
      (bfloat16*)(var_112),
      (const bfloat16*)(var_113),
      (const bfloat16*)(var_114),
      (const float*)(var_115),
      (const float*)(var_116),
      (const float*)(var_117),
      (const bfloat16*)(var_118),
      (const bfloat16*)(var_119),
      (const bfloat16*)(var_120),
      (const bfloat16*)(var_121),
      (const bfloat16*)(var_122),
      (const bfloat16*)(var_123),
      (const long*)(var_124),
      (const bfloat16*)(var_125),
      (const float*)(var_126),
      (const float*)(var_127),
      (float*)(var_128),
      (float*)(var_129),
      (float*)(var_130),
      (float*)(var_131),
      (float*)(var_132),
      (float*)(var_133),
      (float*)(var_134),
      (float*)(var_135),
      (float*)(var_136),
      (float*)(var_137),
      (float*)(var_138),
      (bfloat16*)(var_139),
      (bfloat16*)(var_140));
  buf42.reset();

  buf88.reset();
  buf89.reset();
  buf91.reset();
  buf92.reset();
  // Source Nodes: [mask, y_12], Original ATen:
  // [aten._scaled_dot_product_flash_attention_for_cpu, aten.index,
  // aten.logical_not, aten.masked_fill, aten.zeros_like]
  auto tmp_tensor_handle_32 =
      reinterpret_tensor_wrapper(buf96, 4, int_array_2, int_array_3, 0L);
  auto tmp_tensor_handle_33 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  auto tmp_tensor_handle_34 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  AtenTensorHandle buf99_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf99_handle));
  RAIIAtenTensorHandle buf99(buf99_handle);
  AtenTensorHandle buf100_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf100_handle));
  RAIIAtenTensorHandle buf100(buf100_handle);
  aoti_torch_proxy_executor_call_function(
      proxy_executor,
      4,
      0,
      std::vector<int64_t>{}.data(),
      6,
      std::vector<AtenTensorHandle>{
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_32),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_33),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_34),
          buf97,
          buf99,
          buf100}
          .data());
  buf97.reset();
  auto tmp_tensor_handle_59 =
      reinterpret_tensor_wrapper(buf93, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf101 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_59);
  buf93.reset(); // reuse
  auto buf102 = std::move(buf86); // reuse
  auto tmp_tensor_handle_60 =
      reinterpret_tensor_wrapper(buf96, 3, int_array_17, int_array_18, 0L);
  decltype(auto) buf103 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_60);
  buf96.reset(); // reuse
  auto buf104 = std::move(buf84); // reuse
  auto buf105 = std::move(buf83); // reuse
  auto tmp_tensor_handle_61 =
      reinterpret_tensor_wrapper(buf90, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf106 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_61);
  buf90.reset(); // reuse
  auto buf107 = std::move(buf82); // reuse
  auto buf108 = std::move(buf80); // reuse
  auto buf109 = std::move(buf87); // reuse
  auto tmp_tensor_handle_62 =
      reinterpret_tensor_wrapper(buf63, 5, int_array_9, int_array_10, 0L);
  decltype(auto) buf112 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_62);
  buf63.reset(); // reuse
  auto tmp_tensor_handle_35 =
      reinterpret_tensor_wrapper(buf112, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf110 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_35); // alias
  auto tmp_tensor_handle_36 =
      reinterpret_tensor_wrapper(buf112, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf111 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_36); // alias
  auto tmp_tensor_handle_63 =
      reinterpret_tensor_wrapper(buf58, 5, int_array_9, int_array_10, 0L);
  decltype(auto) buf115 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_63);
  buf58.reset(); // reuse
  auto tmp_tensor_handle_37 =
      reinterpret_tensor_wrapper(buf115, 5, int_array_0, int_array_1, 0L);
  decltype(auto) buf113 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_37); // alias
  auto tmp_tensor_handle_38 =
      reinterpret_tensor_wrapper(buf115, 5, int_array_0, int_array_1, 1L);
  decltype(auto) buf114 =
      wrap_with_raii_handle_if_needed(tmp_tensor_handle_38); // alias
  auto tmp_tensor_handle_64 =
      reinterpret_tensor_wrapper(buf78, 4, int_array_11, int_array_12, 0L);
  decltype(auto) buf118 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_64);
  buf78.reset(); // reuse
  auto* var_141 = get_data_ptr_wrapper(buf99);
  auto* var_142 =
      get_data_ptr_wrapper(L__self___model_layers_4_attention_wo_weight);
  auto* var_143 = get_data_ptr_wrapper(buf81);
  auto* var_144 = get_data_ptr_wrapper(buf85);
  auto* var_145 =
      get_data_ptr_wrapper(L__self___model_layers_4_ffn_norm_weight);
  auto* var_146 =
      get_data_ptr_wrapper(L__self___model_layers_4_feed_forward_w1_weight);
  auto* var_147 =
      get_data_ptr_wrapper(L__self___model_layers_4_feed_forward_w3_weight);
  auto* var_148 =
      get_data_ptr_wrapper(L__self___model_layers_4_feed_forward_w2_weight);
  auto* var_149 =
      get_data_ptr_wrapper(L__self___model_layers_5_attention_norm_weight);
  auto* var_150 =
      get_data_ptr_wrapper(L__self___model_layers_5_attention_wqkv_weight);
  auto* var_151 = get_data_ptr_wrapper(arg60_1);
  auto* var_152 = get_data_ptr_wrapper(L__self___model_freqs_cis);
  auto* var_153 = get_data_ptr_wrapper(buf115);
  auto* var_154 = get_data_ptr_wrapper(buf112);
  auto* var_155 = get_data_ptr_wrapper(buf101);
  auto* var_156 = get_data_ptr_wrapper(buf102);
  auto* var_157 = get_data_ptr_wrapper(buf103);
  auto* var_158 = get_data_ptr_wrapper(buf104);
  auto* var_159 = get_data_ptr_wrapper(buf105);
  auto* var_160 = get_data_ptr_wrapper(buf106);
  auto* var_161 = get_data_ptr_wrapper(buf107);
  auto* var_162 = get_data_ptr_wrapper(buf108);
  auto* var_163 = get_data_ptr_wrapper(buf109);
  auto* var_164 = get_data_ptr_wrapper(buf110);
  auto* var_165 = get_data_ptr_wrapper(buf111);
  auto* var_166 = get_data_ptr_wrapper(buf113);
  auto* var_167 = get_data_ptr_wrapper(buf114);
  auto* var_168 =
      get_data_ptr_wrapper(L__self___model_layers_0_attention_kv_cache_k_cache);
  auto* var_169 = get_data_ptr_wrapper(buf118);
  cpp_fused__to_copy_add_index_put_mean_mm_mul_rsqrt_stack_5(
      (const bfloat16*)(var_141),
      (const bfloat16*)(var_142),
      (const bfloat16*)(var_143),
      (const float*)(var_144),
      (const bfloat16*)(var_145),
      (const bfloat16*)(var_146),
      (const bfloat16*)(var_147),
      (const bfloat16*)(var_148),
      (const bfloat16*)(var_149),
      (const bfloat16*)(var_150),
      (const long*)(var_151),
      (const bfloat16*)(var_152),
      (const float*)(var_153),
      (const float*)(var_154),
      (float*)(var_155),
      (float*)(var_156),
      (bfloat16*)(var_157),
      (float*)(var_158),
      (float*)(var_159),
      (float*)(var_160),
      (float*)(var_161),
      (float*)(var_162),
      (float*)(var_163),
      (float*)(var_164),
      (float*)(var_165),
      (float*)(var_166),
      (float*)(var_167),
      (bfloat16*)(var_168),
      (bfloat16*)(var_169));
  arg60_1.reset();
  buf103.reset();
  buf108.reset();
  buf109.reset();
  buf110.reset();
  buf111.reset();
  buf113.reset();
  buf114.reset();

  buf99.reset();
  // Source Nodes: [mask, y_15], Original ATen:
  // [aten._scaled_dot_product_flash_attention_for_cpu, aten.index,
  // aten.logical_not, aten.masked_fill, aten.zeros_like]
  auto tmp_tensor_handle_39 =
      reinterpret_tensor_wrapper(buf118, 4, int_array_2, int_array_3, 0L);
  auto tmp_tensor_handle_40 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  auto tmp_tensor_handle_41 = reinterpret_tensor_wrapper(
      L__self___model_layers_0_attention_kv_cache_k_cache,
      4,
      int_array_4,
      int_array_5,
      0L);
  AtenTensorHandle buf121_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf121_handle));
  RAIIAtenTensorHandle buf121(buf121_handle);
  AtenTensorHandle buf122_handle; // output buffer
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_new_uninitialized_tensor(&buf122_handle));
  RAIIAtenTensorHandle buf122(buf122_handle);
  aoti_torch_proxy_executor_call_function(
      proxy_executor,
      5,
      0,
      std::vector<int64_t>{}.data(),
      6,
      std::vector<AtenTensorHandle>{
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_39),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_40),
          wrap_with_raii_handle_if_needed(tmp_tensor_handle_41),
          buf119,
          buf121,
          buf122}
          .data());
  buf118.reset();
  buf119.reset();

  auto tmp_tensor_handle_65 =
      reinterpret_tensor_wrapper(buf115, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf123 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_65);
  buf115.reset(); // reuse
  auto buf124 = std::move(buf81); // reuse
  auto buf125 = std::move(buf107); // reuse
  auto buf126 = std::move(buf105); // reuse
  auto buf127 = std::move(buf104); // reuse
  auto tmp_tensor_handle_66 =
      reinterpret_tensor_wrapper(buf112, 2, int_array_15, int_array_16, 0L);
  decltype(auto) buf128 = wrap_with_raii_handle_if_needed(tmp_tensor_handle_66);
  buf112.reset(); // reuse
  auto buf129 = std::move(buf102); // reuse
  static constexpr int64_t int_array_21[] = {1L, 32000L};
  static constexpr int64_t int_array_22[] = {32000L, 1L};
  AtenTensorHandle buf130_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      2,
      int_array_21,
      int_array_22,
      cached_torch_dtype_float32,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf130_handle));
  RAIIAtenTensorHandle buf130(buf130_handle);
  AtenTensorHandle buf131_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      2,
      int_array_21,
      int_array_22,
      cached_torch_dtype_bfloat16,
      cached_torch_device_type_cpu,
      this->device_idx_,
      &buf131_handle));
  RAIIAtenTensorHandle buf131(buf131_handle);
  auto* var_170 = get_data_ptr_wrapper(buf124);
  auto* var_171 = get_data_ptr_wrapper(buf121);
  auto* var_172 =
      get_data_ptr_wrapper(L__self___model_layers_5_attention_wo_weight);
  auto* var_173 = get_data_ptr_wrapper(buf85);
  auto* var_174 = get_data_ptr_wrapper(buf101);
  auto* var_175 = get_data_ptr_wrapper(buf106);
  auto* var_176 =
      get_data_ptr_wrapper(L__self___model_layers_5_ffn_norm_weight);
  auto* var_177 =
      get_data_ptr_wrapper(L__self___model_layers_5_feed_forward_w1_weight);
  auto* var_178 =
      get_data_ptr_wrapper(L__self___model_layers_5_feed_forward_w3_weight);
  auto* var_179 =
      get_data_ptr_wrapper(L__self___model_layers_5_feed_forward_w2_weight);
  auto* var_180 = get_data_ptr_wrapper(L__self___model_norm_weight);
  auto* var_181 = get_data_ptr_wrapper(L__self___model_tok_embeddings_weight);
  auto* var_182 = get_data_ptr_wrapper(buf123);
  auto* var_183 = get_data_ptr_wrapper(buf125);
  auto* var_184 = get_data_ptr_wrapper(buf126);
  auto* var_185 = get_data_ptr_wrapper(buf127);
  auto* var_186 = get_data_ptr_wrapper(buf128);
  auto* var_187 = get_data_ptr_wrapper(buf129);
  auto* var_188 = get_data_ptr_wrapper(buf130);
  auto* var_189 = get_data_ptr_wrapper(buf131);
  cpp_fused__to_copy_add_mean_mm_mul_6(
      (bfloat16*)(var_170),
      (const bfloat16*)(var_171),
      (const bfloat16*)(var_172),
      (const float*)(var_173),
      (const float*)(var_174),
      (const float*)(var_175),
      (const bfloat16*)(var_176),
      (const bfloat16*)(var_177),
      (const bfloat16*)(var_178),
      (const bfloat16*)(var_179),
      (const bfloat16*)(var_180),
      (const bfloat16*)(var_181),
      (float*)(var_182),
      (float*)(var_183),
      (float*)(var_184),
      (float*)(var_185),
      (float*)(var_186),
      (float*)(var_187),
      (float*)(var_188),
      (bfloat16*)(var_189));
  static constexpr int64_t int_array_23[] = {1L, 1L, 32000L};
  static constexpr int64_t int_array_24[] = {0L, 0L, 1L};
  auto tmp_tensor_handle_67 =
      reinterpret_tensor_wrapper(buf131, 3, int_array_23, int_array_24, 0L);
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[0]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_0(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_0.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[0]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_0.tensor(), output_handles[0]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[1]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_1(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_1.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[1]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_1.tensor(), output_handles[1]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[2]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_2(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_2.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[2]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_2.tensor(), output_handles[2]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[3]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_3(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_3.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[3]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_3.tensor(), output_handles[3]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[4]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_4(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_4.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[4]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_4.tensor(), output_handles[4]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[5]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_5(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_5.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[5]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_5.tensor(), output_handles[5]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[6]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_6(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_6.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[6]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_6.tensor(), output_handles[6]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[7]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_7(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_7.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[7]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_7.tensor(), output_handles[7]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[8]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_8(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_8.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[8]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_8.tensor(), output_handles[8]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[9]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_9(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_9.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[9]));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_assign_tensors(cached_output_9.tensor(), output_handles[9]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[10]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_10(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_10.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[10]));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors(
        cached_output_10.tensor(), output_handles[10]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<
              decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>,
          ConstantHandle>) {
    aoti_torch_clone(
        L__self___model_layers_0_attention_kv_cache_k_cache,
        &output_handles[11]);
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(L__self___model_layers_0_attention_kv_cache_k_cache)>>
        cached_output_11(L__self___model_layers_0_attention_kv_cache_k_cache);
    cached_output_11.copy_data_from(
        L__self___model_layers_0_attention_kv_cache_k_cache);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[11]));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors(
        cached_output_11.tensor(), output_handles[11]));
  }
  if constexpr (
      std::is_same_v<
          std::decay_t<decltype(wrap_with_raii_handle_if_needed(
              tmp_tensor_handle_67))>,
          RAIIAtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<decltype(wrap_with_raii_handle_if_needed(
              tmp_tensor_handle_67))>,
          AtenTensorHandle> ||
      std::is_same_v<
          std::decay_t<decltype(wrap_with_raii_handle_if_needed(
              tmp_tensor_handle_67))>,
          ConstantHandle>) {
    output_handles[12] =
        wrap_with_raii_handle_if_needed(tmp_tensor_handle_67).release();
  } else {
    thread_local ThreadLocalCachedOutputTensor<std::decay_t<
        decltype(wrap_with_raii_handle_if_needed(tmp_tensor_handle_67))>>
        cached_output_12(wrap_with_raii_handle_if_needed(tmp_tensor_handle_67));
    cached_output_12.copy_data_from(
        wrap_with_raii_handle_if_needed(tmp_tensor_handle_67));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_new_uninitialized_tensor(&output_handles[12]));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors(
        cached_output_12.tensor(), output_handles[12]));
  }
} // AOTInductorModel::run_impl
} // namespace aot_inductor
} // namespace torch

// Compile cmd
// /mnt/gvfs/third-party2/llvm-fb/15e484798e5d10722fef1beff194b84a2c164a7f/17/platform010/72a2ff8/bin/clang-17
// clcdeucbxejd47smk7lpdy334gnrpjiumuqnbnooa7ctvjzix4we.cpp -fPIC -Wall
// -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas
// -Werror=ignored-optimization-argument -D_GLIBCXX_USE_CXX11_ABI=1
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include/torch/csrc/api/include
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include/TH
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include/THC
// -I/usr/local/fbcode/platform010/include/python3.10
// -I/tmp/torchinductor_mikekg/wy
// -I/mnt/gvfs/third-party2/sleef/04504ce3d344bc68cbe6809aae1e3fbd8d62047f/trunk/platform010/dfaae0f/include
// -I/mnt/gvfs/third-party2/openmp/8804b8005761277a10fa028882f4b4724a598775/12/platform010/76ebdda/include
// -I/mnt/gvfs/third-party2/llvm-fb/15e484798e5d10722fef1beff194b84a2c164a7f/17/platform010/72a2ff8/lib/clang/stable/include
// -I/mnt/gvfs/third-party2/libgcc/d1129753c8361ac8e9453c0f4291337a4507ebe6/11.x/platform010/5684a5a/include/c++/11.2.0
// -I/mnt/gvfs/third-party2/libgcc/d1129753c8361ac8e9453c0f4291337a4507ebe6/11.x/platform010/5684a5a/include/c++/11.2.0/x86_64-facebook-linux
// -I/mnt/gvfs/third-party2/libgcc/d1129753c8361ac8e9453c0f4291337a4507ebe6/11.x/platform010/5684a5a/include/c++/11.2.0/backward
// -I/mnt/gvfs/third-party2/glibc/fed6e93d87571fb162734c86636119d45a398963/2.34/platform010/f259413/include
// -I/mnt/gvfs/third-party2/kernel-headers/624a2f8f6c93c3c1df8aa4a6255d8202631a6c80/fb/platform010/da39a3e/include
// -I/mnt/gvfs/third-party2/cuda/7991bb81a2e29d342009e443f958c3f8fe118e7a/12/platform010/76ebdda/include
// -Iinclude -mavx2 -mfma -mavx2 -mfma -D CPU_CAPABILITY=AVX2 -D
// CPU_CAPABILITY_AVX2 -D HAVE_AVX2_CPU_DEFINITION
// -B/mnt/gvfs/third-party2/glibc/fed6e93d87571fb162734c86636119d45a398963/2.34/platform010/f259413/lib
// -L/mnt/gvfs/third-party2/glibc/fed6e93d87571fb162734c86636119d45a398963/2.34/platform010/f259413/lib
// --rtlib=compiler-rt -fuse-ld=lld -Wl,--script=script.ld -O3 -DNDEBUG
// -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations
// -ffp-contract=off -D C10_USING_CUSTOM_GENERATED_MACROS -Wp,-fopenmp
// /mnt/gvfs/third-party2/openmp/8804b8005761277a10fa028882f4b4724a598775/12/platform010/76ebdda/lib/libomp.so
// -D C10_USE_GLOG -D C10_USE_MINIMAL_GLOG -D
// C10_DISABLE_TENSORIMPL_EXTENSIBILITY -nostdinc -c -o
// clcdeucbxejd47smk7lpdy334gnrpjiumuqnbnooa7ctvjzix4we.o Link cmd
// /mnt/gvfs/third-party2/llvm-fb/15e484798e5d10722fef1beff194b84a2c164a7f/17/platform010/72a2ff8/bin/clang-17
// clcdeucbxejd47smk7lpdy334gnrpjiumuqnbnooa7ctvjzix4we.o
// cedhqkkdrusj5ojzfpm6vfftyqochh2hj5aldjkzpof5cmg6fa7p.o -shared -fPIC -Wall
// -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas
// -Werror=ignored-optimization-argument -D_GLIBCXX_USE_CXX11_ABI=1
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include/torch/csrc/api/include
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include/TH
// -I/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/include/THC
// -I/usr/local/fbcode/platform010/include/python3.10
// -I/tmp/torchinductor_mikekg/wy
// -I/mnt/gvfs/third-party2/sleef/04504ce3d344bc68cbe6809aae1e3fbd8d62047f/trunk/platform010/dfaae0f/include
// -I/mnt/gvfs/third-party2/openmp/8804b8005761277a10fa028882f4b4724a598775/12/platform010/76ebdda/include
// -I/mnt/gvfs/third-party2/llvm-fb/15e484798e5d10722fef1beff194b84a2c164a7f/17/platform010/72a2ff8/lib/clang/stable/include
// -I/mnt/gvfs/third-party2/libgcc/d1129753c8361ac8e9453c0f4291337a4507ebe6/11.x/platform010/5684a5a/include/c++/11.2.0
// -I/mnt/gvfs/third-party2/libgcc/d1129753c8361ac8e9453c0f4291337a4507ebe6/11.x/platform010/5684a5a/include/c++/11.2.0/x86_64-facebook-linux
// -I/mnt/gvfs/third-party2/libgcc/d1129753c8361ac8e9453c0f4291337a4507ebe6/11.x/platform010/5684a5a/include/c++/11.2.0/backward
// -I/mnt/gvfs/third-party2/glibc/fed6e93d87571fb162734c86636119d45a398963/2.34/platform010/f259413/include
// -I/mnt/gvfs/third-party2/kernel-headers/624a2f8f6c93c3c1df8aa4a6255d8202631a6c80/fb/platform010/da39a3e/include
// -I/mnt/gvfs/third-party2/cuda/7991bb81a2e29d342009e443f958c3f8fe118e7a/12/platform010/76ebdda/include
// -Iinclude
// -L/data/users/mikekg/fbsource/buck-out/v2/gen/fbcode/16379dd050272c42/scripts/mikekg/gpt_fast/__export__/export#link-tree/torch/lib
// -L/usr/local/fbcode/platform010/lib -lomp -mavx2 -mfma -mavx2 -mfma -D
// CPU_CAPABILITY=AVX2 -D CPU_CAPABILITY_AVX2 -D HAVE_AVX2_CPU_DEFINITION
// -B/mnt/gvfs/third-party2/glibc/fed6e93d87571fb162734c86636119d45a398963/2.34/platform010/f259413/lib
// -L/mnt/gvfs/third-party2/glibc/fed6e93d87571fb162734c86636119d45a398963/2.34/platform010/f259413/lib
// --rtlib=compiler-rt -fuse-ld=lld -Wl,--script=script.ld -O3 -DNDEBUG
// -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations
// -ffp-contract=off -D C10_USING_CUSTOM_GENERATED_MACROS -Wp,-fopenmp
// /mnt/gvfs/third-party2/openmp/8804b8005761277a10fa028882f4b4724a598775/12/platform010/76ebdda/lib/libomp.so
// -D C10_USE_GLOG -D C10_USE_MINIMAL_GLOG -D
// C10_DISABLE_TENSORIMPL_EXTENSIBILITY -nostdinc -o
// clcdeucbxejd47smk7lpdy334gnrpjiumuqnbnooa7ctvjzix4we.so
