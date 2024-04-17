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
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
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
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
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
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
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
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
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
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

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
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

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
    err << "incorrect numel for input tensor. expected " << numel << ", got " << tensor.numel();
    throw std::runtime_error(err.str());
  }
}
} // namespace aot_inductor
} // namespace torch

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/BinaryOps.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/types.h>
#include <ATen/ops/bernoulli_native.h>

#define reinterpret_tensor torch::inductor::_reinterpret_tensor
#define alloc_from_pool torch::inductor::_alloc_from_pool
#include <c10/util/generic_math.h>

[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_embedding_mean_mul_rsqrt_0(const int* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const signed char* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*tmp3)), 8);
                    auto tmp5 = tmp4 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp5 = out_ptr0[static_cast<long>(x0)];
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*tmp3)), 8);
                auto tmp6 = static_cast<float>(288.0);
                auto tmp7 = tmp5 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                auto tmp10 = 1 / std::sqrt(tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                tmp14.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(248832L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__scaled_dot_product_flash_attention_for_cpu_index_index_put_logical_not_masked_fill_stack_zeros_like_1(const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const int* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0+=static_cast<long>(8L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp5 = at::vec::Vectorized<int>::loadu(in_ptr2 + static_cast<long>(x0), 8);
                    auto tmp16 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp17 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp32 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp33 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp40 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp41 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = static_cast<int>(2048);
                    auto tmp7 = at::vec::Vectorized<int>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = at::vec::Vectorized<int>(tmp9);
                    auto tmp11 = at::vec::VecMask<int,1>(tmp5 < tmp10);
                    auto tmp12 = decltype(tmp8)::blendv(tmp5, tmp8, tmp11.template cast<int,1>());
                    AOTI_TORCH_CHECK((at::vec::VecMask<int,1>((at::vec::Vectorized<int>(0) <= tmp12) & (tmp12 < at::vec::Vectorized<int>(2048L)))).all_masked(), "index out of bounds: 0 <= tmp12 < 2048L")
                    auto tmp13 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp14 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp13[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp15 = tmp4 * tmp14;
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp16 * tmp19;
                    auto tmp21 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp22 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp21[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp15 - tmp23;
                    auto tmp25 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp26 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp25[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp27 = tmp20 * tmp26;
                    auto tmp28 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp29 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp28[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp30 = tmp4 * tmp29;
                    auto tmp31 = tmp27 + tmp30;
                    auto tmp34 = c10::convert<float>(tmp33);
                    auto tmp35 = at::vec::Vectorized<float>(tmp34);
                    auto tmp36 = tmp32 * tmp35;
                    auto tmp37 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp38 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp37[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp39 = tmp36 * tmp38;
                    auto tmp42 = c10::convert<float>(tmp41);
                    auto tmp43 = at::vec::Vectorized<float>(tmp42);
                    auto tmp44 = tmp40 * tmp43;
                    auto tmp45 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp46 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp45[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp39 - tmp47;
                    auto tmp49 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp50 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp49[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp51 = tmp44 * tmp50;
                    auto tmp52 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp53 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp52[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp54 = tmp36 * tmp53;
                    auto tmp55 = tmp51 + tmp54;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp24.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp31.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp48.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp55.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                }
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp10 = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp11 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp20 = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp21 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp25 = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp26 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                    auto tmp5 = decltype(tmp4)(tmp4 + 2048);
                    auto tmp6 = tmp4 < 0;
                    auto tmp7 = tmp6 ? tmp5 : tmp4;
                    AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 2048L), "index out of bounds: 0 <= tmp7 < 2048L")
                    auto tmp8 = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp7))];
                    auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                    auto tmp12 = c10::convert<float>(tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp7))];
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 - tmp15);
                    auto tmp17 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp18 = decltype(tmp3)(tmp3 * tmp14);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp22 = c10::convert<float>(tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp8);
                    auto tmp27 = c10::convert<float>(tmp26);
                    auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp14);
                    auto tmp30 = decltype(tmp24)(tmp24 - tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp8);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp14);
                    auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                    out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp16;
                    out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp19;
                    out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp30;
                    out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp33;
                }
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x0)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (48L*x1) + (288L*x0)), 8);
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(576L + x2 + (48L*x1) + (864L*x0)), 8);
                            auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(576L + x2 + (48L*x1)), 8);
                            auto tmp1 = decltype(tmp0)(tmp0 + 352);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                            auto tmp7 = at::vec::convert<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                            tmp8.store(out_ptr5 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 352);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                        auto tmp4 = at::vec::VecMask<float,1>::from(in_ptr7 + static_cast<long>(x1 + (352L*tmp3)));
                        auto tmp5 = ~tmp4;
                        auto tmp6 = -std::numeric_limits<float>::infinity();
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = at::vec::Vectorized<float>(tmp7);
                        auto tmp10 = decltype(tmp8)::blendv(tmp9, tmp8, tmp5.template cast<float,1>());
                        tmp10.store(out_ptr6 + static_cast<long>(x1 + (352L*x0)));
                        tmp10.store(out_ptr7 + static_cast<long>(x1 + (352L*x0)));
                        tmp10.store(out_ptr8 + static_cast<long>(x1 + (352L*x0)));
                    }
                }
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_2(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(82944L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_embedding_mean_mul_rsqrt_3(const int* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const bfloat16* in_ptr3,
                       const float* in_ptr4,
                       const signed char* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*tmp3)), 8);
                    auto tmp7 = at::vec::convert<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp4 + tmp8;
                    auto tmp10 = tmp9 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp10;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 8);
                auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*tmp3)), 8);
                auto tmp7 = at::vec::convert<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp4 + tmp8;
                auto tmp11 = static_cast<float>(288.0);
                auto tmp12 = tmp10 / tmp11;
                auto tmp13 = static_cast<float>(1e-05);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp15 = 1 / std::sqrt(tmp14);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp9 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_4(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_mul_silu_5(float* in_out_ptr0,
                       const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const signed char* in_ptr3,
                       float* out_ptr0,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1), 8);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp2 = at::vec::convert<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                auto tmp5 = tmp3 * tmp4;
                auto tmp8 = at::vec::convert<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = tmp5 * tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_embedding_mean_mul_rsqrt_6(float* in_out_ptr0,
                       const int* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const float* in_ptr3,
                       const bfloat16* in_ptr4,
                       const float* in_ptr5,
                       const signed char* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp11 = at::vec::Vectorized<bfloat16>::loadu(in_ptr4 + static_cast<long>(x1), 8);
                    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*tmp3)), 8);
                    auto tmp7 = at::vec::convert<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp4 + tmp8;
                    auto tmp12 = at::vec::convert<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    auto tmp15 = tmp14 * tmp14;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp15;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1), 8);
                auto tmp2 = static_cast<float>(288.0);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = 1 / std::sqrt(tmp5);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp0 * tmp7;
                auto tmp10 = tmp8 * tmp9;
                tmp10.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(248832L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr6 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused_index_put_stack_7(const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const int* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0+=static_cast<long>(8L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp5 = at::vec::Vectorized<int>::loadu(in_ptr2 + static_cast<long>(x0), 8);
                    auto tmp16 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp17 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp32 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp33 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp40 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp41 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = static_cast<int>(2048);
                    auto tmp7 = at::vec::Vectorized<int>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = at::vec::Vectorized<int>(tmp9);
                    auto tmp11 = at::vec::VecMask<int,1>(tmp5 < tmp10);
                    auto tmp12 = decltype(tmp8)::blendv(tmp5, tmp8, tmp11.template cast<int,1>());
                    AOTI_TORCH_CHECK((at::vec::VecMask<int,1>((at::vec::Vectorized<int>(0) <= tmp12) & (tmp12 < at::vec::Vectorized<int>(2048L)))).all_masked(), "index out of bounds: 0 <= tmp12 < 2048L")
                    auto tmp13 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp14 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp13[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp15 = tmp4 * tmp14;
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp16 * tmp19;
                    auto tmp21 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp22 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp21[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp15 - tmp23;
                    auto tmp25 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp26 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp25[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp27 = tmp20 * tmp26;
                    auto tmp28 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp29 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp28[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp30 = tmp4 * tmp29;
                    auto tmp31 = tmp27 + tmp30;
                    auto tmp34 = c10::convert<float>(tmp33);
                    auto tmp35 = at::vec::Vectorized<float>(tmp34);
                    auto tmp36 = tmp32 * tmp35;
                    auto tmp37 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp38 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp37[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp39 = tmp36 * tmp38;
                    auto tmp42 = c10::convert<float>(tmp41);
                    auto tmp43 = at::vec::Vectorized<float>(tmp42);
                    auto tmp44 = tmp40 * tmp43;
                    auto tmp45 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp46 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp45[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp39 - tmp47;
                    auto tmp49 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp50 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp49[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp51 = tmp44 * tmp50;
                    auto tmp52 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp53 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp52[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp54 = tmp36 * tmp53;
                    auto tmp55 = tmp51 + tmp54;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp24.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp31.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp48.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp55.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                }
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp10 = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp11 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp20 = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp21 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp25 = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp26 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                    auto tmp5 = decltype(tmp4)(tmp4 + 2048);
                    auto tmp6 = tmp4 < 0;
                    auto tmp7 = tmp6 ? tmp5 : tmp4;
                    AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 2048L), "index out of bounds: 0 <= tmp7 < 2048L")
                    auto tmp8 = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp7))];
                    auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                    auto tmp12 = c10::convert<float>(tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp7))];
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 - tmp15);
                    auto tmp17 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp18 = decltype(tmp3)(tmp3 * tmp14);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp22 = c10::convert<float>(tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp8);
                    auto tmp27 = c10::convert<float>(tmp26);
                    auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp14);
                    auto tmp30 = decltype(tmp24)(tmp24 - tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp8);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp14);
                    auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                    out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp16;
                    out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp19;
                    out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp30;
                    out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp33;
                }
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x0)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (48L*x1) + (288L*x0)), 8);
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(576L + x2 + (48L*x1) + (864L*x0)), 8);
                            auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(576L + x2 + (48L*x1)), 8);
                            auto tmp1 = decltype(tmp0)(tmp0 + 352);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                            auto tmp7 = at::vec::convert<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                            tmp8.store(out_ptr5 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                        }
                    }
                }
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_8(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(82944L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_9(const float* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const float* in_ptr3,
                       const signed char* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp6 = tmp5 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp6 = out_ptr0[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                auto tmp3 = at::vec::convert<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 + tmp4;
                auto tmp7 = static_cast<float>(288.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                tmp15.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_10(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_mul_silu_11(float* in_out_ptr0,
                       const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const signed char* in_ptr3,
                       float* out_ptr0,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1), 8);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp2 = at::vec::convert<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                auto tmp5 = tmp3 * tmp4;
                auto tmp8 = at::vec::convert<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = tmp5 * tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_12(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const float* in_ptr2,
                       const bfloat16* in_ptr3,
                       const float* in_ptr4,
                       const signed char* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1), 8);
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp8 = at::vec::convert<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp5 + tmp9;
                    auto tmp11 = tmp10 * tmp10;
                    tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp11;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 8);
                auto tmp2 = static_cast<float>(288.0);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = 1 / std::sqrt(tmp5);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp0 * tmp7;
                auto tmp10 = tmp8 * tmp9;
                tmp10.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(248832L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused_index_put_stack_13(const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const int* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0+=static_cast<long>(8L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp5 = at::vec::Vectorized<int>::loadu(in_ptr2 + static_cast<long>(x0), 8);
                    auto tmp16 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp17 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp32 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp33 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp40 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp41 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = static_cast<int>(2048);
                    auto tmp7 = at::vec::Vectorized<int>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = at::vec::Vectorized<int>(tmp9);
                    auto tmp11 = at::vec::VecMask<int,1>(tmp5 < tmp10);
                    auto tmp12 = decltype(tmp8)::blendv(tmp5, tmp8, tmp11.template cast<int,1>());
                    AOTI_TORCH_CHECK((at::vec::VecMask<int,1>((at::vec::Vectorized<int>(0) <= tmp12) & (tmp12 < at::vec::Vectorized<int>(2048L)))).all_masked(), "index out of bounds: 0 <= tmp12 < 2048L")
                    auto tmp13 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp14 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp13[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp15 = tmp4 * tmp14;
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp16 * tmp19;
                    auto tmp21 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp22 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp21[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp15 - tmp23;
                    auto tmp25 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp26 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp25[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp27 = tmp20 * tmp26;
                    auto tmp28 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp29 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp28[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp30 = tmp4 * tmp29;
                    auto tmp31 = tmp27 + tmp30;
                    auto tmp34 = c10::convert<float>(tmp33);
                    auto tmp35 = at::vec::Vectorized<float>(tmp34);
                    auto tmp36 = tmp32 * tmp35;
                    auto tmp37 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp38 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp37[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp39 = tmp36 * tmp38;
                    auto tmp42 = c10::convert<float>(tmp41);
                    auto tmp43 = at::vec::Vectorized<float>(tmp42);
                    auto tmp44 = tmp40 * tmp43;
                    auto tmp45 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp46 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp45[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp39 - tmp47;
                    auto tmp49 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp50 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp49[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp51 = tmp44 * tmp50;
                    auto tmp52 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp53 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp52[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp54 = tmp36 * tmp53;
                    auto tmp55 = tmp51 + tmp54;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp24.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp31.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp48.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp55.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                }
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp10 = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp11 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp20 = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp21 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp25 = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp26 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                    auto tmp5 = decltype(tmp4)(tmp4 + 2048);
                    auto tmp6 = tmp4 < 0;
                    auto tmp7 = tmp6 ? tmp5 : tmp4;
                    AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 2048L), "index out of bounds: 0 <= tmp7 < 2048L")
                    auto tmp8 = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp7))];
                    auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                    auto tmp12 = c10::convert<float>(tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp7))];
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 - tmp15);
                    auto tmp17 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp18 = decltype(tmp3)(tmp3 * tmp14);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp22 = c10::convert<float>(tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp8);
                    auto tmp27 = c10::convert<float>(tmp26);
                    auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp14);
                    auto tmp30 = decltype(tmp24)(tmp24 - tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp8);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp14);
                    auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                    out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp16;
                    out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp19;
                    out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp30;
                    out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp33;
                }
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x0)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (48L*x1) + (288L*x0)), 8);
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(576L + x2 + (48L*x1) + (864L*x0)), 8);
                            auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(576L + x2 + (48L*x1)), 8);
                            auto tmp1 = decltype(tmp0)(tmp0 + 352);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                            auto tmp7 = at::vec::convert<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                            tmp8.store(out_ptr5 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                        }
                    }
                }
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_14(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(82944L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_15(const float* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const float* in_ptr3,
                       const signed char* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp6 = tmp5 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp6 = out_ptr0[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                auto tmp3 = at::vec::convert<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 + tmp4;
                auto tmp7 = static_cast<float>(288.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                tmp15.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_16(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_mul_silu_17(float* in_out_ptr0,
                       const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const signed char* in_ptr3,
                       float* out_ptr0,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1), 8);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp2 = at::vec::convert<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                auto tmp5 = tmp3 * tmp4;
                auto tmp8 = at::vec::convert<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = tmp5 * tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_18(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const float* in_ptr2,
                       const bfloat16* in_ptr3,
                       const float* in_ptr4,
                       const signed char* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1), 8);
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp8 = at::vec::convert<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp5 + tmp9;
                    auto tmp11 = tmp10 * tmp10;
                    tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp11;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 8);
                auto tmp2 = static_cast<float>(288.0);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = 1 / std::sqrt(tmp5);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp0 * tmp7;
                auto tmp10 = tmp8 * tmp9;
                tmp10.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(248832L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__scaled_dot_product_flash_attention_for_cpu_index_index_put_logical_not_masked_fill_stack_zeros_like_19(const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const int* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0+=static_cast<long>(8L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp5 = at::vec::Vectorized<int>::loadu(in_ptr2 + static_cast<long>(x0), 8);
                    auto tmp16 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp17 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp32 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp33 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp40 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp41 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = static_cast<int>(2048);
                    auto tmp7 = at::vec::Vectorized<int>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = at::vec::Vectorized<int>(tmp9);
                    auto tmp11 = at::vec::VecMask<int,1>(tmp5 < tmp10);
                    auto tmp12 = decltype(tmp8)::blendv(tmp5, tmp8, tmp11.template cast<int,1>());
                    AOTI_TORCH_CHECK((at::vec::VecMask<int,1>((at::vec::Vectorized<int>(0) <= tmp12) & (tmp12 < at::vec::Vectorized<int>(2048L)))).all_masked(), "index out of bounds: 0 <= tmp12 < 2048L")
                    auto tmp13 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp14 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp13[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp15 = tmp4 * tmp14;
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp16 * tmp19;
                    auto tmp21 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp22 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp21[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp15 - tmp23;
                    auto tmp25 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp26 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp25[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp27 = tmp20 * tmp26;
                    auto tmp28 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp29 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp28[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp30 = tmp4 * tmp29;
                    auto tmp31 = tmp27 + tmp30;
                    auto tmp34 = c10::convert<float>(tmp33);
                    auto tmp35 = at::vec::Vectorized<float>(tmp34);
                    auto tmp36 = tmp32 * tmp35;
                    auto tmp37 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp38 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp37[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp39 = tmp36 * tmp38;
                    auto tmp42 = c10::convert<float>(tmp41);
                    auto tmp43 = at::vec::Vectorized<float>(tmp42);
                    auto tmp44 = tmp40 * tmp43;
                    auto tmp45 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp46 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp45[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp39 - tmp47;
                    auto tmp49 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp50 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp49[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp51 = tmp44 * tmp50;
                    auto tmp52 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp53 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp52[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp54 = tmp36 * tmp53;
                    auto tmp55 = tmp51 + tmp54;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp24.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp31.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp48.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp55.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                }
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp10 = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp11 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp20 = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp21 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp25 = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp26 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                    auto tmp5 = decltype(tmp4)(tmp4 + 2048);
                    auto tmp6 = tmp4 < 0;
                    auto tmp7 = tmp6 ? tmp5 : tmp4;
                    AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 2048L), "index out of bounds: 0 <= tmp7 < 2048L")
                    auto tmp8 = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp7))];
                    auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                    auto tmp12 = c10::convert<float>(tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp7))];
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 - tmp15);
                    auto tmp17 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp18 = decltype(tmp3)(tmp3 * tmp14);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp22 = c10::convert<float>(tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp8);
                    auto tmp27 = c10::convert<float>(tmp26);
                    auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp14);
                    auto tmp30 = decltype(tmp24)(tmp24 - tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp8);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp14);
                    auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                    out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp16;
                    out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp19;
                    out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp30;
                    out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp33;
                }
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x0)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (48L*x1) + (288L*x0)), 8);
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(576L + x2 + (48L*x1) + (864L*x0)), 8);
                            auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(576L + x2 + (48L*x1)), 8);
                            auto tmp1 = decltype(tmp0)(tmp0 + 352);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                            auto tmp7 = at::vec::convert<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                            tmp8.store(out_ptr5 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(352L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 352);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                        auto tmp4 = at::vec::VecMask<float,1>::from(in_ptr7 + static_cast<long>(x1 + (352L*tmp3)));
                        auto tmp5 = ~tmp4;
                        auto tmp6 = -std::numeric_limits<float>::infinity();
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = at::vec::Vectorized<float>(tmp7);
                        auto tmp10 = decltype(tmp8)::blendv(tmp9, tmp8, tmp5.template cast<float,1>());
                        tmp10.store(out_ptr6 + static_cast<long>(x1 + (352L*x0)));
                        tmp10.store(out_ptr7 + static_cast<long>(x1 + (352L*x0)));
                        tmp10.store(out_ptr8 + static_cast<long>(x1 + (352L*x0)));
                    }
                }
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_20(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(82944L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_21(const float* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const float* in_ptr3,
                       const signed char* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp6 = tmp5 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp6 = out_ptr0[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                auto tmp3 = at::vec::convert<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 + tmp4;
                auto tmp7 = static_cast<float>(288.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                tmp15.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_22(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_mul_silu_23(float* in_out_ptr0,
                       const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const signed char* in_ptr3,
                       float* out_ptr0,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1), 8);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp2 = at::vec::convert<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                auto tmp5 = tmp3 * tmp4;
                auto tmp8 = at::vec::convert<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = tmp5 * tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_24(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const float* in_ptr2,
                       const bfloat16* in_ptr3,
                       const float* in_ptr4,
                       const signed char* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1), 8);
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp8 = at::vec::convert<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp5 + tmp9;
                    auto tmp11 = tmp10 * tmp10;
                    tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp11;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 8);
                auto tmp2 = static_cast<float>(288.0);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = 1 / std::sqrt(tmp5);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp0 * tmp7;
                auto tmp10 = tmp8 * tmp9;
                tmp10.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(248832L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused_index_put_stack_25(const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const int* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0+=static_cast<long>(8L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp5 = at::vec::Vectorized<int>::loadu(in_ptr2 + static_cast<long>(x0), 8);
                    auto tmp16 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp17 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp32 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp33 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp40 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp41 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = static_cast<int>(2048);
                    auto tmp7 = at::vec::Vectorized<int>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = at::vec::Vectorized<int>(tmp9);
                    auto tmp11 = at::vec::VecMask<int,1>(tmp5 < tmp10);
                    auto tmp12 = decltype(tmp8)::blendv(tmp5, tmp8, tmp11.template cast<int,1>());
                    AOTI_TORCH_CHECK((at::vec::VecMask<int,1>((at::vec::Vectorized<int>(0) <= tmp12) & (tmp12 < at::vec::Vectorized<int>(2048L)))).all_masked(), "index out of bounds: 0 <= tmp12 < 2048L")
                    auto tmp13 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp14 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp13[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp15 = tmp4 * tmp14;
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp16 * tmp19;
                    auto tmp21 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp22 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp21[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp15 - tmp23;
                    auto tmp25 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp26 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp25[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp27 = tmp20 * tmp26;
                    auto tmp28 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp29 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp28[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp30 = tmp4 * tmp29;
                    auto tmp31 = tmp27 + tmp30;
                    auto tmp34 = c10::convert<float>(tmp33);
                    auto tmp35 = at::vec::Vectorized<float>(tmp34);
                    auto tmp36 = tmp32 * tmp35;
                    auto tmp37 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp38 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp37[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp39 = tmp36 * tmp38;
                    auto tmp42 = c10::convert<float>(tmp41);
                    auto tmp43 = at::vec::Vectorized<float>(tmp42);
                    auto tmp44 = tmp40 * tmp43;
                    auto tmp45 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp46 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp45[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp39 - tmp47;
                    auto tmp49 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp50 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp49[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp51 = tmp44 * tmp50;
                    auto tmp52 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp53 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp52[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp54 = tmp36 * tmp53;
                    auto tmp55 = tmp51 + tmp54;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp24.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp31.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp48.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp55.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                }
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp10 = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp11 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp20 = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp21 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp25 = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp26 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                    auto tmp5 = decltype(tmp4)(tmp4 + 2048);
                    auto tmp6 = tmp4 < 0;
                    auto tmp7 = tmp6 ? tmp5 : tmp4;
                    AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 2048L), "index out of bounds: 0 <= tmp7 < 2048L")
                    auto tmp8 = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp7))];
                    auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                    auto tmp12 = c10::convert<float>(tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp7))];
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 - tmp15);
                    auto tmp17 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp18 = decltype(tmp3)(tmp3 * tmp14);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp22 = c10::convert<float>(tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp8);
                    auto tmp27 = c10::convert<float>(tmp26);
                    auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp14);
                    auto tmp30 = decltype(tmp24)(tmp24 - tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp8);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp14);
                    auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                    out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp16;
                    out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp19;
                    out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp30;
                    out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp33;
                }
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x0)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (48L*x1) + (288L*x0)), 8);
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(576L + x2 + (48L*x1) + (864L*x0)), 8);
                            auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(576L + x2 + (48L*x1)), 8);
                            auto tmp1 = decltype(tmp0)(tmp0 + 352);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                            auto tmp7 = at::vec::convert<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                            tmp8.store(out_ptr5 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                        }
                    }
                }
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_26(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(82944L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_27(const float* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const float* in_ptr3,
                       const signed char* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp6 = tmp5 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp6 = out_ptr0[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                auto tmp3 = at::vec::convert<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 + tmp4;
                auto tmp7 = static_cast<float>(288.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                tmp15.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_28(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_mul_silu_29(float* in_out_ptr0,
                       const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const signed char* in_ptr3,
                       float* out_ptr0,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1), 8);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp2 = at::vec::convert<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                auto tmp5 = tmp3 * tmp4;
                auto tmp8 = at::vec::convert<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = tmp5 * tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_30(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const float* in_ptr2,
                       const bfloat16* in_ptr3,
                       const float* in_ptr4,
                       const signed char* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1), 8);
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp8 = at::vec::convert<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp5 + tmp9;
                    auto tmp11 = tmp10 * tmp10;
                    tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp11;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 8);
                auto tmp2 = static_cast<float>(288.0);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = 1 / std::sqrt(tmp5);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp0 * tmp7;
                auto tmp10 = tmp8 * tmp9;
                tmp10.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(248832L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused_index_put_stack_31(const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const int* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0+=static_cast<long>(8L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp5 = at::vec::Vectorized<int>::loadu(in_ptr2 + static_cast<long>(x0), 8);
                    auto tmp16 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp17 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp32 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp33 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp40 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0) + (864L*x0_inner))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp41 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = static_cast<int>(2048);
                    auto tmp7 = at::vec::Vectorized<int>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = at::vec::Vectorized<int>(tmp9);
                    auto tmp11 = at::vec::VecMask<int,1>(tmp5 < tmp10);
                    auto tmp12 = decltype(tmp8)::blendv(tmp5, tmp8, tmp11.template cast<int,1>());
                    AOTI_TORCH_CHECK((at::vec::VecMask<int,1>((at::vec::Vectorized<int>(0) <= tmp12) & (tmp12 < at::vec::Vectorized<int>(2048L)))).all_masked(), "index out of bounds: 0 <= tmp12 < 2048L")
                    auto tmp13 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp14 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp13[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp15 = tmp4 * tmp14;
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp16 * tmp19;
                    auto tmp21 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp22 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp21[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp15 - tmp23;
                    auto tmp25 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp26 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp25[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp27 = tmp20 * tmp26;
                    auto tmp28 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp29 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp28[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp30 = tmp4 * tmp29;
                    auto tmp31 = tmp27 + tmp30;
                    auto tmp34 = c10::convert<float>(tmp33);
                    auto tmp35 = at::vec::Vectorized<float>(tmp34);
                    auto tmp36 = tmp32 * tmp35;
                    auto tmp37 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp38 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp37[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp39 = tmp36 * tmp38;
                    auto tmp42 = c10::convert<float>(tmp41);
                    auto tmp43 = at::vec::Vectorized<float>(tmp42);
                    auto tmp44 = tmp40 * tmp43;
                    auto tmp45 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp46 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp45[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp47 = tmp44 * tmp46;
                    auto tmp48 = tmp39 - tmp47;
                    auto tmp49 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp50 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp49[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp51 = tmp44 * tmp50;
                    auto tmp52 =
                    [&]
                    {
                        __at_align__ std::array<int, 8> tmpbuf;
                        tmp12.store(tmpbuf.data());
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp53 =
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp52[x0_inner]))];
                        }
                        return at::vec::Vectorized<float>::loadu(tmpbuf.data(), 8);
                    }
                    ()
                    ;
                    auto tmp54 = tmp36 * tmp53;
                    auto tmp55 = tmp51 + tmp54;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp24.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp31.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp48.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                    [&]
                    {
                        __at_align__ std::array<float, 8> tmpbuf;
                        tmp55.store(tmpbuf.data());
                        #pragma unroll 8
                        for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                        {
                            out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner];
                        }
                    }
                    ()
                    ;
                }
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer(ks0, 8L))); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>((2L*x2) + (48L*x1))];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp10 = in_ptr0[static_cast<long>(1L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp11 = in_ptr1[static_cast<long>(1L + (2L*x2) + (48L*x1))];
                    auto tmp20 = in_ptr0[static_cast<long>(288L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp21 = in_ptr1[static_cast<long>(288L + (2L*x2) + (48L*x1))];
                    auto tmp25 = in_ptr0[static_cast<long>(289L + (2L*x2) + (48L*x1) + (864L*x0))];
                    auto tmp26 = in_ptr1[static_cast<long>(289L + (2L*x2) + (48L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                    auto tmp5 = decltype(tmp4)(tmp4 + 2048);
                    auto tmp6 = tmp4 < 0;
                    auto tmp7 = tmp6 ? tmp5 : tmp4;
                    AOTI_TORCH_CHECK((0 <= tmp7) & (tmp7 < 2048L), "index out of bounds: 0 <= tmp7 < 2048L")
                    auto tmp8 = in_ptr3[static_cast<long>((2L*x2) + (48L*tmp7))];
                    auto tmp9 = decltype(tmp3)(tmp3 * tmp8);
                    auto tmp12 = c10::convert<float>(tmp11);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = in_ptr3[static_cast<long>(1L + (2L*x2) + (48L*tmp7))];
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 - tmp15);
                    auto tmp17 = decltype(tmp13)(tmp13 * tmp8);
                    auto tmp18 = decltype(tmp3)(tmp3 * tmp14);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp22 = c10::convert<float>(tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 * tmp8);
                    auto tmp27 = c10::convert<float>(tmp26);
                    auto tmp28 = decltype(tmp25)(tmp25 * tmp27);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp14);
                    auto tmp30 = decltype(tmp24)(tmp24 - tmp29);
                    auto tmp31 = decltype(tmp28)(tmp28 * tmp8);
                    auto tmp32 = decltype(tmp23)(tmp23 * tmp14);
                    auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                    out_ptr0[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp16;
                    out_ptr1[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp19;
                    out_ptr2[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp30;
                    out_ptr3[static_cast<long>((2L*x2) + (48L*x1) + (288L*x0))] = tmp33;
                }
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x0)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (48L*x1) + (288L*x0)), 8);
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(576L + x2 + (48L*x1) + (864L*x0)), 8);
                            auto tmp6 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(576L + x2 + (48L*x1)), 8);
                            auto tmp1 = decltype(tmp0)(tmp0 + 352);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            AOTI_TORCH_CHECK((0 <= tmp3) & (tmp3 < 352L), "index out of bounds: 0 <= tmp3 < 352L")
                            auto tmp7 = at::vec::convert<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                            tmp8.store(out_ptr5 + static_cast<long>(x2 + (48L*tmp3) + (16896L*x1)));
                        }
                    }
                }
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_32(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(82944L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_33(const float* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const float* in_ptr3,
                       const signed char* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp6 = tmp5 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp6 = out_ptr0[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                auto tmp3 = at::vec::convert<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 + tmp4;
                auto tmp7 = static_cast<float>(288.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                tmp15.store(out_ptr1 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_34(const signed char* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr0 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_mul_silu_35(float* in_out_ptr0,
                       const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const bfloat16* in_ptr2,
                       const signed char* in_ptr3,
                       float* out_ptr0,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1), 8);
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)), 8);
                auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr2 + static_cast<long>(x1), 8);
                auto tmp2 = at::vec::convert<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                auto tmp5 = tmp3 * tmp4;
                auto tmp8 = at::vec::convert<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = tmp5 * tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(221184L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused__to_copy_add_mean_mul_rsqrt_36(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bfloat16* in_ptr1,
                       const float* in_ptr2,
                       const bfloat16* in_ptr3,
                       const float* in_ptr4,
                       const signed char* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp2 = at::vec::Vectorized<bfloat16>::loadu(in_ptr1 + static_cast<long>(x1), 8);
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)), 8);
                    auto tmp7 = at::vec::Vectorized<bfloat16>::loadu(in_ptr3 + static_cast<long>(x1), 8);
                    auto tmp3 = at::vec::convert<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    auto tmp8 = at::vec::convert<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp5 + tmp9;
                    auto tmp11 = tmp10 * tmp10;
                    tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp11;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)), 8);
                auto tmp1 = out_ptr0[static_cast<long>(x0)];
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1), 8);
                auto tmp2 = static_cast<float>(288.0);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp6 = 1 / std::sqrt(tmp5);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp0 * tmp7;
                auto tmp10 = tmp8 * tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (288L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216000L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<signed char>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                auto tmp1 = at::vec::convert<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}

#include "/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd/cndd7co72iqjtof53ikp4l7yibmqrbjkni3cu6xj5p7hywloe5yg.h"
extern "C" void cpp_fused_mul_37(float* in_out_ptr0,
                       const bfloat16* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
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
                       float* out_ptr11,
                       const long ks0)
{
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32000L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32000L*x0)), 8);
                    auto tmp1 = at::vec::Vectorized<bfloat16>::loadu(in_ptr0 + static_cast<long>(x1), 8);
                    auto tmp2 = at::vec::convert<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (32000L*x0)));
                }
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr6 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr8 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(101376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0), 8);
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
    }
}
namespace torch {
namespace aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
};
}  // namespace

AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(2, 1, 90, device_str, cubin_dir) {
    inputs_info_[0].name = "arg90_1";
    inputs_info_[1].name = "arg91_1";
    constants_info_[0].name = "L__self___layers_0_attention_norm_weight";
    constants_info_[0].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 1152;
    constants_info_[0].from_folded = false;
    constants_info_[0].shape = {288};
    constants_info_[0].stride = {1};
    constants_info_[0].original_fqn = "L__self___layers_0_attention_norm_weight";
    constants_info_[1].name = "L__self___layers_0_ffn_norm_weight";
    constants_info_[1].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 1152;
    constants_info_[1].from_folded = false;
    constants_info_[1].shape = {288};
    constants_info_[1].stride = {1};
    constants_info_[1].original_fqn = "L__self___layers_0_ffn_norm_weight";
    constants_info_[2].name = "L__self___layers_1_attention_norm_weight";
    constants_info_[2].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 1152;
    constants_info_[2].from_folded = false;
    constants_info_[2].shape = {288};
    constants_info_[2].stride = {1};
    constants_info_[2].original_fqn = "L__self___layers_1_attention_norm_weight";
    constants_info_[3].name = "L__self___layers_1_ffn_norm_weight";
    constants_info_[3].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[3].offset = 0;
    constants_info_[3].data_size = 1152;
    constants_info_[3].from_folded = false;
    constants_info_[3].shape = {288};
    constants_info_[3].stride = {1};
    constants_info_[3].original_fqn = "L__self___layers_1_ffn_norm_weight";
    constants_info_[4].name = "L__self___layers_2_attention_norm_weight";
    constants_info_[4].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[4].offset = 0;
    constants_info_[4].data_size = 1152;
    constants_info_[4].from_folded = false;
    constants_info_[4].shape = {288};
    constants_info_[4].stride = {1};
    constants_info_[4].original_fqn = "L__self___layers_2_attention_norm_weight";
    constants_info_[5].name = "L__self___layers_2_ffn_norm_weight";
    constants_info_[5].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[5].offset = 0;
    constants_info_[5].data_size = 1152;
    constants_info_[5].from_folded = false;
    constants_info_[5].shape = {288};
    constants_info_[5].stride = {1};
    constants_info_[5].original_fqn = "L__self___layers_2_ffn_norm_weight";
    constants_info_[6].name = "L__self___layers_3_attention_norm_weight";
    constants_info_[6].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[6].offset = 0;
    constants_info_[6].data_size = 1152;
    constants_info_[6].from_folded = false;
    constants_info_[6].shape = {288};
    constants_info_[6].stride = {1};
    constants_info_[6].original_fqn = "L__self___layers_3_attention_norm_weight";
    constants_info_[7].name = "L__self___layers_3_ffn_norm_weight";
    constants_info_[7].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[7].offset = 0;
    constants_info_[7].data_size = 1152;
    constants_info_[7].from_folded = false;
    constants_info_[7].shape = {288};
    constants_info_[7].stride = {1};
    constants_info_[7].original_fqn = "L__self___layers_3_ffn_norm_weight";
    constants_info_[8].name = "L__self___layers_4_attention_norm_weight";
    constants_info_[8].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[8].offset = 0;
    constants_info_[8].data_size = 1152;
    constants_info_[8].from_folded = false;
    constants_info_[8].shape = {288};
    constants_info_[8].stride = {1};
    constants_info_[8].original_fqn = "L__self___layers_4_attention_norm_weight";
    constants_info_[9].name = "L__self___layers_4_ffn_norm_weight";
    constants_info_[9].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[9].offset = 0;
    constants_info_[9].data_size = 1152;
    constants_info_[9].from_folded = false;
    constants_info_[9].shape = {288};
    constants_info_[9].stride = {1};
    constants_info_[9].original_fqn = "L__self___layers_4_ffn_norm_weight";
    constants_info_[10].name = "L__self___layers_5_attention_norm_weight";
    constants_info_[10].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[10].offset = 0;
    constants_info_[10].data_size = 1152;
    constants_info_[10].from_folded = false;
    constants_info_[10].shape = {288};
    constants_info_[10].stride = {1};
    constants_info_[10].original_fqn = "L__self___layers_5_attention_norm_weight";
    constants_info_[11].name = "L__self___layers_5_ffn_norm_weight";
    constants_info_[11].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[11].offset = 0;
    constants_info_[11].data_size = 1152;
    constants_info_[11].from_folded = false;
    constants_info_[11].shape = {288};
    constants_info_[11].stride = {1};
    constants_info_[11].original_fqn = "L__self___layers_5_ffn_norm_weight";
    constants_info_[12].name = "L__self___norm_weight";
    constants_info_[12].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[12].offset = 0;
    constants_info_[12].data_size = 1152;
    constants_info_[12].from_folded = false;
    constants_info_[12].shape = {288};
    constants_info_[12].stride = {1};
    constants_info_[12].original_fqn = "norm.weight";
    constants_info_[13].name = "L__self___tok_embeddings_weight";
    constants_info_[13].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[13].offset = 0;
    constants_info_[13].data_size = 36864000;
    constants_info_[13].from_folded = false;
    constants_info_[13].shape = {32000, 288};
    constants_info_[13].stride = {288, 1};
    constants_info_[13].original_fqn = "tok_embeddings.weight";
    constants_info_[14].name = "L__self___freqs_cis";
    constants_info_[14].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[14].offset = 0;
    constants_info_[14].data_size = 393216;
    constants_info_[14].from_folded = false;
    constants_info_[14].shape = {2048, 24, 2};
    constants_info_[14].stride = {48, 2, 1};
    constants_info_[14].original_fqn = "freqs_cis";
    constants_info_[15].name = "L__self___causal_mask";
    constants_info_[15].dtype = static_cast<int32_t>(at::kBool);
    constants_info_[15].offset = 0;
    constants_info_[15].data_size = 123904;
    constants_info_[15].from_folded = false;
    constants_info_[15].shape = {352, 352};
    constants_info_[15].stride = {352, 1};
    constants_info_[15].original_fqn = "causal_mask";
    constants_info_[16].name = "L__self___layers_0_attention_wqkv_scales";
    constants_info_[16].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[16].offset = 0;
    constants_info_[16].data_size = 1728;
    constants_info_[16].from_folded = false;
    constants_info_[16].shape = {864};
    constants_info_[16].stride = {1};
    constants_info_[16].original_fqn = "L__self___layers_0_attention_wqkv_scales";
    constants_info_[17].name = "L__self___layers_0_attention_wqkv_weight";
    constants_info_[17].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[17].offset = 0;
    constants_info_[17].data_size = 248832;
    constants_info_[17].from_folded = false;
    constants_info_[17].shape = {864, 288};
    constants_info_[17].stride = {288, 1};
    constants_info_[17].original_fqn = "L__self___layers_0_attention_wqkv_weight";
    constants_info_[18].name = "L__self___layers_0_attention_kv_cache_k_cache";
    constants_info_[18].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[18].offset = 0;
    constants_info_[18].data_size = 405504;
    constants_info_[18].from_folded = false;
    constants_info_[18].shape = {1, 6, 352, 48};
    constants_info_[18].stride = {101376, 16896, 48, 1};
    constants_info_[18].original_fqn = "L__self___layers_0_attention_kv_cache_k_cache";
    constants_info_[19].name = "L__self___layers_0_attention_kv_cache_v_cache";
    constants_info_[19].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[19].offset = 0;
    constants_info_[19].data_size = 405504;
    constants_info_[19].from_folded = false;
    constants_info_[19].shape = {1, 6, 352, 48};
    constants_info_[19].stride = {101376, 16896, 48, 1};
    constants_info_[19].original_fqn = "L__self___layers_0_attention_kv_cache_v_cache";
    constants_info_[20].name = "L__self___layers_0_attention_wo_scales";
    constants_info_[20].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[20].offset = 0;
    constants_info_[20].data_size = 576;
    constants_info_[20].from_folded = false;
    constants_info_[20].shape = {288};
    constants_info_[20].stride = {1};
    constants_info_[20].original_fqn = "L__self___layers_0_attention_wo_scales";
    constants_info_[21].name = "L__self___layers_0_attention_wo_weight";
    constants_info_[21].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[21].offset = 0;
    constants_info_[21].data_size = 82944;
    constants_info_[21].from_folded = false;
    constants_info_[21].shape = {288, 288};
    constants_info_[21].stride = {288, 1};
    constants_info_[21].original_fqn = "L__self___layers_0_attention_wo_weight";
    constants_info_[22].name = "L__self___layers_0_feed_forward_w1_scales";
    constants_info_[22].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[22].offset = 0;
    constants_info_[22].data_size = 1536;
    constants_info_[22].from_folded = false;
    constants_info_[22].shape = {768};
    constants_info_[22].stride = {1};
    constants_info_[22].original_fqn = "L__self___layers_0_feed_forward_w1_scales";
    constants_info_[23].name = "L__self___layers_0_feed_forward_w1_weight";
    constants_info_[23].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[23].offset = 0;
    constants_info_[23].data_size = 221184;
    constants_info_[23].from_folded = false;
    constants_info_[23].shape = {768, 288};
    constants_info_[23].stride = {288, 1};
    constants_info_[23].original_fqn = "L__self___layers_0_feed_forward_w1_weight";
    constants_info_[24].name = "L__self___layers_0_feed_forward_w3_scales";
    constants_info_[24].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[24].offset = 0;
    constants_info_[24].data_size = 1536;
    constants_info_[24].from_folded = false;
    constants_info_[24].shape = {768};
    constants_info_[24].stride = {1};
    constants_info_[24].original_fqn = "L__self___layers_0_feed_forward_w3_scales";
    constants_info_[25].name = "L__self___layers_0_feed_forward_w3_weight";
    constants_info_[25].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[25].offset = 0;
    constants_info_[25].data_size = 221184;
    constants_info_[25].from_folded = false;
    constants_info_[25].shape = {768, 288};
    constants_info_[25].stride = {288, 1};
    constants_info_[25].original_fqn = "L__self___layers_0_feed_forward_w3_weight";
    constants_info_[26].name = "L__self___layers_0_feed_forward_w2_scales";
    constants_info_[26].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[26].offset = 0;
    constants_info_[26].data_size = 576;
    constants_info_[26].from_folded = false;
    constants_info_[26].shape = {288};
    constants_info_[26].stride = {1};
    constants_info_[26].original_fqn = "L__self___layers_0_feed_forward_w2_scales";
    constants_info_[27].name = "L__self___layers_0_feed_forward_w2_weight";
    constants_info_[27].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[27].offset = 0;
    constants_info_[27].data_size = 221184;
    constants_info_[27].from_folded = false;
    constants_info_[27].shape = {288, 768};
    constants_info_[27].stride = {768, 1};
    constants_info_[27].original_fqn = "L__self___layers_0_feed_forward_w2_weight";
    constants_info_[28].name = "L__self___layers_1_attention_wqkv_scales";
    constants_info_[28].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[28].offset = 0;
    constants_info_[28].data_size = 1728;
    constants_info_[28].from_folded = false;
    constants_info_[28].shape = {864};
    constants_info_[28].stride = {1};
    constants_info_[28].original_fqn = "L__self___layers_1_attention_wqkv_scales";
    constants_info_[29].name = "L__self___layers_1_attention_wqkv_weight";
    constants_info_[29].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[29].offset = 0;
    constants_info_[29].data_size = 248832;
    constants_info_[29].from_folded = false;
    constants_info_[29].shape = {864, 288};
    constants_info_[29].stride = {288, 1};
    constants_info_[29].original_fqn = "L__self___layers_1_attention_wqkv_weight";
    constants_info_[30].name = "L__self___layers_1_attention_kv_cache_k_cache";
    constants_info_[30].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[30].offset = 0;
    constants_info_[30].data_size = 405504;
    constants_info_[30].from_folded = false;
    constants_info_[30].shape = {1, 6, 352, 48};
    constants_info_[30].stride = {101376, 16896, 48, 1};
    constants_info_[30].original_fqn = "L__self___layers_1_attention_kv_cache_k_cache";
    constants_info_[31].name = "L__self___layers_1_attention_kv_cache_v_cache";
    constants_info_[31].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[31].offset = 0;
    constants_info_[31].data_size = 405504;
    constants_info_[31].from_folded = false;
    constants_info_[31].shape = {1, 6, 352, 48};
    constants_info_[31].stride = {101376, 16896, 48, 1};
    constants_info_[31].original_fqn = "L__self___layers_1_attention_kv_cache_v_cache";
    constants_info_[32].name = "L__self___layers_1_attention_wo_scales";
    constants_info_[32].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[32].offset = 0;
    constants_info_[32].data_size = 576;
    constants_info_[32].from_folded = false;
    constants_info_[32].shape = {288};
    constants_info_[32].stride = {1};
    constants_info_[32].original_fqn = "L__self___layers_1_attention_wo_scales";
    constants_info_[33].name = "L__self___layers_1_attention_wo_weight";
    constants_info_[33].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[33].offset = 0;
    constants_info_[33].data_size = 82944;
    constants_info_[33].from_folded = false;
    constants_info_[33].shape = {288, 288};
    constants_info_[33].stride = {288, 1};
    constants_info_[33].original_fqn = "L__self___layers_1_attention_wo_weight";
    constants_info_[34].name = "L__self___layers_1_feed_forward_w1_scales";
    constants_info_[34].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[34].offset = 0;
    constants_info_[34].data_size = 1536;
    constants_info_[34].from_folded = false;
    constants_info_[34].shape = {768};
    constants_info_[34].stride = {1};
    constants_info_[34].original_fqn = "L__self___layers_1_feed_forward_w1_scales";
    constants_info_[35].name = "L__self___layers_1_feed_forward_w1_weight";
    constants_info_[35].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[35].offset = 0;
    constants_info_[35].data_size = 221184;
    constants_info_[35].from_folded = false;
    constants_info_[35].shape = {768, 288};
    constants_info_[35].stride = {288, 1};
    constants_info_[35].original_fqn = "L__self___layers_1_feed_forward_w1_weight";
    constants_info_[36].name = "L__self___layers_1_feed_forward_w3_scales";
    constants_info_[36].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[36].offset = 0;
    constants_info_[36].data_size = 1536;
    constants_info_[36].from_folded = false;
    constants_info_[36].shape = {768};
    constants_info_[36].stride = {1};
    constants_info_[36].original_fqn = "L__self___layers_1_feed_forward_w3_scales";
    constants_info_[37].name = "L__self___layers_1_feed_forward_w3_weight";
    constants_info_[37].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[37].offset = 0;
    constants_info_[37].data_size = 221184;
    constants_info_[37].from_folded = false;
    constants_info_[37].shape = {768, 288};
    constants_info_[37].stride = {288, 1};
    constants_info_[37].original_fqn = "L__self___layers_1_feed_forward_w3_weight";
    constants_info_[38].name = "L__self___layers_1_feed_forward_w2_scales";
    constants_info_[38].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[38].offset = 0;
    constants_info_[38].data_size = 576;
    constants_info_[38].from_folded = false;
    constants_info_[38].shape = {288};
    constants_info_[38].stride = {1};
    constants_info_[38].original_fqn = "L__self___layers_1_feed_forward_w2_scales";
    constants_info_[39].name = "L__self___layers_1_feed_forward_w2_weight";
    constants_info_[39].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[39].offset = 0;
    constants_info_[39].data_size = 221184;
    constants_info_[39].from_folded = false;
    constants_info_[39].shape = {288, 768};
    constants_info_[39].stride = {768, 1};
    constants_info_[39].original_fqn = "L__self___layers_1_feed_forward_w2_weight";
    constants_info_[40].name = "L__self___layers_2_attention_wqkv_scales";
    constants_info_[40].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[40].offset = 0;
    constants_info_[40].data_size = 1728;
    constants_info_[40].from_folded = false;
    constants_info_[40].shape = {864};
    constants_info_[40].stride = {1};
    constants_info_[40].original_fqn = "L__self___layers_2_attention_wqkv_scales";
    constants_info_[41].name = "L__self___layers_2_attention_wqkv_weight";
    constants_info_[41].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[41].offset = 0;
    constants_info_[41].data_size = 248832;
    constants_info_[41].from_folded = false;
    constants_info_[41].shape = {864, 288};
    constants_info_[41].stride = {288, 1};
    constants_info_[41].original_fqn = "L__self___layers_2_attention_wqkv_weight";
    constants_info_[42].name = "L__self___layers_2_attention_kv_cache_k_cache";
    constants_info_[42].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[42].offset = 0;
    constants_info_[42].data_size = 405504;
    constants_info_[42].from_folded = false;
    constants_info_[42].shape = {1, 6, 352, 48};
    constants_info_[42].stride = {101376, 16896, 48, 1};
    constants_info_[42].original_fqn = "L__self___layers_2_attention_kv_cache_k_cache";
    constants_info_[43].name = "L__self___layers_2_attention_kv_cache_v_cache";
    constants_info_[43].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[43].offset = 0;
    constants_info_[43].data_size = 405504;
    constants_info_[43].from_folded = false;
    constants_info_[43].shape = {1, 6, 352, 48};
    constants_info_[43].stride = {101376, 16896, 48, 1};
    constants_info_[43].original_fqn = "L__self___layers_2_attention_kv_cache_v_cache";
    constants_info_[44].name = "L__self___layers_2_attention_wo_scales";
    constants_info_[44].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[44].offset = 0;
    constants_info_[44].data_size = 576;
    constants_info_[44].from_folded = false;
    constants_info_[44].shape = {288};
    constants_info_[44].stride = {1};
    constants_info_[44].original_fqn = "L__self___layers_2_attention_wo_scales";
    constants_info_[45].name = "L__self___layers_2_attention_wo_weight";
    constants_info_[45].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[45].offset = 0;
    constants_info_[45].data_size = 82944;
    constants_info_[45].from_folded = false;
    constants_info_[45].shape = {288, 288};
    constants_info_[45].stride = {288, 1};
    constants_info_[45].original_fqn = "L__self___layers_2_attention_wo_weight";
    constants_info_[46].name = "L__self___layers_2_feed_forward_w1_scales";
    constants_info_[46].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[46].offset = 0;
    constants_info_[46].data_size = 1536;
    constants_info_[46].from_folded = false;
    constants_info_[46].shape = {768};
    constants_info_[46].stride = {1};
    constants_info_[46].original_fqn = "L__self___layers_2_feed_forward_w1_scales";
    constants_info_[47].name = "L__self___layers_2_feed_forward_w1_weight";
    constants_info_[47].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[47].offset = 0;
    constants_info_[47].data_size = 221184;
    constants_info_[47].from_folded = false;
    constants_info_[47].shape = {768, 288};
    constants_info_[47].stride = {288, 1};
    constants_info_[47].original_fqn = "L__self___layers_2_feed_forward_w1_weight";
    constants_info_[48].name = "L__self___layers_2_feed_forward_w3_scales";
    constants_info_[48].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[48].offset = 0;
    constants_info_[48].data_size = 1536;
    constants_info_[48].from_folded = false;
    constants_info_[48].shape = {768};
    constants_info_[48].stride = {1};
    constants_info_[48].original_fqn = "L__self___layers_2_feed_forward_w3_scales";
    constants_info_[49].name = "L__self___layers_2_feed_forward_w3_weight";
    constants_info_[49].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[49].offset = 0;
    constants_info_[49].data_size = 221184;
    constants_info_[49].from_folded = false;
    constants_info_[49].shape = {768, 288};
    constants_info_[49].stride = {288, 1};
    constants_info_[49].original_fqn = "L__self___layers_2_feed_forward_w3_weight";
    constants_info_[50].name = "L__self___layers_2_feed_forward_w2_scales";
    constants_info_[50].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[50].offset = 0;
    constants_info_[50].data_size = 576;
    constants_info_[50].from_folded = false;
    constants_info_[50].shape = {288};
    constants_info_[50].stride = {1};
    constants_info_[50].original_fqn = "L__self___layers_2_feed_forward_w2_scales";
    constants_info_[51].name = "L__self___layers_2_feed_forward_w2_weight";
    constants_info_[51].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[51].offset = 0;
    constants_info_[51].data_size = 221184;
    constants_info_[51].from_folded = false;
    constants_info_[51].shape = {288, 768};
    constants_info_[51].stride = {768, 1};
    constants_info_[51].original_fqn = "L__self___layers_2_feed_forward_w2_weight";
    constants_info_[52].name = "L__self___layers_3_attention_wqkv_scales";
    constants_info_[52].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[52].offset = 0;
    constants_info_[52].data_size = 1728;
    constants_info_[52].from_folded = false;
    constants_info_[52].shape = {864};
    constants_info_[52].stride = {1};
    constants_info_[52].original_fqn = "L__self___layers_3_attention_wqkv_scales";
    constants_info_[53].name = "L__self___layers_3_attention_wqkv_weight";
    constants_info_[53].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[53].offset = 0;
    constants_info_[53].data_size = 248832;
    constants_info_[53].from_folded = false;
    constants_info_[53].shape = {864, 288};
    constants_info_[53].stride = {288, 1};
    constants_info_[53].original_fqn = "L__self___layers_3_attention_wqkv_weight";
    constants_info_[54].name = "L__self___layers_3_attention_kv_cache_k_cache";
    constants_info_[54].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[54].offset = 0;
    constants_info_[54].data_size = 405504;
    constants_info_[54].from_folded = false;
    constants_info_[54].shape = {1, 6, 352, 48};
    constants_info_[54].stride = {101376, 16896, 48, 1};
    constants_info_[54].original_fqn = "L__self___layers_3_attention_kv_cache_k_cache";
    constants_info_[55].name = "L__self___layers_3_attention_kv_cache_v_cache";
    constants_info_[55].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[55].offset = 0;
    constants_info_[55].data_size = 405504;
    constants_info_[55].from_folded = false;
    constants_info_[55].shape = {1, 6, 352, 48};
    constants_info_[55].stride = {101376, 16896, 48, 1};
    constants_info_[55].original_fqn = "L__self___layers_3_attention_kv_cache_v_cache";
    constants_info_[56].name = "L__self___layers_3_attention_wo_scales";
    constants_info_[56].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[56].offset = 0;
    constants_info_[56].data_size = 576;
    constants_info_[56].from_folded = false;
    constants_info_[56].shape = {288};
    constants_info_[56].stride = {1};
    constants_info_[56].original_fqn = "L__self___layers_3_attention_wo_scales";
    constants_info_[57].name = "L__self___layers_3_attention_wo_weight";
    constants_info_[57].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[57].offset = 0;
    constants_info_[57].data_size = 82944;
    constants_info_[57].from_folded = false;
    constants_info_[57].shape = {288, 288};
    constants_info_[57].stride = {288, 1};
    constants_info_[57].original_fqn = "L__self___layers_3_attention_wo_weight";
    constants_info_[58].name = "L__self___layers_3_feed_forward_w1_scales";
    constants_info_[58].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[58].offset = 0;
    constants_info_[58].data_size = 1536;
    constants_info_[58].from_folded = false;
    constants_info_[58].shape = {768};
    constants_info_[58].stride = {1};
    constants_info_[58].original_fqn = "L__self___layers_3_feed_forward_w1_scales";
    constants_info_[59].name = "L__self___layers_3_feed_forward_w1_weight";
    constants_info_[59].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[59].offset = 0;
    constants_info_[59].data_size = 221184;
    constants_info_[59].from_folded = false;
    constants_info_[59].shape = {768, 288};
    constants_info_[59].stride = {288, 1};
    constants_info_[59].original_fqn = "L__self___layers_3_feed_forward_w1_weight";
    constants_info_[60].name = "L__self___layers_3_feed_forward_w3_scales";
    constants_info_[60].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[60].offset = 0;
    constants_info_[60].data_size = 1536;
    constants_info_[60].from_folded = false;
    constants_info_[60].shape = {768};
    constants_info_[60].stride = {1};
    constants_info_[60].original_fqn = "L__self___layers_3_feed_forward_w3_scales";
    constants_info_[61].name = "L__self___layers_3_feed_forward_w3_weight";
    constants_info_[61].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[61].offset = 0;
    constants_info_[61].data_size = 221184;
    constants_info_[61].from_folded = false;
    constants_info_[61].shape = {768, 288};
    constants_info_[61].stride = {288, 1};
    constants_info_[61].original_fqn = "L__self___layers_3_feed_forward_w3_weight";
    constants_info_[62].name = "L__self___layers_3_feed_forward_w2_scales";
    constants_info_[62].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[62].offset = 0;
    constants_info_[62].data_size = 576;
    constants_info_[62].from_folded = false;
    constants_info_[62].shape = {288};
    constants_info_[62].stride = {1};
    constants_info_[62].original_fqn = "L__self___layers_3_feed_forward_w2_scales";
    constants_info_[63].name = "L__self___layers_3_feed_forward_w2_weight";
    constants_info_[63].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[63].offset = 0;
    constants_info_[63].data_size = 221184;
    constants_info_[63].from_folded = false;
    constants_info_[63].shape = {288, 768};
    constants_info_[63].stride = {768, 1};
    constants_info_[63].original_fqn = "L__self___layers_3_feed_forward_w2_weight";
    constants_info_[64].name = "L__self___layers_4_attention_wqkv_scales";
    constants_info_[64].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[64].offset = 0;
    constants_info_[64].data_size = 1728;
    constants_info_[64].from_folded = false;
    constants_info_[64].shape = {864};
    constants_info_[64].stride = {1};
    constants_info_[64].original_fqn = "L__self___layers_4_attention_wqkv_scales";
    constants_info_[65].name = "L__self___layers_4_attention_wqkv_weight";
    constants_info_[65].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[65].offset = 0;
    constants_info_[65].data_size = 248832;
    constants_info_[65].from_folded = false;
    constants_info_[65].shape = {864, 288};
    constants_info_[65].stride = {288, 1};
    constants_info_[65].original_fqn = "L__self___layers_4_attention_wqkv_weight";
    constants_info_[66].name = "L__self___layers_4_attention_kv_cache_k_cache";
    constants_info_[66].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[66].offset = 0;
    constants_info_[66].data_size = 405504;
    constants_info_[66].from_folded = false;
    constants_info_[66].shape = {1, 6, 352, 48};
    constants_info_[66].stride = {101376, 16896, 48, 1};
    constants_info_[66].original_fqn = "L__self___layers_4_attention_kv_cache_k_cache";
    constants_info_[67].name = "L__self___layers_4_attention_kv_cache_v_cache";
    constants_info_[67].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[67].offset = 0;
    constants_info_[67].data_size = 405504;
    constants_info_[67].from_folded = false;
    constants_info_[67].shape = {1, 6, 352, 48};
    constants_info_[67].stride = {101376, 16896, 48, 1};
    constants_info_[67].original_fqn = "L__self___layers_4_attention_kv_cache_v_cache";
    constants_info_[68].name = "L__self___layers_4_attention_wo_scales";
    constants_info_[68].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[68].offset = 0;
    constants_info_[68].data_size = 576;
    constants_info_[68].from_folded = false;
    constants_info_[68].shape = {288};
    constants_info_[68].stride = {1};
    constants_info_[68].original_fqn = "L__self___layers_4_attention_wo_scales";
    constants_info_[69].name = "L__self___layers_4_attention_wo_weight";
    constants_info_[69].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[69].offset = 0;
    constants_info_[69].data_size = 82944;
    constants_info_[69].from_folded = false;
    constants_info_[69].shape = {288, 288};
    constants_info_[69].stride = {288, 1};
    constants_info_[69].original_fqn = "L__self___layers_4_attention_wo_weight";
    constants_info_[70].name = "L__self___layers_4_feed_forward_w1_scales";
    constants_info_[70].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[70].offset = 0;
    constants_info_[70].data_size = 1536;
    constants_info_[70].from_folded = false;
    constants_info_[70].shape = {768};
    constants_info_[70].stride = {1};
    constants_info_[70].original_fqn = "L__self___layers_4_feed_forward_w1_scales";
    constants_info_[71].name = "L__self___layers_4_feed_forward_w1_weight";
    constants_info_[71].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[71].offset = 0;
    constants_info_[71].data_size = 221184;
    constants_info_[71].from_folded = false;
    constants_info_[71].shape = {768, 288};
    constants_info_[71].stride = {288, 1};
    constants_info_[71].original_fqn = "L__self___layers_4_feed_forward_w1_weight";
    constants_info_[72].name = "L__self___layers_4_feed_forward_w3_scales";
    constants_info_[72].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[72].offset = 0;
    constants_info_[72].data_size = 1536;
    constants_info_[72].from_folded = false;
    constants_info_[72].shape = {768};
    constants_info_[72].stride = {1};
    constants_info_[72].original_fqn = "L__self___layers_4_feed_forward_w3_scales";
    constants_info_[73].name = "L__self___layers_4_feed_forward_w3_weight";
    constants_info_[73].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[73].offset = 0;
    constants_info_[73].data_size = 221184;
    constants_info_[73].from_folded = false;
    constants_info_[73].shape = {768, 288};
    constants_info_[73].stride = {288, 1};
    constants_info_[73].original_fqn = "L__self___layers_4_feed_forward_w3_weight";
    constants_info_[74].name = "L__self___layers_4_feed_forward_w2_scales";
    constants_info_[74].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[74].offset = 0;
    constants_info_[74].data_size = 576;
    constants_info_[74].from_folded = false;
    constants_info_[74].shape = {288};
    constants_info_[74].stride = {1};
    constants_info_[74].original_fqn = "L__self___layers_4_feed_forward_w2_scales";
    constants_info_[75].name = "L__self___layers_4_feed_forward_w2_weight";
    constants_info_[75].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[75].offset = 0;
    constants_info_[75].data_size = 221184;
    constants_info_[75].from_folded = false;
    constants_info_[75].shape = {288, 768};
    constants_info_[75].stride = {768, 1};
    constants_info_[75].original_fqn = "L__self___layers_4_feed_forward_w2_weight";
    constants_info_[76].name = "L__self___layers_5_attention_wqkv_scales";
    constants_info_[76].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[76].offset = 0;
    constants_info_[76].data_size = 1728;
    constants_info_[76].from_folded = false;
    constants_info_[76].shape = {864};
    constants_info_[76].stride = {1};
    constants_info_[76].original_fqn = "L__self___layers_5_attention_wqkv_scales";
    constants_info_[77].name = "L__self___layers_5_attention_wqkv_weight";
    constants_info_[77].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[77].offset = 0;
    constants_info_[77].data_size = 248832;
    constants_info_[77].from_folded = false;
    constants_info_[77].shape = {864, 288};
    constants_info_[77].stride = {288, 1};
    constants_info_[77].original_fqn = "L__self___layers_5_attention_wqkv_weight";
    constants_info_[78].name = "L__self___layers_5_attention_kv_cache_k_cache";
    constants_info_[78].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[78].offset = 0;
    constants_info_[78].data_size = 405504;
    constants_info_[78].from_folded = false;
    constants_info_[78].shape = {1, 6, 352, 48};
    constants_info_[78].stride = {101376, 16896, 48, 1};
    constants_info_[78].original_fqn = "L__self___layers_5_attention_kv_cache_k_cache";
    constants_info_[79].name = "L__self___layers_5_attention_kv_cache_v_cache";
    constants_info_[79].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[79].offset = 0;
    constants_info_[79].data_size = 405504;
    constants_info_[79].from_folded = false;
    constants_info_[79].shape = {1, 6, 352, 48};
    constants_info_[79].stride = {101376, 16896, 48, 1};
    constants_info_[79].original_fqn = "L__self___layers_5_attention_kv_cache_v_cache";
    constants_info_[80].name = "L__self___layers_5_attention_wo_scales";
    constants_info_[80].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[80].offset = 0;
    constants_info_[80].data_size = 576;
    constants_info_[80].from_folded = false;
    constants_info_[80].shape = {288};
    constants_info_[80].stride = {1};
    constants_info_[80].original_fqn = "L__self___layers_5_attention_wo_scales";
    constants_info_[81].name = "L__self___layers_5_attention_wo_weight";
    constants_info_[81].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[81].offset = 0;
    constants_info_[81].data_size = 82944;
    constants_info_[81].from_folded = false;
    constants_info_[81].shape = {288, 288};
    constants_info_[81].stride = {288, 1};
    constants_info_[81].original_fqn = "L__self___layers_5_attention_wo_weight";
    constants_info_[82].name = "L__self___layers_5_feed_forward_w1_scales";
    constants_info_[82].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[82].offset = 0;
    constants_info_[82].data_size = 1536;
    constants_info_[82].from_folded = false;
    constants_info_[82].shape = {768};
    constants_info_[82].stride = {1};
    constants_info_[82].original_fqn = "L__self___layers_5_feed_forward_w1_scales";
    constants_info_[83].name = "L__self___layers_5_feed_forward_w1_weight";
    constants_info_[83].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[83].offset = 0;
    constants_info_[83].data_size = 221184;
    constants_info_[83].from_folded = false;
    constants_info_[83].shape = {768, 288};
    constants_info_[83].stride = {288, 1};
    constants_info_[83].original_fqn = "L__self___layers_5_feed_forward_w1_weight";
    constants_info_[84].name = "L__self___layers_5_feed_forward_w3_scales";
    constants_info_[84].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[84].offset = 0;
    constants_info_[84].data_size = 1536;
    constants_info_[84].from_folded = false;
    constants_info_[84].shape = {768};
    constants_info_[84].stride = {1};
    constants_info_[84].original_fqn = "L__self___layers_5_feed_forward_w3_scales";
    constants_info_[85].name = "L__self___layers_5_feed_forward_w3_weight";
    constants_info_[85].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[85].offset = 0;
    constants_info_[85].data_size = 221184;
    constants_info_[85].from_folded = false;
    constants_info_[85].shape = {768, 288};
    constants_info_[85].stride = {288, 1};
    constants_info_[85].original_fqn = "L__self___layers_5_feed_forward_w3_weight";
    constants_info_[86].name = "L__self___layers_5_feed_forward_w2_scales";
    constants_info_[86].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[86].offset = 0;
    constants_info_[86].data_size = 576;
    constants_info_[86].from_folded = false;
    constants_info_[86].shape = {288};
    constants_info_[86].stride = {1};
    constants_info_[86].original_fqn = "L__self___layers_5_feed_forward_w2_scales";
    constants_info_[87].name = "L__self___layers_5_feed_forward_w2_weight";
    constants_info_[87].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[87].offset = 0;
    constants_info_[87].data_size = 221184;
    constants_info_[87].from_folded = false;
    constants_info_[87].shape = {288, 768};
    constants_info_[87].stride = {768, 1};
    constants_info_[87].original_fqn = "L__self___layers_5_feed_forward_w2_weight";
    constants_info_[88].name = "L__self___output_scales";
    constants_info_[88].dtype = static_cast<int32_t>(at::kBFloat16);
    constants_info_[88].offset = 0;
    constants_info_[88].data_size = 64000;
    constants_info_[88].from_folded = false;
    constants_info_[88].shape = {32000};
    constants_info_[88].stride = {1};
    constants_info_[88].original_fqn = "output.scales";
    constants_info_[89].name = "L__self___output_weight";
    constants_info_[89].dtype = static_cast<int32_t>(at::kChar);
    constants_info_[89].offset = 0;
    constants_info_[89].data_size = 9216000;
    constants_info_[89].from_folded = false;
    constants_info_[89].shape = {32000, 288};
    constants_info_[89].stride = {288, 1};
    constants_info_[89].original_fqn = "output.weight";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}]}, {\"type\": \"builtins.dict\", \"context\": \"[]\", \"children_spec\": []}]}]";
    out_spec_ = "[1, {\"type\": null, \"context\": null, \"children_spec\": []}]";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {

    auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, 2);
    auto arg90_1 = std::move(inputs[0]);
    auto arg91_1 = std::move(inputs[1]);
    auto L__self___layers_0_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(0));
    auto L__self___layers_0_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(1));
    auto L__self___layers_1_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(2));
    auto L__self___layers_1_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(3));
    auto L__self___layers_2_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(4));
    auto L__self___layers_2_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(5));
    auto L__self___layers_3_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(6));
    auto L__self___layers_3_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(7));
    auto L__self___layers_4_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(8));
    auto L__self___layers_4_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(9));
    auto L__self___layers_5_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(10));
    auto L__self___layers_5_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(11));
    auto L__self___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(12));
    auto L__self___tok_embeddings_weight = *tensor_handle_to_tensor_pointer(constants_->at(13));
    auto L__self___freqs_cis = *tensor_handle_to_tensor_pointer(constants_->at(14));
    auto L__self___causal_mask = *tensor_handle_to_tensor_pointer(constants_->at(15));
    auto L__self___layers_0_attention_wqkv_scales = *tensor_handle_to_tensor_pointer(constants_->at(16));
    auto L__self___layers_0_attention_wqkv_weight = *tensor_handle_to_tensor_pointer(constants_->at(17));
    auto L__self___layers_0_attention_kv_cache_k_cache = *tensor_handle_to_tensor_pointer(constants_->at(18));
    auto L__self___layers_0_attention_kv_cache_v_cache = *tensor_handle_to_tensor_pointer(constants_->at(19));
    auto L__self___layers_0_attention_wo_scales = *tensor_handle_to_tensor_pointer(constants_->at(20));
    auto L__self___layers_0_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(21));
    auto L__self___layers_0_feed_forward_w1_scales = *tensor_handle_to_tensor_pointer(constants_->at(22));
    auto L__self___layers_0_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(23));
    auto L__self___layers_0_feed_forward_w3_scales = *tensor_handle_to_tensor_pointer(constants_->at(24));
    auto L__self___layers_0_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(25));
    auto L__self___layers_0_feed_forward_w2_scales = *tensor_handle_to_tensor_pointer(constants_->at(26));
    auto L__self___layers_0_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(27));
    auto L__self___layers_1_attention_wqkv_scales = *tensor_handle_to_tensor_pointer(constants_->at(28));
    auto L__self___layers_1_attention_wqkv_weight = *tensor_handle_to_tensor_pointer(constants_->at(29));
    auto L__self___layers_1_attention_kv_cache_k_cache = *tensor_handle_to_tensor_pointer(constants_->at(30));
    auto L__self___layers_1_attention_kv_cache_v_cache = *tensor_handle_to_tensor_pointer(constants_->at(31));
    auto L__self___layers_1_attention_wo_scales = *tensor_handle_to_tensor_pointer(constants_->at(32));
    auto L__self___layers_1_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(33));
    auto L__self___layers_1_feed_forward_w1_scales = *tensor_handle_to_tensor_pointer(constants_->at(34));
    auto L__self___layers_1_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(35));
    auto L__self___layers_1_feed_forward_w3_scales = *tensor_handle_to_tensor_pointer(constants_->at(36));
    auto L__self___layers_1_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(37));
    auto L__self___layers_1_feed_forward_w2_scales = *tensor_handle_to_tensor_pointer(constants_->at(38));
    auto L__self___layers_1_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(39));
    auto L__self___layers_2_attention_wqkv_scales = *tensor_handle_to_tensor_pointer(constants_->at(40));
    auto L__self___layers_2_attention_wqkv_weight = *tensor_handle_to_tensor_pointer(constants_->at(41));
    auto L__self___layers_2_attention_kv_cache_k_cache = *tensor_handle_to_tensor_pointer(constants_->at(42));
    auto L__self___layers_2_attention_kv_cache_v_cache = *tensor_handle_to_tensor_pointer(constants_->at(43));
    auto L__self___layers_2_attention_wo_scales = *tensor_handle_to_tensor_pointer(constants_->at(44));
    auto L__self___layers_2_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(45));
    auto L__self___layers_2_feed_forward_w1_scales = *tensor_handle_to_tensor_pointer(constants_->at(46));
    auto L__self___layers_2_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(47));
    auto L__self___layers_2_feed_forward_w3_scales = *tensor_handle_to_tensor_pointer(constants_->at(48));
    auto L__self___layers_2_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(49));
    auto L__self___layers_2_feed_forward_w2_scales = *tensor_handle_to_tensor_pointer(constants_->at(50));
    auto L__self___layers_2_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(51));
    auto L__self___layers_3_attention_wqkv_scales = *tensor_handle_to_tensor_pointer(constants_->at(52));
    auto L__self___layers_3_attention_wqkv_weight = *tensor_handle_to_tensor_pointer(constants_->at(53));
    auto L__self___layers_3_attention_kv_cache_k_cache = *tensor_handle_to_tensor_pointer(constants_->at(54));
    auto L__self___layers_3_attention_kv_cache_v_cache = *tensor_handle_to_tensor_pointer(constants_->at(55));
    auto L__self___layers_3_attention_wo_scales = *tensor_handle_to_tensor_pointer(constants_->at(56));
    auto L__self___layers_3_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(57));
    auto L__self___layers_3_feed_forward_w1_scales = *tensor_handle_to_tensor_pointer(constants_->at(58));
    auto L__self___layers_3_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(59));
    auto L__self___layers_3_feed_forward_w3_scales = *tensor_handle_to_tensor_pointer(constants_->at(60));
    auto L__self___layers_3_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(61));
    auto L__self___layers_3_feed_forward_w2_scales = *tensor_handle_to_tensor_pointer(constants_->at(62));
    auto L__self___layers_3_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(63));
    auto L__self___layers_4_attention_wqkv_scales = *tensor_handle_to_tensor_pointer(constants_->at(64));
    auto L__self___layers_4_attention_wqkv_weight = *tensor_handle_to_tensor_pointer(constants_->at(65));
    auto L__self___layers_4_attention_kv_cache_k_cache = *tensor_handle_to_tensor_pointer(constants_->at(66));
    auto L__self___layers_4_attention_kv_cache_v_cache = *tensor_handle_to_tensor_pointer(constants_->at(67));
    auto L__self___layers_4_attention_wo_scales = *tensor_handle_to_tensor_pointer(constants_->at(68));
    auto L__self___layers_4_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(69));
    auto L__self___layers_4_feed_forward_w1_scales = *tensor_handle_to_tensor_pointer(constants_->at(70));
    auto L__self___layers_4_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(71));
    auto L__self___layers_4_feed_forward_w3_scales = *tensor_handle_to_tensor_pointer(constants_->at(72));
    auto L__self___layers_4_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(73));
    auto L__self___layers_4_feed_forward_w2_scales = *tensor_handle_to_tensor_pointer(constants_->at(74));
    auto L__self___layers_4_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(75));
    auto L__self___layers_5_attention_wqkv_scales = *tensor_handle_to_tensor_pointer(constants_->at(76));
    auto L__self___layers_5_attention_wqkv_weight = *tensor_handle_to_tensor_pointer(constants_->at(77));
    auto L__self___layers_5_attention_kv_cache_k_cache = *tensor_handle_to_tensor_pointer(constants_->at(78));
    auto L__self___layers_5_attention_kv_cache_v_cache = *tensor_handle_to_tensor_pointer(constants_->at(79));
    auto L__self___layers_5_attention_wo_scales = *tensor_handle_to_tensor_pointer(constants_->at(80));
    auto L__self___layers_5_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(81));
    auto L__self___layers_5_feed_forward_w1_scales = *tensor_handle_to_tensor_pointer(constants_->at(82));
    auto L__self___layers_5_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(83));
    auto L__self___layers_5_feed_forward_w3_scales = *tensor_handle_to_tensor_pointer(constants_->at(84));
    auto L__self___layers_5_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(85));
    auto L__self___layers_5_feed_forward_w2_scales = *tensor_handle_to_tensor_pointer(constants_->at(86));
    auto L__self___layers_5_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(87));
    auto L__self___output_scales = *tensor_handle_to_tensor_pointer(constants_->at(88));
    auto L__self___output_weight = *tensor_handle_to_tensor_pointer(constants_->at(89));
    auto arg90_1_size = arg90_1.sizes();
    auto s0 = arg90_1_size[1];
    inputs.clear();
    auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
    at::Tensor buf0 = at::detail::empty_strided_cpu({1L, s0, 1L}, {s0, 1L, s0}, at::kFloat);
    at::Tensor buf1 = at::detail::empty_strided_cpu({1L, s0, 288L}, {288L*s0, 288L, 1L}, at::kFloat);
    at::Tensor buf2 = at::detail::empty_strided_cpu({864L, 288L}, {288L, 1L}, at::kFloat);
    cpp_fused__to_copy_add_embedding_mean_mul_rsqrt_0((int*)(arg90_1.data_ptr()), (float*)(L__self___tok_embeddings_weight.data_ptr()), (float*)(L__self___layers_0_attention_norm_weight.data_ptr()), (signed char*)(L__self___layers_0_attention_wqkv_weight.data_ptr()), (float*)(buf0.data_ptr()), (float*)(buf1.data_ptr()), (float*)(buf2.data_ptr()), s0);
    at::Tensor buf3 = at::detail::empty_strided_cpu({s0, 864L}, {864L, 1L}, at::kFloat);
    // Source Nodes: [linear], Original ATen: [aten.mm]
    at::mm_out(buf3, reinterpret_tensor(buf1, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf2, {288L, 864L}, {1L, 288L}, 0L));
    decltype(auto) buf6 = reinterpret_tensor(buf1, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf1.reset();  // reuse
    decltype(auto) buf4 = reinterpret_tensor(buf6, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf5 = reinterpret_tensor(buf6, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    at::Tensor buf9 = at::detail::empty_strided_cpu({1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, at::kFloat);
    decltype(auto) buf7 = reinterpret_tensor(buf9, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf8 = reinterpret_tensor(buf9, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    at::Tensor buf10 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    at::Tensor buf12 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    at::Tensor buf14 = at::detail::empty_strided_cpu({1L, 1L, s0, 352L}, {352L*s0, 352L*s0, 352L, 1L}, at::kFloat);
    at::Tensor buf44 = at::detail::empty_strided_cpu({1L, 1L, s0, 352L}, {352L*s0, 352L*s0, 352L, 1L}, at::kFloat);
    at::Tensor buf74 = at::detail::empty_strided_cpu({1L, 1L, s0, 352L}, {352L*s0, 352L*s0, 352L, 1L}, at::kFloat);
    cpp_fused__scaled_dot_product_flash_attention_for_cpu_index_index_put_logical_not_masked_fill_stack_zeros_like_1((float*)(buf3.data_ptr()), (bfloat16*)(L__self___layers_0_attention_wqkv_scales.data_ptr()), (int*)(arg91_1.data_ptr()), (float*)(L__self___freqs_cis.data_ptr()), (float*)(L__self___layers_0_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_0_attention_kv_cache_v_cache.data_ptr()), (float*)(buf9.data_ptr()), (bool*)(L__self___causal_mask.data_ptr()), (float*)(buf4.data_ptr()), (float*)(buf5.data_ptr()), (float*)(buf7.data_ptr()), (float*)(buf8.data_ptr()), (float*)(buf10.data_ptr()), (float*)(buf12.data_ptr()), (float*)(buf14.data_ptr()), (float*)(buf44.data_ptr()), (float*)(buf74.data_ptr()), s0);
    buf4.reset();
    buf5.reset();
    buf7.reset();
    buf8.reset();
    // Source Nodes: [mask, y], Original ATen: [aten._scaled_dot_product_flash_attention_for_cpu, aten.index, aten.logical_not, aten.masked_fill, aten.zeros_like]
    auto buf15 = at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(reinterpret_tensor(buf6, {1L, 6L, s0, 48L}, {288L*s0, 48L, 288L, 1L}, 0L), reinterpret_tensor(buf10, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), reinterpret_tensor(buf12, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), 0.0, false, buf14, c10::nullopt);
    auto buf16 = std::get<0>(buf15);

    at::Tensor buf18 = at::detail::empty_strided_cpu({288L, 288L}, {288L, 1L}, at::kFloat);
    cpp_fused__to_copy_2((signed char*)(L__self___layers_0_attention_wo_weight.data_ptr()), (float*)(buf18.data_ptr()));
    decltype(auto) buf19 = reinterpret_tensor(buf6, {s0, 288L}, {288L, 1L}, 0L); buf6.reset();  // reuse
    // Source Nodes: [linear_1], Original ATen: [aten.mm]
    at::mm_out(buf19, reinterpret_tensor(buf16, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf18, {288L, 288L}, {1L, 288L}, 0L));
    decltype(auto) buf20 = buf0; buf0.reset();;  // reuse
    decltype(auto) buf21 = reinterpret_tensor(buf16, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf16.reset();  // reuse
    at::Tensor buf22 = at::detail::empty_strided_cpu({768L, 288L}, {288L, 1L}, at::kFloat);
    cpp_fused__to_copy_add_embedding_mean_mul_rsqrt_3((int*)(arg90_1.data_ptr()), (float*)(L__self___tok_embeddings_weight.data_ptr()), (float*)(buf19.data_ptr()), (bfloat16*)(L__self___layers_0_attention_wo_scales.data_ptr()), (float*)(L__self___layers_0_ffn_norm_weight.data_ptr()), (signed char*)(L__self___layers_0_feed_forward_w1_weight.data_ptr()), (float*)(buf20.data_ptr()), (float*)(buf21.data_ptr()), (float*)(buf22.data_ptr()), s0);
    at::Tensor buf23 = at::detail::empty_strided_cpu({s0, 768L}, {768L, 1L}, at::kFloat);
    // Source Nodes: [linear_2], Original ATen: [aten.mm]
    at::mm_out(buf23, reinterpret_tensor(buf21, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf22, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf24 = buf22; buf22.reset();;  // reuse
    cpp_fused__to_copy_4((signed char*)(L__self___layers_0_feed_forward_w3_weight.data_ptr()), (float*)(buf24.data_ptr()));
    at::Tensor buf25 = at::detail::empty_strided_cpu({s0, 768L}, {768L, 1L}, at::kFloat);
    // Source Nodes: [linear_3], Original ATen: [aten.mm]
    at::mm_out(buf25, reinterpret_tensor(buf21, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf24, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf26 = reinterpret_tensor(buf23, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L); buf23.reset();  // reuse
    decltype(auto) buf27 = reinterpret_tensor(buf24, {288L, 768L}, {768L, 1L}, 0L); buf24.reset();  // reuse
    cpp_fused__to_copy_mul_silu_5((float*)(buf26.data_ptr()), (bfloat16*)(L__self___layers_0_feed_forward_w1_scales.data_ptr()), (float*)(buf25.data_ptr()), (bfloat16*)(L__self___layers_0_feed_forward_w3_scales.data_ptr()), (signed char*)(L__self___layers_0_feed_forward_w2_weight.data_ptr()), (float*)(buf27.data_ptr()), s0);
    decltype(auto) buf28 = reinterpret_tensor(buf21, {s0, 288L}, {288L, 1L}, 0L); buf21.reset();  // reuse
    // Source Nodes: [linear_4], Original ATen: [aten.mm]
    at::mm_out(buf28, reinterpret_tensor(buf26, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(buf27, {768L, 288L}, {1L, 768L}, 0L));
    decltype(auto) buf29 = reinterpret_tensor(buf19, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf19.reset();  // reuse
    decltype(auto) buf30 = buf20; buf20.reset();;  // reuse
    decltype(auto) buf31 = reinterpret_tensor(buf9, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf9.reset();  // reuse
    decltype(auto) buf32 = buf2; buf2.reset();;  // reuse
    cpp_fused__to_copy_add_embedding_mean_mul_rsqrt_6((float*)(buf29.data_ptr()), (int*)(arg90_1.data_ptr()), (float*)(L__self___tok_embeddings_weight.data_ptr()), (bfloat16*)(L__self___layers_0_attention_wo_scales.data_ptr()), (float*)(buf28.data_ptr()), (bfloat16*)(L__self___layers_0_feed_forward_w2_scales.data_ptr()), (float*)(L__self___layers_1_attention_norm_weight.data_ptr()), (signed char*)(L__self___layers_1_attention_wqkv_weight.data_ptr()), (float*)(buf30.data_ptr()), (float*)(buf31.data_ptr()), (float*)(buf32.data_ptr()), s0);
    arg90_1.reset();
    decltype(auto) buf33 = buf3; buf3.reset();;  // reuse
    // Source Nodes: [linear_5], Original ATen: [aten.mm]
    at::mm_out(buf33, reinterpret_tensor(buf31, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf32, {288L, 864L}, {1L, 288L}, 0L));
    decltype(auto) buf36 = reinterpret_tensor(buf31, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf31.reset();  // reuse
    decltype(auto) buf34 = reinterpret_tensor(buf36, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf35 = reinterpret_tensor(buf36, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    decltype(auto) buf39 = reinterpret_tensor(buf28, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf28.reset();  // reuse
    decltype(auto) buf37 = reinterpret_tensor(buf39, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf38 = reinterpret_tensor(buf39, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    at::Tensor buf40 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    at::Tensor buf42 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    cpp_fused_index_put_stack_7((float*)(buf33.data_ptr()), (bfloat16*)(L__self___layers_1_attention_wqkv_scales.data_ptr()), (int*)(arg91_1.data_ptr()), (float*)(L__self___freqs_cis.data_ptr()), (float*)(L__self___layers_1_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_1_attention_kv_cache_v_cache.data_ptr()), (float*)(buf39.data_ptr()), (float*)(buf34.data_ptr()), (float*)(buf35.data_ptr()), (float*)(buf37.data_ptr()), (float*)(buf38.data_ptr()), (float*)(buf40.data_ptr()), (float*)(buf42.data_ptr()), s0);
    buf34.reset();
    buf35.reset();
    buf37.reset();
    buf38.reset();
    // Source Nodes: [mask, y_3], Original ATen: [aten._scaled_dot_product_flash_attention_for_cpu, aten.index, aten.logical_not, aten.masked_fill, aten.zeros_like]
    auto buf45 = at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(reinterpret_tensor(buf36, {1L, 6L, s0, 48L}, {288L*s0, 48L, 288L, 1L}, 0L), reinterpret_tensor(buf40, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), reinterpret_tensor(buf42, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), 0.0, false, buf44, c10::nullopt);
    auto buf46 = std::get<0>(buf45);

    decltype(auto) buf48 = buf18; buf18.reset();;  // reuse
    cpp_fused__to_copy_8((signed char*)(L__self___layers_1_attention_wo_weight.data_ptr()), (float*)(buf48.data_ptr()));
    decltype(auto) buf49 = reinterpret_tensor(buf36, {s0, 288L}, {288L, 1L}, 0L); buf36.reset();  // reuse
    // Source Nodes: [linear_6], Original ATen: [aten.mm]
    at::mm_out(buf49, reinterpret_tensor(buf46, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf48, {288L, 288L}, {1L, 288L}, 0L));
    decltype(auto) buf50 = buf30; buf30.reset();;  // reuse
    decltype(auto) buf51 = reinterpret_tensor(buf46, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf46.reset();  // reuse
    decltype(auto) buf52 = reinterpret_tensor(buf27, {768L, 288L}, {288L, 1L}, 0L); buf27.reset();  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_9((float*)(buf29.data_ptr()), (float*)(buf49.data_ptr()), (bfloat16*)(L__self___layers_1_attention_wo_scales.data_ptr()), (float*)(L__self___layers_1_ffn_norm_weight.data_ptr()), (signed char*)(L__self___layers_1_feed_forward_w1_weight.data_ptr()), (float*)(buf50.data_ptr()), (float*)(buf51.data_ptr()), (float*)(buf52.data_ptr()), s0);
    decltype(auto) buf53 = reinterpret_tensor(buf26, {s0, 768L}, {768L, 1L}, 0L); buf26.reset();  // reuse
    // Source Nodes: [linear_7], Original ATen: [aten.mm]
    at::mm_out(buf53, reinterpret_tensor(buf51, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf52, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf54 = buf52; buf52.reset();;  // reuse
    cpp_fused__to_copy_10((signed char*)(L__self___layers_1_feed_forward_w3_weight.data_ptr()), (float*)(buf54.data_ptr()));
    decltype(auto) buf55 = buf25; buf25.reset();;  // reuse
    // Source Nodes: [linear_8], Original ATen: [aten.mm]
    at::mm_out(buf55, reinterpret_tensor(buf51, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf54, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf56 = reinterpret_tensor(buf53, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L); buf53.reset();  // reuse
    decltype(auto) buf57 = reinterpret_tensor(buf54, {288L, 768L}, {768L, 1L}, 0L); buf54.reset();  // reuse
    cpp_fused__to_copy_mul_silu_11((float*)(buf56.data_ptr()), (bfloat16*)(L__self___layers_1_feed_forward_w1_scales.data_ptr()), (float*)(buf55.data_ptr()), (bfloat16*)(L__self___layers_1_feed_forward_w3_scales.data_ptr()), (signed char*)(L__self___layers_1_feed_forward_w2_weight.data_ptr()), (float*)(buf57.data_ptr()), s0);
    decltype(auto) buf58 = reinterpret_tensor(buf51, {s0, 288L}, {288L, 1L}, 0L); buf51.reset();  // reuse
    // Source Nodes: [linear_9], Original ATen: [aten.mm]
    at::mm_out(buf58, reinterpret_tensor(buf56, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(buf57, {768L, 288L}, {1L, 768L}, 0L));
    decltype(auto) buf59 = buf29; buf29.reset();;  // reuse
    decltype(auto) buf60 = buf50; buf50.reset();;  // reuse
    decltype(auto) buf61 = reinterpret_tensor(buf39, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf39.reset();  // reuse
    decltype(auto) buf62 = buf32; buf32.reset();;  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_12((float*)(buf59.data_ptr()), (float*)(buf49.data_ptr()), (bfloat16*)(L__self___layers_1_attention_wo_scales.data_ptr()), (float*)(buf58.data_ptr()), (bfloat16*)(L__self___layers_1_feed_forward_w2_scales.data_ptr()), (float*)(L__self___layers_2_attention_norm_weight.data_ptr()), (signed char*)(L__self___layers_2_attention_wqkv_weight.data_ptr()), (float*)(buf60.data_ptr()), (float*)(buf61.data_ptr()), (float*)(buf62.data_ptr()), s0);
    buf49.reset();
    decltype(auto) buf63 = buf33; buf33.reset();;  // reuse
    // Source Nodes: [linear_10], Original ATen: [aten.mm]
    at::mm_out(buf63, reinterpret_tensor(buf61, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf62, {288L, 864L}, {1L, 288L}, 0L));
    decltype(auto) buf66 = reinterpret_tensor(buf61, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf61.reset();  // reuse
    decltype(auto) buf64 = reinterpret_tensor(buf66, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf65 = reinterpret_tensor(buf66, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    decltype(auto) buf69 = reinterpret_tensor(buf58, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf58.reset();  // reuse
    decltype(auto) buf67 = reinterpret_tensor(buf69, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf68 = reinterpret_tensor(buf69, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    at::Tensor buf70 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    at::Tensor buf72 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    cpp_fused_index_put_stack_13((float*)(buf63.data_ptr()), (bfloat16*)(L__self___layers_2_attention_wqkv_scales.data_ptr()), (int*)(arg91_1.data_ptr()), (float*)(L__self___freqs_cis.data_ptr()), (float*)(L__self___layers_2_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_2_attention_kv_cache_v_cache.data_ptr()), (float*)(buf69.data_ptr()), (float*)(buf64.data_ptr()), (float*)(buf65.data_ptr()), (float*)(buf67.data_ptr()), (float*)(buf68.data_ptr()), (float*)(buf70.data_ptr()), (float*)(buf72.data_ptr()), s0);
    buf64.reset();
    buf65.reset();
    buf67.reset();
    buf68.reset();
    // Source Nodes: [mask, y_6], Original ATen: [aten._scaled_dot_product_flash_attention_for_cpu, aten.index, aten.logical_not, aten.masked_fill, aten.zeros_like]
    auto buf75 = at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(reinterpret_tensor(buf66, {1L, 6L, s0, 48L}, {288L*s0, 48L, 288L, 1L}, 0L), reinterpret_tensor(buf70, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), reinterpret_tensor(buf72, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), 0.0, false, buf74, c10::nullopt);
    auto buf76 = std::get<0>(buf75);

    decltype(auto) buf78 = buf48; buf48.reset();;  // reuse
    cpp_fused__to_copy_14((signed char*)(L__self___layers_2_attention_wo_weight.data_ptr()), (float*)(buf78.data_ptr()));
    decltype(auto) buf79 = reinterpret_tensor(buf66, {s0, 288L}, {288L, 1L}, 0L); buf66.reset();  // reuse
    // Source Nodes: [linear_11], Original ATen: [aten.mm]
    at::mm_out(buf79, reinterpret_tensor(buf76, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf78, {288L, 288L}, {1L, 288L}, 0L));
    decltype(auto) buf80 = buf60; buf60.reset();;  // reuse
    decltype(auto) buf81 = reinterpret_tensor(buf76, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf76.reset();  // reuse
    decltype(auto) buf82 = reinterpret_tensor(buf57, {768L, 288L}, {288L, 1L}, 0L); buf57.reset();  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_15((float*)(buf59.data_ptr()), (float*)(buf79.data_ptr()), (bfloat16*)(L__self___layers_2_attention_wo_scales.data_ptr()), (float*)(L__self___layers_2_ffn_norm_weight.data_ptr()), (signed char*)(L__self___layers_2_feed_forward_w1_weight.data_ptr()), (float*)(buf80.data_ptr()), (float*)(buf81.data_ptr()), (float*)(buf82.data_ptr()), s0);
    decltype(auto) buf83 = reinterpret_tensor(buf56, {s0, 768L}, {768L, 1L}, 0L); buf56.reset();  // reuse
    // Source Nodes: [linear_12], Original ATen: [aten.mm]
    at::mm_out(buf83, reinterpret_tensor(buf81, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf82, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf84 = buf82; buf82.reset();;  // reuse
    cpp_fused__to_copy_16((signed char*)(L__self___layers_2_feed_forward_w3_weight.data_ptr()), (float*)(buf84.data_ptr()));
    decltype(auto) buf85 = buf55; buf55.reset();;  // reuse
    // Source Nodes: [linear_13], Original ATen: [aten.mm]
    at::mm_out(buf85, reinterpret_tensor(buf81, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf84, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf86 = reinterpret_tensor(buf83, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L); buf83.reset();  // reuse
    decltype(auto) buf87 = reinterpret_tensor(buf84, {288L, 768L}, {768L, 1L}, 0L); buf84.reset();  // reuse
    cpp_fused__to_copy_mul_silu_17((float*)(buf86.data_ptr()), (bfloat16*)(L__self___layers_2_feed_forward_w1_scales.data_ptr()), (float*)(buf85.data_ptr()), (bfloat16*)(L__self___layers_2_feed_forward_w3_scales.data_ptr()), (signed char*)(L__self___layers_2_feed_forward_w2_weight.data_ptr()), (float*)(buf87.data_ptr()), s0);
    decltype(auto) buf88 = reinterpret_tensor(buf81, {s0, 288L}, {288L, 1L}, 0L); buf81.reset();  // reuse
    // Source Nodes: [linear_14], Original ATen: [aten.mm]
    at::mm_out(buf88, reinterpret_tensor(buf86, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(buf87, {768L, 288L}, {1L, 768L}, 0L));
    decltype(auto) buf89 = buf59; buf59.reset();;  // reuse
    decltype(auto) buf90 = buf80; buf80.reset();;  // reuse
    decltype(auto) buf91 = reinterpret_tensor(buf69, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf69.reset();  // reuse
    decltype(auto) buf92 = buf62; buf62.reset();;  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_18((float*)(buf89.data_ptr()), (float*)(buf79.data_ptr()), (bfloat16*)(L__self___layers_2_attention_wo_scales.data_ptr()), (float*)(buf88.data_ptr()), (bfloat16*)(L__self___layers_2_feed_forward_w2_scales.data_ptr()), (float*)(L__self___layers_3_attention_norm_weight.data_ptr()), (signed char*)(L__self___layers_3_attention_wqkv_weight.data_ptr()), (float*)(buf90.data_ptr()), (float*)(buf91.data_ptr()), (float*)(buf92.data_ptr()), s0);
    buf79.reset();
    decltype(auto) buf93 = buf63; buf63.reset();;  // reuse
    // Source Nodes: [linear_15], Original ATen: [aten.mm]
    at::mm_out(buf93, reinterpret_tensor(buf91, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf92, {288L, 864L}, {1L, 288L}, 0L));
    decltype(auto) buf96 = reinterpret_tensor(buf91, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf91.reset();  // reuse
    decltype(auto) buf94 = reinterpret_tensor(buf96, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf95 = reinterpret_tensor(buf96, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    decltype(auto) buf99 = reinterpret_tensor(buf88, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf88.reset();  // reuse
    decltype(auto) buf97 = reinterpret_tensor(buf99, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf98 = reinterpret_tensor(buf99, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    at::Tensor buf100 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    at::Tensor buf102 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    decltype(auto) buf104 = buf74; buf74.reset();;  // reuse
    decltype(auto) buf134 = buf44; buf44.reset();;  // reuse
    decltype(auto) buf164 = buf14; buf14.reset();;  // reuse
    cpp_fused__scaled_dot_product_flash_attention_for_cpu_index_index_put_logical_not_masked_fill_stack_zeros_like_19((float*)(buf93.data_ptr()), (bfloat16*)(L__self___layers_3_attention_wqkv_scales.data_ptr()), (int*)(arg91_1.data_ptr()), (float*)(L__self___freqs_cis.data_ptr()), (float*)(L__self___layers_3_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_3_attention_kv_cache_v_cache.data_ptr()), (float*)(buf99.data_ptr()), (bool*)(L__self___causal_mask.data_ptr()), (float*)(buf94.data_ptr()), (float*)(buf95.data_ptr()), (float*)(buf97.data_ptr()), (float*)(buf98.data_ptr()), (float*)(buf100.data_ptr()), (float*)(buf102.data_ptr()), (float*)(buf104.data_ptr()), (float*)(buf134.data_ptr()), (float*)(buf164.data_ptr()), s0);
    buf94.reset();
    buf95.reset();
    buf97.reset();
    buf98.reset();
    // Source Nodes: [mask, y_9], Original ATen: [aten._scaled_dot_product_flash_attention_for_cpu, aten.index, aten.logical_not, aten.masked_fill, aten.zeros_like]
    auto buf105 = at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(reinterpret_tensor(buf96, {1L, 6L, s0, 48L}, {288L*s0, 48L, 288L, 1L}, 0L), reinterpret_tensor(buf100, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), reinterpret_tensor(buf102, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), 0.0, false, buf104, c10::nullopt);
    buf104.reset();
    auto buf106 = std::get<0>(buf105);

    decltype(auto) buf108 = buf78; buf78.reset();;  // reuse
    cpp_fused__to_copy_20((signed char*)(L__self___layers_3_attention_wo_weight.data_ptr()), (float*)(buf108.data_ptr()));
    decltype(auto) buf109 = reinterpret_tensor(buf96, {s0, 288L}, {288L, 1L}, 0L); buf96.reset();  // reuse
    // Source Nodes: [linear_16], Original ATen: [aten.mm]
    at::mm_out(buf109, reinterpret_tensor(buf106, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf108, {288L, 288L}, {1L, 288L}, 0L));
    decltype(auto) buf110 = buf90; buf90.reset();;  // reuse
    decltype(auto) buf111 = reinterpret_tensor(buf106, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf106.reset();  // reuse
    decltype(auto) buf112 = reinterpret_tensor(buf87, {768L, 288L}, {288L, 1L}, 0L); buf87.reset();  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_21((float*)(buf89.data_ptr()), (float*)(buf109.data_ptr()), (bfloat16*)(L__self___layers_3_attention_wo_scales.data_ptr()), (float*)(L__self___layers_3_ffn_norm_weight.data_ptr()), (signed char*)(L__self___layers_3_feed_forward_w1_weight.data_ptr()), (float*)(buf110.data_ptr()), (float*)(buf111.data_ptr()), (float*)(buf112.data_ptr()), s0);
    decltype(auto) buf113 = reinterpret_tensor(buf86, {s0, 768L}, {768L, 1L}, 0L); buf86.reset();  // reuse
    // Source Nodes: [linear_17], Original ATen: [aten.mm]
    at::mm_out(buf113, reinterpret_tensor(buf111, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf112, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf114 = buf112; buf112.reset();;  // reuse
    cpp_fused__to_copy_22((signed char*)(L__self___layers_3_feed_forward_w3_weight.data_ptr()), (float*)(buf114.data_ptr()));
    decltype(auto) buf115 = buf85; buf85.reset();;  // reuse
    // Source Nodes: [linear_18], Original ATen: [aten.mm]
    at::mm_out(buf115, reinterpret_tensor(buf111, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf114, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf116 = reinterpret_tensor(buf113, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L); buf113.reset();  // reuse
    decltype(auto) buf117 = reinterpret_tensor(buf114, {288L, 768L}, {768L, 1L}, 0L); buf114.reset();  // reuse
    cpp_fused__to_copy_mul_silu_23((float*)(buf116.data_ptr()), (bfloat16*)(L__self___layers_3_feed_forward_w1_scales.data_ptr()), (float*)(buf115.data_ptr()), (bfloat16*)(L__self___layers_3_feed_forward_w3_scales.data_ptr()), (signed char*)(L__self___layers_3_feed_forward_w2_weight.data_ptr()), (float*)(buf117.data_ptr()), s0);
    decltype(auto) buf118 = reinterpret_tensor(buf111, {s0, 288L}, {288L, 1L}, 0L); buf111.reset();  // reuse
    // Source Nodes: [linear_19], Original ATen: [aten.mm]
    at::mm_out(buf118, reinterpret_tensor(buf116, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(buf117, {768L, 288L}, {1L, 768L}, 0L));
    decltype(auto) buf119 = reinterpret_tensor(buf109, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf109.reset();  // reuse
    decltype(auto) buf120 = buf110; buf110.reset();;  // reuse
    decltype(auto) buf121 = reinterpret_tensor(buf99, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf99.reset();  // reuse
    decltype(auto) buf122 = buf92; buf92.reset();;  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_24((float*)(buf119.data_ptr()), (float*)(buf89.data_ptr()), (bfloat16*)(L__self___layers_3_attention_wo_scales.data_ptr()), (float*)(buf118.data_ptr()), (bfloat16*)(L__self___layers_3_feed_forward_w2_scales.data_ptr()), (float*)(L__self___layers_4_attention_norm_weight.data_ptr()), (signed char*)(L__self___layers_4_attention_wqkv_weight.data_ptr()), (float*)(buf120.data_ptr()), (float*)(buf121.data_ptr()), (float*)(buf122.data_ptr()), s0);
    buf118.reset();
    decltype(auto) buf123 = buf93; buf93.reset();;  // reuse
    // Source Nodes: [linear_20], Original ATen: [aten.mm]
    at::mm_out(buf123, reinterpret_tensor(buf121, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf122, {288L, 864L}, {1L, 288L}, 0L));
    decltype(auto) buf126 = reinterpret_tensor(buf121, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf121.reset();  // reuse
    decltype(auto) buf124 = reinterpret_tensor(buf126, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf125 = reinterpret_tensor(buf126, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    decltype(auto) buf129 = reinterpret_tensor(buf89, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf89.reset();  // reuse
    decltype(auto) buf127 = reinterpret_tensor(buf129, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf128 = reinterpret_tensor(buf129, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    at::Tensor buf130 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    at::Tensor buf132 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    cpp_fused_index_put_stack_25((float*)(buf123.data_ptr()), (bfloat16*)(L__self___layers_4_attention_wqkv_scales.data_ptr()), (int*)(arg91_1.data_ptr()), (float*)(L__self___freqs_cis.data_ptr()), (float*)(L__self___layers_4_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_4_attention_kv_cache_v_cache.data_ptr()), (float*)(buf129.data_ptr()), (float*)(buf124.data_ptr()), (float*)(buf125.data_ptr()), (float*)(buf127.data_ptr()), (float*)(buf128.data_ptr()), (float*)(buf130.data_ptr()), (float*)(buf132.data_ptr()), s0);
    buf124.reset();
    buf125.reset();
    buf127.reset();
    buf128.reset();
    // Source Nodes: [mask, y_12], Original ATen: [aten._scaled_dot_product_flash_attention_for_cpu, aten.index, aten.logical_not, aten.masked_fill, aten.zeros_like]
    auto buf135 = at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(reinterpret_tensor(buf126, {1L, 6L, s0, 48L}, {288L*s0, 48L, 288L, 1L}, 0L), reinterpret_tensor(buf130, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), reinterpret_tensor(buf132, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), 0.0, false, buf134, c10::nullopt);
    buf134.reset();
    auto buf136 = std::get<0>(buf135);

    decltype(auto) buf138 = buf108; buf108.reset();;  // reuse
    cpp_fused__to_copy_26((signed char*)(L__self___layers_4_attention_wo_weight.data_ptr()), (float*)(buf138.data_ptr()));
    decltype(auto) buf139 = reinterpret_tensor(buf126, {s0, 288L}, {288L, 1L}, 0L); buf126.reset();  // reuse
    // Source Nodes: [linear_21], Original ATen: [aten.mm]
    at::mm_out(buf139, reinterpret_tensor(buf136, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf138, {288L, 288L}, {1L, 288L}, 0L));
    decltype(auto) buf140 = buf120; buf120.reset();;  // reuse
    decltype(auto) buf141 = reinterpret_tensor(buf136, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf136.reset();  // reuse
    decltype(auto) buf142 = reinterpret_tensor(buf117, {768L, 288L}, {288L, 1L}, 0L); buf117.reset();  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_27((float*)(buf119.data_ptr()), (float*)(buf139.data_ptr()), (bfloat16*)(L__self___layers_4_attention_wo_scales.data_ptr()), (float*)(L__self___layers_4_ffn_norm_weight.data_ptr()), (signed char*)(L__self___layers_4_feed_forward_w1_weight.data_ptr()), (float*)(buf140.data_ptr()), (float*)(buf141.data_ptr()), (float*)(buf142.data_ptr()), s0);
    decltype(auto) buf143 = reinterpret_tensor(buf116, {s0, 768L}, {768L, 1L}, 0L); buf116.reset();  // reuse
    // Source Nodes: [linear_22], Original ATen: [aten.mm]
    at::mm_out(buf143, reinterpret_tensor(buf141, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf142, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf144 = buf142; buf142.reset();;  // reuse
    cpp_fused__to_copy_28((signed char*)(L__self___layers_4_feed_forward_w3_weight.data_ptr()), (float*)(buf144.data_ptr()));
    decltype(auto) buf145 = buf115; buf115.reset();;  // reuse
    // Source Nodes: [linear_23], Original ATen: [aten.mm]
    at::mm_out(buf145, reinterpret_tensor(buf141, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf144, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf146 = reinterpret_tensor(buf143, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L); buf143.reset();  // reuse
    decltype(auto) buf147 = reinterpret_tensor(buf144, {288L, 768L}, {768L, 1L}, 0L); buf144.reset();  // reuse
    cpp_fused__to_copy_mul_silu_29((float*)(buf146.data_ptr()), (bfloat16*)(L__self___layers_4_feed_forward_w1_scales.data_ptr()), (float*)(buf145.data_ptr()), (bfloat16*)(L__self___layers_4_feed_forward_w3_scales.data_ptr()), (signed char*)(L__self___layers_4_feed_forward_w2_weight.data_ptr()), (float*)(buf147.data_ptr()), s0);
    decltype(auto) buf148 = reinterpret_tensor(buf141, {s0, 288L}, {288L, 1L}, 0L); buf141.reset();  // reuse
    // Source Nodes: [linear_24], Original ATen: [aten.mm]
    at::mm_out(buf148, reinterpret_tensor(buf146, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(buf147, {768L, 288L}, {1L, 768L}, 0L));
    decltype(auto) buf149 = buf119; buf119.reset();;  // reuse
    decltype(auto) buf150 = buf140; buf140.reset();;  // reuse
    decltype(auto) buf151 = reinterpret_tensor(buf129, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf129.reset();  // reuse
    decltype(auto) buf152 = buf122; buf122.reset();;  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_30((float*)(buf149.data_ptr()), (float*)(buf139.data_ptr()), (bfloat16*)(L__self___layers_4_attention_wo_scales.data_ptr()), (float*)(buf148.data_ptr()), (bfloat16*)(L__self___layers_4_feed_forward_w2_scales.data_ptr()), (float*)(L__self___layers_5_attention_norm_weight.data_ptr()), (signed char*)(L__self___layers_5_attention_wqkv_weight.data_ptr()), (float*)(buf150.data_ptr()), (float*)(buf151.data_ptr()), (float*)(buf152.data_ptr()), s0);
    buf139.reset();
    decltype(auto) buf153 = buf123; buf123.reset();;  // reuse
    // Source Nodes: [linear_25], Original ATen: [aten.mm]
    at::mm_out(buf153, reinterpret_tensor(buf151, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf152, {288L, 864L}, {1L, 288L}, 0L));
    buf152.reset();
    decltype(auto) buf156 = reinterpret_tensor(buf151, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf151.reset();  // reuse
    decltype(auto) buf154 = reinterpret_tensor(buf156, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf155 = reinterpret_tensor(buf156, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    decltype(auto) buf159 = reinterpret_tensor(buf148, {1L, s0, 6L, 24L, 2L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L); buf148.reset();  // reuse
    decltype(auto) buf157 = reinterpret_tensor(buf159, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 0L);  // alias
    decltype(auto) buf158 = reinterpret_tensor(buf159, {1L, s0, 6L, 24L, 1L}, {288L*s0, 288L, 48L, 2L, 1L}, 1L);  // alias
    at::Tensor buf160 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    at::Tensor buf162 = at::detail::empty_strided_cpu({1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, at::kFloat);
    cpp_fused_index_put_stack_31((float*)(buf153.data_ptr()), (bfloat16*)(L__self___layers_5_attention_wqkv_scales.data_ptr()), (int*)(arg91_1.data_ptr()), (float*)(L__self___freqs_cis.data_ptr()), (float*)(L__self___layers_5_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_5_attention_kv_cache_v_cache.data_ptr()), (float*)(buf159.data_ptr()), (float*)(buf154.data_ptr()), (float*)(buf155.data_ptr()), (float*)(buf157.data_ptr()), (float*)(buf158.data_ptr()), (float*)(buf160.data_ptr()), (float*)(buf162.data_ptr()), s0);
    arg91_1.reset();
    buf153.reset();
    buf154.reset();
    buf155.reset();
    buf157.reset();
    buf158.reset();
    buf159.reset();
    // Source Nodes: [mask, y_15], Original ATen: [aten._scaled_dot_product_flash_attention_for_cpu, aten.index, aten.logical_not, aten.masked_fill, aten.zeros_like]
    auto buf165 = at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(reinterpret_tensor(buf156, {1L, 6L, s0, 48L}, {288L*s0, 48L, 288L, 1L}, 0L), reinterpret_tensor(buf160, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), reinterpret_tensor(buf162, {1L, 6L, 352L, 48L}, {101376L, 16896L, 48L, 1L}, 0L), 0.0, false, buf164, c10::nullopt);
    buf164.reset();
    auto buf166 = std::get<0>(buf165);

    decltype(auto) buf168 = buf138; buf138.reset();;  // reuse
    cpp_fused__to_copy_32((signed char*)(L__self___layers_5_attention_wo_weight.data_ptr()), (float*)(buf168.data_ptr()));
    decltype(auto) buf169 = reinterpret_tensor(buf156, {s0, 288L}, {288L, 1L}, 0L); buf156.reset();  // reuse
    // Source Nodes: [linear_26], Original ATen: [aten.mm]
    at::mm_out(buf169, reinterpret_tensor(buf166, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf168, {288L, 288L}, {1L, 288L}, 0L));
    buf168.reset();
    decltype(auto) buf170 = buf150; buf150.reset();;  // reuse
    decltype(auto) buf171 = reinterpret_tensor(buf166, {1L, s0, 288L}, {288L*s0, 288L, 1L}, 0L); buf166.reset();  // reuse
    decltype(auto) buf172 = reinterpret_tensor(buf147, {768L, 288L}, {288L, 1L}, 0L); buf147.reset();  // reuse
    cpp_fused__to_copy_add_mean_mul_rsqrt_33((float*)(buf149.data_ptr()), (float*)(buf169.data_ptr()), (bfloat16*)(L__self___layers_5_attention_wo_scales.data_ptr()), (float*)(L__self___layers_5_ffn_norm_weight.data_ptr()), (signed char*)(L__self___layers_5_feed_forward_w1_weight.data_ptr()), (float*)(buf170.data_ptr()), (float*)(buf171.data_ptr()), (float*)(buf172.data_ptr()), s0);
    decltype(auto) buf173 = reinterpret_tensor(buf146, {s0, 768L}, {768L, 1L}, 0L); buf146.reset();  // reuse
    // Source Nodes: [linear_27], Original ATen: [aten.mm]
    at::mm_out(buf173, reinterpret_tensor(buf171, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf172, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf174 = buf172; buf172.reset();;  // reuse
    cpp_fused__to_copy_34((signed char*)(L__self___layers_5_feed_forward_w3_weight.data_ptr()), (float*)(buf174.data_ptr()));
    decltype(auto) buf175 = buf145; buf145.reset();;  // reuse
    // Source Nodes: [linear_28], Original ATen: [aten.mm]
    at::mm_out(buf175, reinterpret_tensor(buf171, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf174, {288L, 768L}, {1L, 288L}, 0L));
    decltype(auto) buf176 = reinterpret_tensor(buf173, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L); buf173.reset();  // reuse
    decltype(auto) buf177 = reinterpret_tensor(buf174, {288L, 768L}, {768L, 1L}, 0L); buf174.reset();  // reuse
    cpp_fused__to_copy_mul_silu_35((float*)(buf176.data_ptr()), (bfloat16*)(L__self___layers_5_feed_forward_w1_scales.data_ptr()), (float*)(buf175.data_ptr()), (bfloat16*)(L__self___layers_5_feed_forward_w3_scales.data_ptr()), (signed char*)(L__self___layers_5_feed_forward_w2_weight.data_ptr()), (float*)(buf177.data_ptr()), s0);
    buf175.reset();
    decltype(auto) buf178 = reinterpret_tensor(buf171, {s0, 288L}, {288L, 1L}, 0L); buf171.reset();  // reuse
    // Source Nodes: [linear_29], Original ATen: [aten.mm]
    at::mm_out(buf178, reinterpret_tensor(buf176, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(buf177, {768L, 288L}, {1L, 768L}, 0L));
    buf176.reset();
    buf177.reset();
    decltype(auto) buf179 = buf149; buf149.reset();;  // reuse
    decltype(auto) buf180 = buf170; buf170.reset();;  // reuse
    decltype(auto) buf181 = buf179; buf179.reset();;  // reuse
    at::Tensor buf182 = at::detail::empty_strided_cpu({32000L, 288L}, {288L, 1L}, at::kFloat);
    cpp_fused__to_copy_add_mean_mul_rsqrt_36((float*)(buf181.data_ptr()), (float*)(buf169.data_ptr()), (bfloat16*)(L__self___layers_5_attention_wo_scales.data_ptr()), (float*)(buf178.data_ptr()), (bfloat16*)(L__self___layers_5_feed_forward_w2_scales.data_ptr()), (float*)(L__self___norm_weight.data_ptr()), (signed char*)(L__self___output_weight.data_ptr()), (float*)(buf180.data_ptr()), (float*)(buf182.data_ptr()), s0);
    buf169.reset();
    buf178.reset();
    buf180.reset();
    at::Tensor buf183 = at::detail::empty_strided_cpu({s0, 32000L}, {32000L, 1L}, at::kFloat);
    // Source Nodes: [linear_30], Original ATen: [aten.mm]
    at::mm_out(buf183, reinterpret_tensor(buf181, {s0, 288L}, {288L, 1L}, 0L), reinterpret_tensor(buf182, {288L, 32000L}, {1L, 288L}, 0L));
    buf181.reset();
    buf182.reset();
    decltype(auto) buf184 = reinterpret_tensor(buf183, {1L, s0, 32000L}, {32000L*s0, 32000L, 1L}, 0L); buf183.reset();  // reuse
    cpp_fused_mul_37((float*)(buf184.data_ptr()), (bfloat16*)(L__self___output_scales.data_ptr()), (float*)(buf10.data_ptr()), (float*)(buf12.data_ptr()), (float*)(buf40.data_ptr()), (float*)(buf42.data_ptr()), (float*)(buf70.data_ptr()), (float*)(buf72.data_ptr()), (float*)(buf100.data_ptr()), (float*)(buf102.data_ptr()), (float*)(buf130.data_ptr()), (float*)(buf132.data_ptr()), (float*)(buf160.data_ptr()), (float*)(buf162.data_ptr()), (float*)(L__self___layers_0_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_0_attention_kv_cache_v_cache.data_ptr()), (float*)(L__self___layers_1_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_1_attention_kv_cache_v_cache.data_ptr()), (float*)(L__self___layers_2_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_2_attention_kv_cache_v_cache.data_ptr()), (float*)(L__self___layers_3_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_3_attention_kv_cache_v_cache.data_ptr()), (float*)(L__self___layers_4_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_4_attention_kv_cache_v_cache.data_ptr()), (float*)(L__self___layers_5_attention_kv_cache_k_cache.data_ptr()), (float*)(L__self___layers_5_attention_kv_cache_v_cache.data_ptr()), s0);
    output_handles[0] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf184));
} // AOTInductorModel::run_impl
} // namespace aot_inductor
} // namespace torch

// Compile cmd
// clang++ /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/cxnpmgqddvot4sxjyjjiyasrt2nw6i6ip5c6e5heccg3jobhn7g2.cpp -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -Werror=ignored-optimization-argument -D_GLIBCXX_USE_CXX11_ABI=0 -I/Users/mikekg/pytorch/pytorch/torch/include -I/Users/mikekg/pytorch/pytorch/torch/include/torch/csrc/api/include -I/Users/mikekg/pytorch/pytorch/torch/include/TH -I/Users/mikekg/pytorch/pytorch/torch/include/THC -I/Users/mikekg/miniconda3/envs/py311/include/python3.11 -I/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd -DCPU_CAPABILITY_NEON -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -Xclang -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -c -o /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/cxnpmgqddvot4sxjyjjiyasrt2nw6i6ip5c6e5heccg3jobhn7g2.o
// Link cmd
// clang++ /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/cxnpmgqddvot4sxjyjjiyasrt2nw6i6ip5c6e5heccg3jobhn7g2.o /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/clao3uuulhp55qvru3odpuqyjekkp2p7xmr3epbkgn5tgy4prx4f.o -shared -fPIC -undefined dynamic_lookup -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -Werror=ignored-optimization-argument -D_GLIBCXX_USE_CXX11_ABI=0 -I/Users/mikekg/pytorch/pytorch/torch/include -I/Users/mikekg/pytorch/pytorch/torch/include/torch/csrc/api/include -I/Users/mikekg/pytorch/pytorch/torch/include/TH -I/Users/mikekg/pytorch/pytorch/torch/include/THC -I/Users/mikekg/miniconda3/envs/py311/include/python3.11 -I/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd -L/Users/mikekg/pytorch/pytorch/torch/lib -lomp -lc10 -DCPU_CAPABILITY_NEON -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -Xclang -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -o /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/stories15M_int8.so

// Compile cmd
// clang++ /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/cxnpmgqddvot4sxjyjjiyasrt2nw6i6ip5c6e5heccg3jobhn7g2.cpp -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -Werror=ignored-optimization-argument -D_GLIBCXX_USE_CXX11_ABI=0 -I/Users/mikekg/pytorch/pytorch/torch/include -I/Users/mikekg/pytorch/pytorch/torch/include/torch/csrc/api/include -I/Users/mikekg/pytorch/pytorch/torch/include/TH -I/Users/mikekg/pytorch/pytorch/torch/include/THC -I/Users/mikekg/miniconda3/envs/py311/include/python3.11 -I/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd -DCPU_CAPABILITY_NEON -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -Xclang -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -c -o /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/cxnpmgqddvot4sxjyjjiyasrt2nw6i6ip5c6e5heccg3jobhn7g2.o
// Link cmd
// clang++ /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/cxnpmgqddvot4sxjyjjiyasrt2nw6i6ip5c6e5heccg3jobhn7g2.o /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/clao3uuulhp55qvru3odpuqyjekkp2p7xmr3epbkgn5tgy4prx4f.o -shared -fPIC -undefined dynamic_lookup -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -Werror=ignored-optimization-argument -D_GLIBCXX_USE_CXX11_ABI=0 -I/Users/mikekg/pytorch/pytorch/torch/include -I/Users/mikekg/pytorch/pytorch/torch/include/torch/csrc/api/include -I/Users/mikekg/pytorch/pytorch/torch/include/TH -I/Users/mikekg/pytorch/pytorch/torch/include/THC -I/Users/mikekg/miniconda3/envs/py311/include/python3.11 -I/var/folders/k8/wvpsw3ln58d5kh_yxsrc6y740000gn/T/torchinductor_mikekg/nd -L/Users/mikekg/pytorch/pytorch/torch/lib -lomp -lc10 -DCPU_CAPABILITY_NEON -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -Xclang -fopenmp -D C10_USING_CUSTOM_GENERATED_MACROS -o /Users/mikekg/pytorch/llama-fast/checkpoints/stories15M/stories15M_int8.so
