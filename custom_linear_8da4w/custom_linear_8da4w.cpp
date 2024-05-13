#include <ATen/native/xnnpack/Common.h>
// #include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torchchat/custom_linear_8da4w/threadpool/threadpool.h>
#include <torch/library.h>
#include <torch/script.h>


#define CUSTOM_LINEAR_THREAD_POOL torchchat::threadpool::get_pthreadpool()

at::native::xnnpack::Operator create_fully_connected_nc_qd8_f32_qb4w(
    at::Tensor weight,
    at::Tensor weight_scales) {
  TORCH_CHECK(weight.dim() == 2, "weight must be 2-dimensional");
  TORCH_CHECK(
      weight.size(1) % 2 == 0, "weight columns must be even (packed int4)");

  TORCH_CHECK(weight_scales.dim() == 2, "weight_scales must be 2-dimensional");
  TORCH_CHECK(
      weight.size(0) == weight_scales.size(0),
      "weight and weight_scale must have same number of rows");

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();
  const uint8_t weight_zero_point = 8;

  auto input_channels =
      2 * weight.size(1); // Multiply by 2 because weights are packed
  auto output_channels = weight.size(0);

  TORCH_CHECK(
      (input_channels % weight_scales.size(1)) == 0,
      "number of columns in weight_scales should divide input_channels");
  size_t group_size = input_channels / weight_scales.size(1);
  TORCH_CHECK(group_size > 1, "inferred group_size must be > 1");
  TORCH_CHECK(
      (group_size & (group_size - 1)) == 0,
      "inferred group_size must be a power of 2");

  auto weight_ptr = (void*)weight.const_data_ptr();
  auto weight_scales_ptr = weight_scales.const_data_ptr<float>();

  TORCH_CHECK(weight_ptr != nullptr);
  TORCH_CHECK(weight_scales_ptr != nullptr);

  xnn_operator_t fc_op = nullptr;
  auto status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
      input_channels, /*size_t input_channels*/
      output_channels, /*size_t output_channels*/
      input_channels, /*size_t input_stride*/
      output_channels, /*size_t output_stride*/
      group_size, /*size_t block_size*/
      weight_zero_point, /*uint8_t kernel_zero_point*/
      weight_scales_ptr, /*const float* kernel_scale*/
      weight_ptr, /*const void* kernel*/
      nullptr, /*const float* bias*/
      output_min, /*float output_min*/
      output_max, /*float output_max*/
      0, /*uint32_t flags*/
      nullptr, /*xnn_code_cache_t code_cache*/
      nullptr, /*xnn_weights_cache_t weights_cache*/
      &fc_op /*xnn_operator_t* fully_connected_op_out*/
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "Operator xnn_create_fully_connected_nc_qd8_f32_qb4w failed with status ",
      status,
      ".");
  TORCH_CHECK(fc_op != nullptr);

  return at::native::xnnpack::Operator(fc_op);
}

at::native::xnnpack::Operator create_convert_nc_f32_qd8() {
  xnn_operator_t convert_op = nullptr;
  auto status = xnn_create_convert_nc_f32_qd8(
      0, /*uint32_t flags*/
      &convert_op /*xnn_operator_t* convert_op_out*/
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "Operator xnn_create_convert_nc_f32_qd8 failed with status ",
      status,
      ".");
  TORCH_CHECK(convert_op != nullptr);
  return at::native::xnnpack::Operator(convert_op);
}

at::Tensor run_linear_qd8_f32_qb4w(
    xnn_operator_t convert_op,
    xnn_operator_t fc_op,
    int64_t output_channels,
    at::Tensor input) {
  TORCH_CHECK(input.dim() == 2);

  auto batch_size = input.size(0);
  auto input_channels = input.size(1);
  xnn_status status;

  // Holds output of convert
  std::vector<int8_t> output_convert(
      batch_size * input_channels + XNN_EXTRA_BYTES);
  std::vector<xnn_dynamic_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  // Run input convert
  status = xnn_reshape_convert_nc_f32_qd8(
      convert_op, /*xnn_operator_t convert_op*/
      batch_size, /*size_t batch_size*/
      input_channels, /*size_t channels*/
      input_channels, /*size_t input_stride*/
      input_channels, /*size_t output_stride*/
      CUSTOM_LINEAR_THREAD_POOL /*pthreadpool_t threadpool*/
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "Operator xnn_reshape_convert_nc_f32_qd8 failed with status ",
      status,
      ".");

  status = xnn_setup_convert_nc_f32_qd8(
      convert_op, /*xnn_operator_t convert_op*/
      input.const_data_ptr<float>(), /*const float* input*/
      output_convert.data(), /*int8_t* output*/
      quantization_params.data() /*struct xnn_dynamic_quantization_params*
                                    quantization_params*/
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "Operator xnn_setup_convert_nc_f32_qd8 failed with status ",
      status,
      ".");

  status =
      xnn_run_operator(convert_op, /*threadpool=*/CUSTOM_LINEAR_THREAD_POOL);
  TORCH_CHECK(
      status == xnn_status_success,
      "Running convert_op failed with status ",
      status,
      ".");

  // Holds output of linear
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto output_tensor = torch::empty({batch_size, output_channels}, options);

  // Run linear
  status = xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
      fc_op, /*xnn_operator_t fully_connected_op*/
      batch_size, /*size_t batch_size*/
      CUSTOM_LINEAR_THREAD_POOL /*pthreadpool_t threadpool*/ // TODO: set to
                                                             // something
                                                             // sensible
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "Operator xnn_reshape_fully_connected_nc_qd8_f32_qb4w failed with status ",
      status,
      ".");

  status = xnn_setup_fully_connected_nc_qd8_f32_qb4w(
      fc_op, /*xnn_operator_t fully_connected_op*/
      output_convert.data(), /*const int8_t* input*/
      output_tensor.data_ptr<float>(), /*float* output*/
      quantization_params.data() /*const struct xnn_dynamic_quantization_params*
                                    quantization_params*/
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "Operator xnn_setup_fully_connected_nc_qd8_f32_qb4w failed with status ",
      status,
      ".");

  status = xnn_run_operator(fc_op, /*threadpool=*/CUSTOM_LINEAR_THREAD_POOL);
  TORCH_CHECK(
      status == xnn_status_success,
      "Running fc_op failed with status ",
      status,
      ".");

  return output_tensor;
}

class PrepackedContext : public torch::jit::CustomClassHolder {
 private:
  at::native::xnnpack::Operator convert_op_;
  at::native::xnnpack::Operator fc_op_;
  size_t output_channels_;

  at::Tensor weight_;
  at::Tensor weight_scales_;

 public:
  PrepackedContext(
      at::native::xnnpack::Operator convert_op,
      at::native::xnnpack::Operator fc_op,
      size_t output_channels,
      at::Tensor weight,
      at::Tensor weight_scales)
      : convert_op_(std::move(convert_op)),
        fc_op_(std::move(fc_op)),
        output_channels_(output_channels),
        weight_(weight),
        weight_scales_(weight_scales) {}
  xnn_operator_t convert_op() {
    return convert_op_.get();
  }

  xnn_operator_t fc_op() {
    return fc_op_.get();
  }

  size_t output_channels() {
    return output_channels_;
  }
};

c10::intrusive_ptr<PrepackedContext> prepack(
    at::Tensor weight,
    at::Tensor weight_scales) {

  // auto status = xnn_initialize(/*allocator=*/nullptr);
  // TORCH_CHECK(status == xnn_status_success);
  bool is_initialized = at::native::xnnpack::available();
  TORCH_CHECK(is_initialized, "XNNPACK is not initialized");
  auto convert_op = create_convert_nc_f32_qd8();
  auto fc_op = create_fully_connected_nc_qd8_f32_qb4w(weight, weight_scales);
  auto output_channels = weight.size(0);

  return c10::make_intrusive<PrepackedContext>(
      at::native::xnnpack::Operator(std::move(convert_op)),
      at::native::xnnpack::Operator(std::move(fc_op)),
      output_channels,
      std::move(weight),
      std::move(weight_scales));
}

at::Tensor run(
    c10::intrusive_ptr<PrepackedContext> prepacked_context,
    at::Tensor input) {
  bool is_initialized = at::native::xnnpack::available();
  TORCH_CHECK(is_initialized, "XNNPACK is not initialized");

  return run_linear_qd8_f32_qb4w(
      prepacked_context->convert_op(),
      prepacked_context->fc_op(),
      prepacked_context->output_channels(),
      input);
}

at::Tensor
prepack_and_run(at::Tensor weight, at::Tensor weight_scales, at::Tensor input) {
  bool is_initialized = at::native::xnnpack::available();
  TORCH_CHECK(is_initialized, "XNNPACK is not initialized");

  auto prepacked_context = prepack(weight, weight_scales);
  return run(prepacked_context, input);
}

TORCH_LIBRARY(torchchat, m) {
  m.class_<PrepackedContext>("PrepackedContext");
  m.def("prepack", prepack);
  m.def("run", run);
  m.def("prepack_and_run", prepack_and_run);
}
