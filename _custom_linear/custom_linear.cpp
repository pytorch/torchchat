#include <torch/library.h>
#include <torch/script.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <torchchat/_custom_linear/custom_linear.h>


// Used for: make_zero_points_and_scales_tensor
// #include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>


#include <ATen/native/quantized/cpu/XnnpackUtils.h>


c10::intrusive_ptr<at::native::xnnpack::LinearOpContext> prepack(
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    const c10::optional<at::Scalar>& output_min,
    const c10::optional<at::Scalar>& output_max) {
  return at::native::xnnpack::XNNPackLinearOpContext::create_context(
      std::move(weight), std::move(bias), output_min, output_max);
}

torch::Tensor run(const torch::Tensor& input, const c10::intrusive_ptr<at::native::xnnpack::LinearOpContext>& op_context) {
    return op_context->run(input);
}

torch::Tensor prepack_and_run(
    const torch::Tensor& input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    const c10::optional<at::Scalar>& output_min,
    const c10::optional<at::Scalar>& output_max) {
    auto prepacked_op_context = prepack(weight, bias, output_min, output_max);
    return run(input, prepacked_op_context);
}

at::Tensor prepack_and_run_qd8_f32_qb4w(
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor input,
    int64_t group_size) {

        TORCH_CHECK(group_size >= 1, "group_size must be >= 1");
        TORCH_CHECK((group_size&(group_size-1)) == 0, "group_size must be a power of 2");

        xnn_status status;

        status = xnn_initialize(/*allocator=*/nullptr);
        TORCH_CHECK(status == xnn_status_success);

        const float output_min = -std::numeric_limits<float>::infinity();
        const float output_max = std::numeric_limits<float>::infinity();
        const uint8_t weight_zero_point = 8;

        auto input_channels = 2*weight.size(1); // Multiply by 2 because weights are packed
        auto output_channels = weight.size(0);

    std::cout << "input_channels: " << input_channels << std::endl;
    std::cout << "output_channels: " << output_channels << std::endl;
    std::cout << "group_size: " << group_size << std::endl;

// Create FC
xnn_operator_t fc_op = nullptr;
  status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
    input_channels, /*size_t input_channels*/
    output_channels, /*size_t output_channels*/
    input_channels, /*size_t input_stride*/
    output_channels, /*size_t output_stride*/
    group_size, /*size_t block_size*/
    weight_zero_point, /*uint8_t kernel_zero_point*/
    weight_scales.const_data_ptr<float>(), /*const float* kernel_scale*/
    weight.const_data_ptr(), /*const void* kernel*/
    nullptr, /*const float* bias*/
    output_min, /*float output_min*/
    output_max, /*float output_max*/
    0, /*uint32_t flags*/
    nullptr, /*xnn_code_cache_t code_cache*/
    nullptr, /*xnn_weights_cache_t weights_cache*/
    &fc_op /*xnn_operator_t* fully_connected_op_out*/
);
TORCH_CHECK(status == xnn_status_success, "Operator xnn_create_fully_connected_nc_qd8_f32_qb4w failed with status ", status, ".");
TORCH_CHECK(fc_op != nullptr);

// std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(fc_op, xnn_delete_operator);


// Create, reshape, setup, and run convert
TORCH_CHECK(input.dim() == 2);
auto batch_size = input.size(0);

// Holds output of convert
std::vector<int8_t> output_convert(batch_size * input_channels + XNN_EXTRA_BYTES);
std::vector<xnn_dynamic_quantization_params> quantization_params(batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_f32_qd8(
  0, /*uint32_t flags*/
  &convert_op /*xnn_operator_t* convert_op_out*/
 );
 TORCH_CHECK(status == xnn_status_success, "Operator xnn_create_convert_nc_f32_qd8 failed with status ", status, ".");
 TORCH_CHECK(convert_op != nullptr);

status = xnn_reshape_convert_nc_f32_qd8(
  convert_op, /*xnn_operator_t convert_op*/
  batch_size, /*size_t batch_size*/
  input_channels, /*size_t channels*/
  input_channels, /*size_t input_stride*/
  input_channels, /*size_t output_stride*/
  nullptr /*pthreadpool_t threadpool*/
);
 TORCH_CHECK(status == xnn_status_success, "Operator xnn_reshape_convert_nc_f32_qd8 failed with status ", status, ".");


 status = xnn_setup_convert_nc_f32_qd8(
  convert_op, /*xnn_operator_t convert_op*/
 input.const_data_ptr<float>(), /*const float* input*/
 output_convert.data(), /*int8_t* output*/
  quantization_params.data() /*struct xnn_dynamic_quantization_params* quantization_params*/
  );
  TORCH_CHECK(status == xnn_status_success, "Operator xnn_setup_convert_nc_f32_qd8 failed with status ", status, ".");

  status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
  TORCH_CHECK(status == xnn_status_success, "Running convert_op failed with status ", status, ".");



 // Reshape, setup, and run FC
  status = xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
    fc_op, /*xnn_operator_t fully_connected_op*/
    batch_size, /*size_t batch_size*/
    nullptr /*pthreadpool_t threadpool*/ // TODO: set to something sensible
    );
TORCH_CHECK(status == xnn_status_success, "Operator xnn_reshape_fully_connected_nc_qd8_f32_qb4w failed with status ", status, ".");

// Create tensor to hold output
auto options = torch::TensorOptions().dtype(torch::kFloat32);
auto output_tensor = torch::empty({batch_size, output_channels}, options);

 status = xnn_setup_fully_connected_nc_qd8_f32_qb4w(
    fc_op, /*xnn_operator_t fully_connected_op*/
    output_convert.data(), /*const int8_t* input*/
    output_tensor.data_ptr<float>(), /*float* output*/
    quantization_params.data() /*const struct xnn_dynamic_quantization_params* quantization_params*/
 );
 TORCH_CHECK(status == xnn_status_success, "Operator xnn_setup_fully_connected_nc_qd8_f32_qb4w failed with status ", status, ".");


status = xnn_run_operator(fc_op, /*threadpool=*/nullptr);
TORCH_CHECK(status == xnn_status_success, "Running fc_op failed with status ", status, ".");

std::cout << "RETURNING." << std::endl;

return output_tensor;
}




TORCH_LIBRARY(torchchat, m) {
  m.def("prepack", prepack);
  m.def("run", run);
  m.def("prepack_and_run", prepack_and_run);
  m.def("prepack_and_run_qd8_f32_qb4w", prepack_and_run_qd8_f32_qb4w);
}
