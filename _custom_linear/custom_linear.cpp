#include <torch/library.h>
#include <torch/script.h>
#include <ATen/native/xnnpack/Common.h>


class XnnpackOperatorClass : public torch::jit::CustomClassHolder {
 private:
  at::native::xnnpack::Operator op_;

public:
  XnnpackOperatorClass(at::native::xnnpack::Operator op) : op_(std::move(op)) {}
  xnn_operator_t get() {
    return op_.get();
  }
};


c10::intrusive_ptr<XnnpackOperatorClass> create_fully_connected_nc_qd8_f32_qb4w(
    at::Tensor weight,
    at::Tensor weight_scales) {

        TORCH_CHECK(weight.dim() == 2, "weight must be 2-dimensional");
        TORCH_CHECK(weight.size(1) % 2 == 0, "weight columns must be even (packed int4)");

        TORCH_CHECK(weight_scales.dim() == 2, "weight_scales must be 2-dimensional");
        TORCH_CHECK(weight.size(0) == weight_scales.size(0), "weight and weight_scale must have same number of rows");

        const float output_min = -std::numeric_limits<float>::infinity();
        const float output_max = std::numeric_limits<float>::infinity();
        const uint8_t weight_zero_point = 8;

        auto input_channels = 2*weight.size(1); // Multiply by 2 because weights are packed
        auto output_channels = weight.size(0);


        TORCH_CHECK((input_channels % weight_scales.size(1)) == 0, "number of columns in weight_scales should divide input_channels");
        size_t group_size = input_channels / weight_scales.size(1);
        TORCH_CHECK(group_size > 1, "inferred group_size must be > 1");
        TORCH_CHECK((group_size&(group_size-1)) == 0, "inferred group_size must be a power of 2");

// Create FC
xnn_operator_t fc_op = nullptr;
  auto status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
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

   return c10::make_intrusive<XnnpackOperatorClass>(at::native::xnnpack::Operator(fc_op));
}


c10::intrusive_ptr<XnnpackOperatorClass> create_convert_nc_f32_qd8() {
    xnn_operator_t convert_op = nullptr;
  auto status = xnn_create_convert_nc_f32_qd8(
  0, /*uint32_t flags*/
  &convert_op /*xnn_operator_t* convert_op_out*/
 );
 TORCH_CHECK(status == xnn_status_success, "Operator xnn_create_convert_nc_f32_qd8 failed with status ", status, ".");
 TORCH_CHECK(convert_op != nullptr);
 return c10::make_intrusive<XnnpackOperatorClass>(at::native::xnnpack::Operator(convert_op));
}


at::Tensor run_linear_qd8_f32_qb4w(c10::intrusive_ptr<XnnpackOperatorClass>  fc_op, c10::intrusive_ptr<XnnpackOperatorClass>  convert_op, int64_t output_channels, at::Tensor input) {
    TORCH_CHECK(input.dim() == 2);

auto batch_size = input.size(0);
auto input_channels = input.size(1);
xnn_status status;

// Holds output of convert
std::vector<int8_t> output_convert(batch_size * input_channels + XNN_EXTRA_BYTES);
std::vector<xnn_dynamic_quantization_params> quantization_params(batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

// Run input convert
status = xnn_reshape_convert_nc_f32_qd8(
  convert_op->get(), /*xnn_operator_t convert_op*/
  batch_size, /*size_t batch_size*/
  input_channels, /*size_t channels*/
  input_channels, /*size_t input_stride*/
  input_channels, /*size_t output_stride*/
  nullptr /*pthreadpool_t threadpool*/
);
 TORCH_CHECK(status == xnn_status_success, "Operator xnn_reshape_convert_nc_f32_qd8 failed with status ", status, ".");

 status = xnn_setup_convert_nc_f32_qd8(
  convert_op->get(), /*xnn_operator_t convert_op*/
 input.const_data_ptr<float>(), /*const float* input*/
 output_convert.data(), /*int8_t* output*/
  quantization_params.data() /*struct xnn_dynamic_quantization_params* quantization_params*/
  );
  TORCH_CHECK(status == xnn_status_success, "Operator xnn_setup_convert_nc_f32_qd8 failed with status ", status, ".");

  status = xnn_run_operator(convert_op->get(), /*threadpool=*/nullptr);
  TORCH_CHECK(status == xnn_status_success, "Running convert_op failed with status ", status, ".");


// Holds output of linear
auto options = torch::TensorOptions().dtype(torch::kFloat32);
auto output_tensor = torch::empty({batch_size, output_channels}, options);

 // Run linear
  status = xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
    fc_op->get(), /*xnn_operator_t fully_connected_op*/
    batch_size, /*size_t batch_size*/
    nullptr /*pthreadpool_t threadpool*/ // TODO: set to something sensible
    );
    TORCH_CHECK(status == xnn_status_success, "Operator xnn_reshape_fully_connected_nc_qd8_f32_qb4w failed with status ", status, ".");

 status = xnn_setup_fully_connected_nc_qd8_f32_qb4w(
    fc_op->get(), /*xnn_operator_t fully_connected_op*/
    output_convert.data(), /*const int8_t* input*/
    output_tensor.data_ptr<float>(), /*float* output*/
    quantization_params.data() /*const struct xnn_dynamic_quantization_params* quantization_params*/
 );
 TORCH_CHECK(status == xnn_status_success, "Operator xnn_setup_fully_connected_nc_qd8_f32_qb4w failed with status ", status, ".");


status = xnn_run_operator(fc_op->get(), /*threadpool=*/nullptr);
TORCH_CHECK(status == xnn_status_success, "Running fc_op failed with status ", status, ".");

return output_tensor;
}

at::Tensor create_and_run(
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor input) {

    auto status = xnn_initialize(/*allocator=*/nullptr);
    TORCH_CHECK(status == xnn_status_success);
    auto fc_op = create_fully_connected_nc_qd8_f32_qb4w(weight, weight_scales);
    auto convert_op = create_convert_nc_f32_qd8();
    auto output_channels = weight.size(0);
    return run_linear_qd8_f32_qb4w(std::move(fc_op), std::move(convert_op), output_channels, input);
}


TORCH_LIBRARY(torchchat, m) {
    m.class_<XnnpackOperatorClass>("XnnpackOperatorClass");
    m.def("create_and_run", create_and_run);
    m.def("create_fully_connected_nc_qd8_f32_qb4w", create_fully_connected_nc_qd8_f32_qb4w);
     m.def("create_convert_nc_f32_qd8", create_convert_nc_f32_qd8);
     m.def("run_linear_qd8_f32_qb4w", run_linear_qd8_f32_qb4w);
}
