#include <torch/library.h>
#include <torch/script.h>
// #include <ATen/native/xnnpack/Common.h>

#include <xnnpack.h>
// #include <caffe2/utils/threadpool/pthreadpool-cpp.h>
// #include <c10/util/ArrayRef.h>
// #include <limits>
// #include <memory>

namespace at::native::xnnpack {

struct Deleter final {
  void operator()(const xnn_operator_t op) const {
    xnn_delete_operator(op);
  }
};

using Operator = std::unique_ptr<xnn_operator, Deleter>;

} // namespace at::native::xnnpack

int main() {
  xnn_status status;
  status = xnn_initialize(/*allocator=*/nullptr);
  TORCH_CHECK(status == xnn_status_success);

  auto w_col = 384;
  auto input_channels = w_col * 2;
  auto output_channels = 32000;
  auto group_size = 32;
  int n_groups = input_channels / group_size;
  TORCH_CHECK(n_groups * group_size == input_channels);

  auto options = torch::TensorOptions().dtype(torch::kByte);
  auto weight = torch::ones({output_channels, w_col}, options);
  auto weight_scales = torch::ones({output_channels, n_groups});

  auto weight_vector = std::vector<uint8_t>(weight.numel(), 0);
  for (int i = 0; i < weight.numel(); ++i) {
    weight_vector[i] = weight.const_data_ptr<uint8_t>()[i];
  }

  auto weight_scales_vector = std::vector<float>(weight_scales.numel(), 0);
  for (int i = 0; i < weight_scales.numel(); ++i) {
    weight_scales_vector[i] = weight_scales.const_data_ptr<float>()[i];
  }

  // auto weight_ptr = (void*)weight_vector.data();
  // auto weight_scales_ptr = weight_scales_vector.data();

  auto weight_ptr = (void*)weight.const_data_ptr();
  auto weight_scales_ptr = weight_scales.const_data_ptr<float>();

  TORCH_CHECK(weight_ptr != nullptr);
  TORCH_CHECK(weight_scales_ptr != nullptr);

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();
  const uint8_t weight_zero_point = 8;

  xnn_operator_t fc_op = nullptr;
  status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
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
}
