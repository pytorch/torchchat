#include <torch/library.h>
#include <torch/script.h>
#include <ATen/native/xnnpack/Common.h>

int main() {

    xnn_status status;
    // status = xnn_initialize(/*allocator=*/nullptr);
    // TORCH_CHECK(status == xnn_status_success);

    auto w_col = 384;
    auto input_channels = w_col*2;
    auto output_channels = 32000;
    auto group_size = 32;
    auto n_groups = 24;
    TORCH_CHECK(n_groups * group_size == input_channels);

    // auto options = torch::TensorOptions().dtype(torch::kByte);
    // auto weight = torch::ones({output_channels, w_col}, options);
    // auto weight_scales = torch::ones({output_channels, n_groups});

    auto weight_data = std::vector<uint8_t>();
    for (int i = 0; i < output_channels * w_col; ++i) {
        weight_data.push_back(1);
    }

    auto weight_scales = std::vector<float>();
    for (int i = 0; i < output_channels * n_groups; ++i) {
        weight_data.push_back(1.0);
    }

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
    weight_scales.data(), /*const float* kernel_scale*/
    (void*)weight_data.data(), /*const void* kernel*/
    nullptr, /*const float* bias*/
    output_min, /*float output_min*/
    output_max, /*float output_max*/
    0, /*uint32_t flags*/
    nullptr, /*xnn_code_cache_t code_cache*/
    nullptr, /*xnn_weights_cache_t weights_cache*/
    &fc_op /*xnn_operator_t* fully_connected_op_out*/
);

}
