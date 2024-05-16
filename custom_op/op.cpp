#include <op.h>

int64_t get_key(
    at::Tensor weight) {
    uintptr_t weight_ptr = (uintptr_t)weight.data_ptr<float>();
    assert (weight_ptr < std::numeric_limits<int64_t>::max());
    return (int64_t)weight_ptr;
}

at::Tensor prepack(
    at::Tensor weight) {
    xnn_status status = xnn_initialize(/*allocator=*/nullptr);
    TORCH_CHECK(status == xnn_status_success, "failed to initialize XNNPACK");

    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = std::numeric_limits<float>::infinity();

    auto output_channels = weight.size(0);
    auto input_channels = weight.size(1);
    auto xnn_weights_cache = std::make_unique<XNNWeightsCache>();

    xnn_weights_cache->registerWeight("weight_1", weight.data_ptr<float>());

    xnn_operator_t linear_op = nullptr;

    xnn_status create_status = xnn_create_fully_connected_nc_f32(
        input_channels,                        // input_channels
        output_channels,                       // output_channels
        input_channels,                        // input_pixel_stride
        output_channels,                       // output_pixel_stride
        weight.contiguous().data_ptr<float>(), // kernel
        nullptr,                               // bias
        output_min,                            // output_min
        output_max,                            // output_max
        0u,                                    // flags
        nullptr,                               // xnn_caches_t
        xnn_weights_cache->Get(),               // xnn_weights_cache_t
        &linear_op);                           // operator

    TORCH_CHECK(
        xnn_status_success == create_status,
        "xnn_create_fully_connected_nc_f32 failed!", create_status);

    xnn_weights_cache->Finalize();

    const auto cache_tensor = xnn_weights_cache->to_tensor();
    return cache_tensor;
}

at::Tensor run(
    at::Tensor input,
    const at::Tensor& packed_weights,
    const int64_t input_channels,
    const int64_t output_channels,
    const int64_t weight_ptr) {

    auto batch_size = input.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto output = torch::empty({batch_size, (int64_t) output_channels}, options);

    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = std::numeric_limits<float>::infinity();

    auto xnn_weights_cache = std::make_unique<XNNWeightsCache>(packed_weights);

    xnn_operator_t linear_op = nullptr;
    const xnn_status create_status = xnn_create_fully_connected_nc_f32(
        (size_t) input_channels,               // input_channels
        (size_t) output_channels,              // output_channels
        input_channels,                        // input_pixel_stride
        output_channels,                       // output_pixel_stride
        (const float*) weight_ptr,             // kernel - should be only used as a key to the weight cache, else we will go belly up!
        nullptr,                               // bias
        output_min,                            // output_min
        output_max,                            // output_max
        0u,                                    // flags
        nullptr,                               // xnn_caches_t
        xnn_weights_cache->Get(),              // xnn_weights_cache_t
        &linear_op);                           // operator

    xnn_weights_cache->Finalize();

    const xnn_status reshape_status = xnn_reshape_fully_connected_nc_f32(
      linear_op,                              // operator
      batch_size,                             // batch_size
      nullptr);                               // threadpool

    TORCH_CHECK(
        xnn_status_success == reshape_status,
        "xnn_reshape_fully_connected_nc_f32 failed!", reshape_status);

    const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
      linear_op,                              // operator
      input.data_ptr<float>(),                // input
      output.data_ptr<float>());              // output

    TORCH_CHECK(
        xnn_status_success == setup_status,
        "xnn_setup_fully_connected_nc_f32 failed!", setup_status);

    const xnn_status run_status = xnn_run_operator(
      linear_op,                              // operator
      nullptr);                               // threadpool

    TORCH_INTERNAL_ASSERT(
        xnn_status_success == run_status,
        "xnn_run_operator failed!", run_status);

    return output;
}

at::Tensor prepack_and_run(
    at::Tensor input,
    at::Tensor weights) {
    int64_t weight_key = get_key(weights);
    auto packed_weights = prepack(weights);
    return run(input, packed_weights, weights.size(1), weights.size(0), weight_key);
}

TORCH_LIBRARY(customlinear, m) {
  m.def("get_weights_key", get_key);
  m.def("prepack_weights", prepack);
  m.def("run", run);
  m.def("prepack_and_run", prepack_and_run);
}
