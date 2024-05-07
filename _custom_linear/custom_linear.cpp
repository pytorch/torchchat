#include <torch/library.h>
#include <torch/script.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <torchchat/_custom_linear/custom_linear.h>


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

TORCH_LIBRARY(torchchat, m) {
  m.def("prepack", prepack);
  m.def("run", run);
  m.def("prepack_and_run", prepack_and_run);
}
