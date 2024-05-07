#pragma once

#include <torch/library.h>
#include <torch/script.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>

c10::intrusive_ptr<at::native::xnnpack::LinearOpContext> prepack(
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    const c10::optional<at::Scalar>& output_min,
    const c10::optional<at::Scalar>& output_max);

torch::Tensor run(const torch::Tensor& input, const c10::intrusive_ptr<at::native::xnnpack::LinearOpContext>& op_context);

torch::Tensor prepack_and_run(
    const torch::Tensor& input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    const c10::optional<at::Scalar>& output_min,
    const c10::optional<at::Scalar>& output_max);
