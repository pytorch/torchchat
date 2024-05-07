#include <torch/library.h>
#include <torch/script.h>
#include <ATen/native/xnnpack/Common.h>

torch::Tensor prepack(torch::Tensor weight, torch::Tensor bias) {
    return weight; // TODO: fill iin STUB
}

torch::Tensor run(torch::Tensor input, torch::Tensor packed_weight) {
    return packed_weight; // fill in STUB
}
