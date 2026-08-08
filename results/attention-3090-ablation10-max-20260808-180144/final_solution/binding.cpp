#include <torch/extension.h>

torch::Tensor attention_forward(torch::Tensor q, torch::Tensor k,
                                torch::Tensor v, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward,
          "Attention forward (BF16, [B,H,S,D], D=128)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal"));
}
