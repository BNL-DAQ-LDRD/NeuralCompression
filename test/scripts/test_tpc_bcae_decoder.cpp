#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: test_tpc_bcae_decoder " <<
        "<path-to-scripted-decoder>." << std::endl;
        return -1;
    }


    // Load BCAE decoder:
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[1]);
        std::cout << "loading decoder: ok\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the decoder\n";
        return -1;
    }

    // Decoding
    std::vector<torch::jit::IValue> input;
    input.push_back(torch::randn({32, 8, 12, 15, 16}));

    at::Tensor output = model.forward(input).toTensor();
    std::cout << output.sizes() << std::endl;
}
