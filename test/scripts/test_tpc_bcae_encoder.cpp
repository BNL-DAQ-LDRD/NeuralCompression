#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: test_tpc_bcae_encoder " <<
        "<path-to-scripted-encoder>." << std::endl;
        return -1;
    }


    // Load bcae encoder
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[1]);
        std::cout << "loading encoder: ok\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the encoder\n";
        return -1;
    }

    // Encoding
    std::vector<torch::jit::IValue> input;
    input.push_back(torch::randn({32, 1, 192, 249, 16}));

    at::Tensor output = model.forward(input).toTensor();
    std::cout << output.sizes() << std::endl;
}
