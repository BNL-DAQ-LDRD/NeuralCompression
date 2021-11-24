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
    torch::jit::script::Module encoder;
    try {
        encoder = torch::jit::load(argv[1]);
        std::cout << "loading encoder: ok\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the encoder\n";
        return -1;
    }

    // Encoding
    std::vector<torch::jit::IValue> encoder_inputs;
    encoder_inputs.push_back(torch::randn({32, 1, 192, 249, 16}));

    at::Tensor code = encoder.forward(encoder_inputs).toTensor();
    std::cout << code.sizes() << std::endl;
}
