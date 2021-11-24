#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: test_tpc_blocks " <<
        "<path-to-scripted-encoder> " <<
        "<path-to-scripted-decoder>." << std::endl;
        return -1;
    }


    // Load encoder and decoder:
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module encoder;
    try {
        encoder = torch::jit::load(argv[1]);
        std::cout << "loading encoder: ok\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the encoder\n";
        return -1;
    }

    torch::jit::script::Module decoder;
    try {
        decoder = torch::jit::load(argv[2]);
        std::cout << "loading decoder: ok\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the decoder\n";
        return -1;
    }


    // Encoding
    std::vector<torch::jit::IValue> encoder_inputs;
    encoder_inputs.push_back(torch::randn({32, 8, 16, 16, 16}));

    at::Tensor code = encoder.forward(encoder_inputs).toTensor();
    std::cout << code.sizes() << std::endl;

    // Decoding
    std::vector<torch::jit::IValue> decoder_input;
    decoder_input.push_back(code);

    at::Tensor decoded = decoder.forward(decoder_input).toTensor();
    std::cout << decoded.sizes() << std::endl;
}
