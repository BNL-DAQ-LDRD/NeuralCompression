#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: test_tpc_bcae " <<
        "<path-to-scripted-bcae>." << std::endl;
        return -1;
    }


    // Load bcae
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[1]);
        std::cout << "loading bcae: ok\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the bcae\n";
        return -1;
    }

    // Encoding
    std::vector<torch::jit::IValue> input;
    input.push_back(torch::randn({32, 1, 192, 249, 16}));

    auto output = model.forward(input).toTuple();
    torch::Tensor output_c = output->elements()[0].toTensor();
    torch::Tensor output_r = output->elements()[1].toTensor();
    std::cout << output_c.sizes() << std::endl;
    std::cout << output_r.sizes() << std::endl;
}
