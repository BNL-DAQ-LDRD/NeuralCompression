cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(test_tpc_blocks test_tpc_blocks.cpp)
target_link_libraries(test_tpc_blocks "${TORCH_LIBRARIES}")
set_property(TARGET test_tpc_blocks PROPERTY CXX_STANDARD 14)

add_executable(test_tpc_bcae_encoder test_tpc_bcae_encoder.cpp)
target_link_libraries(test_tpc_bcae_encoder "${TORCH_LIBRARIES}")
set_property(TARGET test_tpc_bcae_encoder PROPERTY CXX_STANDARD 14)

add_executable(test_tpc_bcae_decoder test_tpc_bcae_decoder.cpp)
target_link_libraries(test_tpc_bcae_decoder "${TORCH_LIBRARIES}")
set_property(TARGET test_tpc_bcae_decoder PROPERTY CXX_STANDARD 14)
