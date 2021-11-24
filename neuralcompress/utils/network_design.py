#! /usr/bin/env python
import numpy as np

class network_design:
    def __conv(self, i, f, s, p):
        return int((i + 2 * p - f) / s) + 1

    def __convTrans(self, i, f, s, p):
        return (i - 1) * s - 2 * p + f

    def get_output_shape_conv(self, input_shape, kernel_size, stride, padding=None):
        if padding is None:
            padding = [f // 2 for f in kernel_size]
        output_shape = []
        for i, f, s, p in zip(input_shape, kernel_size, stride, padding):
            output_dim = self.__conv(i, f, s, p)
            output_shape.append(output_dim)
        return np.array(output_shape)

    def get_output_shape_convTrans(self, input_shape, kernel_size, stride, padding=None):
        if padding is None:
            padding = [f // 2 for f in kernel_size]
        output_shape = []
        for i, f, s, p in zip(input_shape, kernel_size, stride, padding):
            output_dim = self.__convTrans(i, f, s, p)
            output_shape.append(output_dim)
        return np.array(output_shape)


    def get_output_padding(self, input_shape, kernel_sizes, strides, paddings=None):
        input_shape = np.array(input_shape)
        assert len(kernel_sizes) == len(strides)
        if paddings is None:
            paddings = [[f // 2 for f in ks] for ks in kernel_sizes]

        output_paddings = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]

            output_shape = self.get_output_shape_conv(input_shape, kernel_size, stride, padding)
            input_shape_inverse = self.get_output_shape_convTrans(output_shape, kernel_size, stride, padding)

            # Torch scripting for C++ cannot handle numpy data type,
            # and hence we have to cast each entry to int from numpy.int64
            temp = list(input_shape - input_shape_inverse)
            output_padding = [int(t) for t in temp]
            output_paddings.append(output_padding)

            print(f'layer {i + 1}:\n\t{input_shape} -> {output_shape}\
                \n\t{input_shape_inverse}, output_padding={output_padding}\n')

            input_shape = output_shape

        return output_paddings

if __name__ == '__main__':
    input_shape = [192, 249, 16]
    filters = [[4, 5, 3], [5, 5, 3], [3, 4, 3]]
    strides = [[4, 4, 1], [2, 2, 1], [2, 2, 1]]
    output_paddings = network_design().get_output_padding(input_shape, filters, strides)
