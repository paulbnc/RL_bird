def conv_out_size(size, kernel_size=3, stride=2, padding=1):
    return (size + 2 * padding - kernel_size) // stride + 1
