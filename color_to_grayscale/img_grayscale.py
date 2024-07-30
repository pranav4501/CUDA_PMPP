
import torch
from torch.utils.cpp_extension import load_inline
from pathlib import Path
import os
from torchvision.io import read_image, write_jpeg


cuda_source = Path("./cuda_img_grayscale.cu").read_text()

cpp_source = "torch::Tensor color_to_grayscale(torch::Tensor image);"


os.makedirs("./cuda_build", exist_ok=True)
grayscale_extension = load_inline(
    name='grayscale_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['color_to_grayscale'],
    with_cuda=True,
    extra_cflags=["-O2"],
    build_directory='./cuda_build'
)

input_image = read_image("real.jpg").permute(1, 2, 0).cuda()
print(input_image.shape, input_image.dtype)
output_image = grayscale_extension.color_to_grayscale(input_image)
print(output_image.shape, output_image.dtype)
write_jpeg(output_image.permute(2, 0, 1).cpu(), "./real_gray.jpeg")


