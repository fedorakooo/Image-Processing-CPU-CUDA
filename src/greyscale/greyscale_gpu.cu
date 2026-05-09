#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#include "../utils/image_utils.h"

struct PixelGpu {
    unsigned char r, g, b;
};

__global__
void greyscaleKernel(
    const PixelGpu* input,
    PixelGpu* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    unsigned char grey = static_cast<unsigned char>(0.2126f * input[idx].r + 0.7152f * input[idx].g + 0.0722f * input[idx].b);

    output[idx] = {grey, grey, grey};
}

std::vector<Pixel> greyscaleImageCUDA(
    const std::vector<Pixel>& input,
    int width,
    int height
) {
    size_t count = width * height;

    PixelGpu* d_input;
    PixelGpu* d_output;

    size_t bytes = count * sizeof(PixelGpu);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    greyscaleKernel<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    std::vector<PixelGpu> gpuOut(count);
    cudaMemcpy(gpuOut.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    std::vector<Pixel> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = { gpuOut[i].r, gpuOut[i].g, gpuOut[i].b };
    }

    return result;
}

int main() {
    std::string inputPath, outputPath;

    std::cout << "Enter input image path: ";
    std::getline(std::cin, inputPath);

    if (!std::filesystem::exists(inputPath)) {
        std::cerr << "File not found: " << inputPath << std::endl;
        return 1;
    }

    std::cout << "Enter output image path: ";
    std::getline(std::cin, outputPath);

    std::filesystem::path outputDir = std::filesystem::path(outputPath).parent_path();
    if (!outputDir.empty() && !std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }

    int width, height;
    std::cout << "Loading image..." << std::endl;
    auto image = loadImage(inputPath, width, height);

    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto grey = greyscaleImageCUDA(image, width, height);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Greyscale conversion on GPU completed in " << duration.count() << " ns\n";

    if (saveImage(outputPath, grey, width, height)) {
        std::cout << "Saved successfully.\n";
    } else {
        std::cout << "Error writing image.\n";
    }

    return 0;
}
