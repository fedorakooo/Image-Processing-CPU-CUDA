#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include "../utils/image_utils.h"

__global__ void pixelizeKernel(Pixel* input, Pixel* output, int width, int height, int blockSize) {
    int blockX = blockIdx.x * blockDim.x + threadIdx.x;
    int blockY = blockIdx.y * blockDim.y + threadIdx.y;

    int startX = blockX * blockSize;
    int startY = blockY * blockSize;

    if (startX >= width || startY >= height) return;

    int endX = min(startX + blockSize, width);
    int endY = min(startY + blockSize, height);

    unsigned long long sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            int idx = y * width + x;
            sumR += input[idx].r;
            sumG += input[idx].g;
            sumB += input[idx].b;
            count++;
        }
    }

    if (count == 0) return;

    Pixel avg;
    avg.r = sumR / count;
    avg.g = sumG / count;
    avg.b = sumB / count;

    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            int idx = y * width + x;
            output[idx] = avg;
        }
    }
}

std::vector<Pixel> pixelizeImageCUDA(const std::vector<Pixel>& input, int width, int height, int blockSize) {
    Pixel *d_input, *d_output;
    size_t size = input.size() * sizeof(Pixel);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + blockSize * threadsPerBlock.x - 1) / (blockSize * threadsPerBlock.x),
        (height + blockSize * threadsPerBlock.y - 1) / (blockSize * threadsPerBlock.y)
    );

    pixelizeKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, blockSize);
    cudaDeviceSynchronize();

    std::vector<Pixel> result(input.size());
    cudaMemcpy(result.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

int main() {
    std::string inputPath, outputPath;
    int blockSize;

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

    std::cout << "Enter pixelization block size: ";
    std::cin >> blockSize;

    if (blockSize < 1) {
        std::cerr << "Block size must be a positive number" << std::endl;
        return 1;
    }

    std::cout << "Loading image: " << inputPath << std::endl;

    int width, height;
    auto image = loadImage(inputPath, width, height);

    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << std::endl;
    std::cout << "Pixelizing image with block size " << blockSize << "x" << blockSize << " on GPU" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto pixelized = pixelizeImageCUDA(image, width, height, blockSize);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Pixelization completed in " << duration.count() << " ns" << std::endl;

    if (saveImage(outputPath, pixelized, width, height)) {
        std::cout << "Image successfully saved!" << std::endl;
    } else {
        std::cerr << "Error saving image" << std::endl;
        return 1;
    }

    return 0;
}
