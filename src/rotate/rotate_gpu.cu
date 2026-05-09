#include <iostream>
#include <filesystem>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#include "../utils/image_utils.h"

struct RotationParams {
    double angle_rad;
    double center_x;
    double center_y;
    Pixel background;
};

__device__ Pixel getPixelSafe(
    const Pixel* input, 
    int width, 
    int height, 
    int x, 
    int y, 
    Pixel background
) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return input[y * width + x];
    }
    return background;
}

__global__ void rotateKernel(
    const Pixel* input, 
    Pixel* output, 
    int width, 
    int height, 
    double angle_rad,
    double center_x,
    double center_y,
    unsigned char bg_r,
    unsigned char bg_g,
    unsigned char bg_b
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    Pixel background = {bg_r, bg_g, bg_b};
    
    double x_rel = x - center_x;
    double y_rel = y - center_y;
    
    double sin_angle = sin(angle_rad);
    double cos_angle = cos(angle_rad);
    
    double src_x = x_rel * cos_angle + y_rel * sin_angle + center_x;
    double src_y = -x_rel * sin_angle + y_rel * cos_angle + center_y;
    
    int x0 = static_cast<int>(floor(src_x));
    int y0 = static_cast<int>(floor(src_y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    double dx = src_x - x0;
    double dy = src_y - y0;
    
    Pixel p00 = getPixelSafe(input, width, height, x0, y0, background);
    Pixel p10 = getPixelSafe(input, width, height, x1, y0, background);
    Pixel p01 = getPixelSafe(input, width, height, x0, y1, background);
    Pixel p11 = getPixelSafe(input, width, height, x1, y1, background);
    
    Pixel result;
    double r = p00.r * (1 - dx) * (1 - dy) + p10.r * dx * (1 - dy) + p01.r * (1 - dx) * dy + p11.r * dx * dy;
    double g = p00.g * (1 - dx) * (1 - dy) + p10.g * dx * (1 - dy) + p01.g * (1 - dx) * dy + p11.g * dx * dy;
    double b = p00.b * (1 - dx) * (1 - dy) + p10.b * dx * (1 - dy) + p01.b * (1 - dx) * dy + p11.b * dx * dy;
    
    result.r = static_cast<unsigned char>(fmin(fmax(r, 0.0), 255.0));
    result.g = static_cast<unsigned char>(fmin(fmax(g, 0.0), 255.0));
    result.b = static_cast<unsigned char>(fmin(fmax(b, 0.0), 255.0));
    
    output[y * width + x] = result;
}

std::vector<Pixel> rotateImageCUDA(
    const std::vector<Pixel>& input, 
    int width, 
    int height, 
    double angle_degrees,
    Pixel background = {0, 0, 0}
) {
    std::vector<Pixel> output(width * height);
    
    double angle_rad = angle_degrees * M_PI / 180.0;
    double center_x = (width - 1) / 2.0;
    double center_y = (height - 1) / 2.0;
    
    RotationParams params{angle_rad, center_x, center_y, background};
    
    Pixel *d_input, *d_output;
    size_t size = input.size() * sizeof(Pixel);
    
    cudaError_t err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error (input): " << cudaGetErrorString(err) << std::endl;
        return output;
    }
    
    err = cudaMalloc(&d_output, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error (output): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return output;
    }
    
    err = cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error (input): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return output;
    }
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    rotateKernel<<<numBlocks, threadsPerBlock>>>(
        d_input, 
        d_output, 
        width, 
        height, 
        params.angle_rad,
        params.center_x,
        params.center_y,
        params.background.r,
        params.background.g,
        params.background.b
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return output;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return output;
    }
    
    err = cudaMemcpy(output.data(), d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error (output): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return output;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return output;
}

int main() {
    std::string inputPath, outputPath;
    double angle;

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

    std::cout << "Enter rotation angle (degrees): ";
    std::cin >> angle;

    std::cout << "Loading image..." << std::endl;
    
    int width, height;
    std::vector<Pixel> image = loadImage(inputPath, width, height);
    
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }
    
    std::cout << "Image loaded: " << width << "x" << height << std::endl;
    std::cout << "Rotating image by " << angle << " degrees around center on GPU..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Pixel> rotated = rotateImageCUDA(image, width, height, angle);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Rotation completed in " << duration.count() << " ms" << std::endl;

    if (saveImage(outputPath, rotated, width, height)) {
        std::cout << "Image successfully saved to: " << outputPath << std::endl;
    } else {
        std::cerr << "Error saving image" << std::endl;
        return 1;
    }

    return 0;
}