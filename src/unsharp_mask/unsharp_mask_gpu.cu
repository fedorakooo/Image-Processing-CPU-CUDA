#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <cuda_runtime.h>
#include "../utils/image_utils.h"

__device__ Pixel getClampedGPU(Pixel* img, int x, int y, int w, int h) {
    x = max(0, min(x, w - 1));
    y = max(0, min(y, h - 1));
    return img[y * w + x];
}

__global__ void gaussianBlurKernel(
    Pixel* input, Pixel* output,
    int width, int height,
    const float* kernel, int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    int kSize = radius * 2 + 1;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            Pixel p = getClampedGPU(input, x + dx, y + dy, width, height);
            float k = kernel[(dy + radius) * kSize + (dx + radius)];
            r += p.r * k;
            g += p.g * k;
            b += p.b * k;
        }
    }

    output[y * width + x] = {
        (unsigned char)fminf(fmaxf(r + 0.5f, 0.0f), 255.0f),
        (unsigned char)fminf(fmaxf(g + 0.5f, 0.0f), 255.0f),
        (unsigned char)fminf(fmaxf(b + 0.5f, 0.0f), 255.0f)
    };
}

__global__ void maskKernel(
    Pixel* original, Pixel* blurred,
    Pixel* mask,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    mask[idx].r = (signed char)(original[idx].r - blurred[idx].r);
    mask[idx].g = (signed char)(original[idx].g - blurred[idx].g);
    mask[idx].b = (signed char)(original[idx].b - blurred[idx].b);
}

__global__ void applyMaskKernel(
    Pixel* original, Pixel* mask,
    Pixel* output,
    float amount,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    float r = original[idx].r + (mask[idx].r * amount);
    float g = original[idx].g + (mask[idx].g * amount);
    float b = original[idx].b + (mask[idx].b * amount);

    output[idx] = {
        (unsigned char)fminf(fmaxf(r + 0.5f, 0.0f), 255.0f),
        (unsigned char)fminf(fmaxf(g + 0.5f, 0.0f), 255.0f),
        (unsigned char)fminf(fmaxf(b + 0.5f, 0.0f), 255.0f)
    };
}

std::vector<Pixel> unsharpMaskCUDA(
    const std::vector<Pixel>& input,
    int width, int height,
    int radius,
    float amount
) {
    if (input.empty() || width <= 0 || height <= 0) {
        std::cerr << "Invalid input parameters" << std::endl;
        return input;
    }

    size_t size = input.size();
    size_t sizeBytes = size * sizeof(Pixel);
    std::vector<Pixel> result(size);

    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices available!" << std::endl;
        return input;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    std::cout << "Allocating GPU memory..." << std::endl;

    Pixel *d_in = nullptr, *d_blur = nullptr, *d_mask = nullptr, *d_out = nullptr;
    float* d_kernel = nullptr;

    err = cudaMalloc(&d_in, sizeBytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input: " << cudaGetErrorString(err) << std::endl;
        return result;
    }

    err = cudaMalloc(&d_blur, sizeBytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for blur: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        return result;
    }

    err = cudaMalloc(&d_mask, sizeBytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for mask: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        return result;
    }

    err = cudaMalloc(&d_out, sizeBytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        return result;
    }

    cudaMemset(d_out, 0, sizeBytes);

    std::cout << "Copying input to GPU..." << std::endl;

    err = cudaMemcpy(d_in, input.data(), sizeBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy input to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        return result;
    }

    std::cout << "Creating Gaussian kernel..." << std::endl;
    int kSize = radius * 2 + 1;
    std::vector<float> kernelCPU(kSize * kSize);
    
    float sigma = radius / 2.0f;
    float twoSigma2 = 2.0f * sigma * sigma;
    float sum = 0.0f;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float val = expf(-(x*x + y*y) / twoSigma2);
            kernelCPU[(y + radius) * kSize + (x + radius)] = val;
            sum += val;
        }
    }

    for (int i = 0; i < kSize * kSize; i++) {
        kernelCPU[i] /= sum;
    }

    err = cudaMalloc(&d_kernel, kernelCPU.size() * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for kernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        return result;
    }

    err = cudaMemcpy(d_kernel, kernelCPU.data(), kernelCPU.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy kernel to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        cudaFree(d_kernel);
        return result;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "Launching kernels with grid: " << numBlocks.x << "x" << numBlocks.y 
              << " blocks, " << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads" << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;

    std::cout << "Running Gaussian blur..." << std::endl;
    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(
        d_in, d_blur, width, height, d_kernel, radius
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Gaussian blur kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        cudaFree(d_kernel);
        return result;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Sync after blur failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        cudaFree(d_kernel);
        return result;
    }

    std::cout << "Creating mask..." << std::endl;
    maskKernel<<<numBlocks, threadsPerBlock>>>(
        d_in, d_blur, d_mask, width, height
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Mask kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        cudaFree(d_kernel);
        return result;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Sync after mask failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        cudaFree(d_kernel);
        return result;
    }

    std::cout << "Applying mask..." << std::endl;
    applyMaskKernel<<<numBlocks, threadsPerBlock>>>(
        d_in, d_mask, d_out, amount, width, height
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Apply mask kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        cudaFree(d_kernel);
        return result;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Sync after apply mask failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        cudaFree(d_kernel);
        return result;
    }

    std::cout << "Copying result back to CPU..." << std::endl;

    err = cudaMemcpy(result.data(), d_out, sizeBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy result from device: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_blur);
    cudaFree(d_mask);
    cudaFree(d_out);
    cudaFree(d_kernel);

    bool allZero = true;
    for (int i = 0; i < std::min(10, (int)result.size()); i++) {
        if (result[i].r != 0 || result[i].g != 0 || result[i].b != 0) {
            allZero = false;
            break;
        }
    }
    
    if (allZero) {
        std::cerr << "WARNING: Result appears to be all zeros!" << std::endl;
        return input;
    }

    std::cout << "GPU processing completed successfully" << std::endl;

    return result;
}

int main() {
    std::string inPath, outPath;
    int radius;
    float amount;

    std::cout << "Enter input image path: ";
    std::getline(std::cin, inPath);

    if (!std::filesystem::exists(inPath)) {
        std::cerr << "Input file not found: " << inPath << std::endl;
        return 1;
    }

    std::cout << "Enter output image path: ";
    std::getline(std::cin, outPath);

    std::filesystem::path outputDir = std::filesystem::path(outPath).parent_path();
    if (!outputDir.empty() && !std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }

    std::cout << "Enter blur radius (1-10): ";
    std::cin >> radius;

    std::cout << "Enter amount (0.5-3.0): ";
    std::cin >> amount;

    if (radius < 1 || radius > 10) {
        std::cerr << "Radius should be between 1 and 10" << std::endl;
        return 1;
    }

    if (amount < 0.5f || amount > 3.0f) {
        std::cerr << "Amount should be between 0.5 and 3.0" << std::endl;
        return 1;
    }

    int w, h;
    std::cout << "Loading image..." << std::endl;
    std::vector<Pixel> img = loadImage(inPath, w, h);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << w << "x" << h << " (" << img.size() << " pixels)" << std::endl;
    std::cout << "Applying unsharp mask on GPU..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Pixel> result = unsharpMaskCUDA(img, w, h, radius, amount);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total processing time: " << duration.count() << " ms" << std::endl;

    if (saveImage(outPath, result, w, h)) {
        std::cout << "Image successfully saved to: " << outPath << std::endl;
    } else {
        std::cerr << "Error saving image!" << std::endl;
        return 1;
    }

    return 0;
}