#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "../utils/image_utils.h"

struct CannyParams {
    double sigma;
    int kernelSize;
    int threshold;
};

__global__ void convertToGrayscaleKernel(
    const Pixel* input,
    unsigned char* output,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        Pixel p = input[idx];
        output[idx] = static_cast<unsigned char>(0.299f * p.r + 0.587f * p.g + 0.114f * p.b);
    }
}


__global__ void gaussianBlurKernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    const float* kernel,
    int kernelSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int half = kernelSize / 2;
        float sum = 0.0f;

        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                sum += input[py * width + px] * kernel[(ky + half) * kernelSize + (kx + half)];
            }
        }

        output[y * width + x] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
    }
}

__global__ void sobelKernel(
    const unsigned char* input,
    float* magnitude,
    float* direction,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

        int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)] + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

        int idx = y * width + x;
        magnitude[idx] = sqrtf(gx * gx + gy * gy);
        direction[idx] = atan2f(gy, gx);
    }
}

__global__ void nonMaximumSuppressionKernel(
    const float* magnitude,
    const float* direction,
    unsigned char* output,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int idx = y * width + x;
        float angle = direction[idx] * 180.0f / M_PI;
        if (angle < 0) angle += 180;

        float q = 0, r = 0;

        if (angle < 22.5 || angle >= 157.5) {
            q = magnitude[idx + 1];
            r = magnitude[idx - 1];
        } else if (angle < 67.5) {
            q = magnitude[(y - 1) * width + (x + 1)];
            r = magnitude[(y + 1) * width + (x - 1)];
        } else if (angle < 112.5) {
            q = magnitude[(y - 1) * width + x];
            r = magnitude[(y + 1) * width + x];
        } else {
            q = magnitude[(y - 1) * width + (x - 1)];
            r = magnitude[(y + 1) * width + (x + 1)];
        }

        output[idx] =
            (magnitude[idx] >= q && magnitude[idx] >= r) ? static_cast<unsigned char>(min(magnitude[idx], 255.0f)) : 0;
    }
}

__global__ void thresholdKernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    int threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = (input[idx] >= threshold) ? 255 : 0;
    }
}

std::vector<Pixel> cannyEdgeDetectionCUDA(
    const std::vector<Pixel>& input,
    int width, int height,
    const CannyParams& params
) {
    Pixel* d_input;
    unsigned char *d_gray, *d_blur, *d_nms, *d_edges;
    float *d_mag, *d_dir, *d_kernel;

    size_t imgBytes = width * height * sizeof(unsigned char);
    size_t pixBytes = width * height * sizeof(Pixel);
    size_t fltBytes = width * height * sizeof(float);

    cudaMalloc(&d_input, pixBytes);
    cudaMalloc(&d_gray, imgBytes);
    cudaMalloc(&d_blur, imgBytes);
    cudaMalloc(&d_nms, imgBytes);
    cudaMalloc(&d_edges, imgBytes);
    cudaMalloc(&d_mag, fltBytes);
    cudaMalloc(&d_dir, fltBytes);

    int kSize = params.kernelSize * params.kernelSize;
    std::vector<float> h_kernel(kSize);
    int half = params.kernelSize / 2;
    float sum = 0.0f;

    for (int y = -half; y <= half; y++)
        for (int x = -half; x <= half; x++) {
            float v = expf(-(x * x + y * y) / (2 * params.sigma * params.sigma));
            h_kernel[(y + half) * params.kernelSize + (x + half)] = v;
            sum += v;
        }

    for (float& v : h_kernel) v /= sum;

    cudaMalloc(&d_kernel, kSize * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel.data(), kSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input.data(), pixBytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    convertToGrayscaleKernel<<<grid, block>>>(d_input, d_gray, width, height);
    gaussianBlurKernel<<<grid, block>>>(d_gray, d_blur, width, height, d_kernel, params.kernelSize);
    sobelKernel<<<grid, block>>>(d_blur, d_mag, d_dir, width, height);
    nonMaximumSuppressionKernel<<<grid, block>>>(d_mag, d_dir, d_nms, width, height);
    thresholdKernel<<<grid, block>>>(d_nms, d_edges, width, height, params.threshold);

    std::vector<unsigned char> h_edges(width * height);
    cudaMemcpy(h_edges.data(), d_edges, imgBytes, cudaMemcpyDeviceToHost);

    std::vector<Pixel> result(width * height);
    for (int i = 0; i < width * height; i++)
        result[i] = {h_edges[i], h_edges[i], h_edges[i]};

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_nms);
    cudaFree(d_edges);
    cudaFree(d_mag);
    cudaFree(d_dir);
    cudaFree(d_kernel);

    return result;
}

int main() {
    std::string inputPath, outputPath;
    CannyParams params;

    std::cout << "Input image path: ";
    std::getline(std::cin, inputPath);

    std::cout << "Output image path: ";
    std::getline(std::cin, outputPath);

    std::cout << "Gaussian sigma: ";
    std::cin >> params.sigma;

    std::cout << "Gaussian kernel size (odd): ";
    std::cin >> params.kernelSize;

    std::cout << "Threshold: ";
    std::cin >> params.threshold;

    int width, height;
    auto image = loadImage(inputPath, width, height);

    auto start = std::chrono::high_resolution_clock::now();
    auto edges = cannyEdgeDetectionCUDA(image, width, height, params);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "CUDA Canny time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    saveImage(outputPath, edges, width, height);
    return 0;
}
