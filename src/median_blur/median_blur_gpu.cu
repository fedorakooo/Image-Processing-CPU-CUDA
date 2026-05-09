#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <chrono>
#include "../utils/image_utils.h"

__device__ Pixel getPixelClampedGPU(Pixel* img, int x, int y, int width, int height) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return img[y * width + x];
}

__global__ void medianBlurKernel(Pixel* input, Pixel* output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = kernelSize / 2;
    const int K = 49;

    unsigned char r[K], g[K], b[K];
    int idx = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            Pixel p = getPixelClampedGPU(input, x + dx, y + dy, width, height);
            r[idx] = p.r;
            g[idx] = p.g;
            b[idx] = p.b;
            idx++;
        }
    }

    auto medianGPU = [&](unsigned char* arr, int size) {
        for (int i = 0; i <= size / 2; i++) {
            int minIdx = i;
            for (int j = i + 1; j < size; j++)
                if (arr[j] < arr[minIdx]) minIdx = j;
            unsigned char tmp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = tmp;
        }
        return arr[size / 2];
    };

    Pixel out;
    out.r = medianGPU(r, idx);
    out.g = medianGPU(g, idx);
    out.b = medianGPU(b, idx);

    output[y * width + x] = out;
}

std::vector<Pixel> medianBlurCUDA(const std::vector<Pixel>& input, int width, int height, int kernelSize) {
    Pixel *d_in, *d_out;
    size_t size = input.size() * sizeof(Pixel);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, input.data(), size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    medianBlurKernel<<<blocks, threads>>>(d_in, d_out, width, height, kernelSize);
    cudaDeviceSynchronize();

    std::vector<Pixel> result(input.size());
    cudaMemcpy(result.data(), d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return result;
}

int main() {
    std::string inputPath, outputPath;
    int kernelSize;

    std::cout << "Enter input image path: ";
    std::getline(std::cin, inputPath);

    if (!std::filesystem::exists(inputPath)) {
        std::cerr << "File not found\n";
        return 1;
    }

    std::cout << "Enter output image path: ";
    std::getline(std::cin, outputPath);

    std::cout << "Enter median kernel size (odd): ";
    std::cin >> kernelSize;

    if (kernelSize % 2 == 0 || kernelSize < 1) {
        std::cerr << "Kernel size must be odd and > 0\n";
        return 1;
    }

    int width, height;
    auto image = loadImage(inputPath, width, height);

    std::cout << "Applying median blur with kernel " << kernelSize << " on GPU\n";

    auto start = std::chrono::high_resolution_clock::now();
    auto blurred = medianBlurCUDA(image, width, height, kernelSize);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    saveImage(outputPath, blurred, width, height);
    return 0;
}
