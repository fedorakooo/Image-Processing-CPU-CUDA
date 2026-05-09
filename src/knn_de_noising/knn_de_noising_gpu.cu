#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <algorithm>
#include "../utils/image_utils.h"

struct KNNParams {
    int patchSize;
    int searchWindowSize;
    int kNeighbors;
};

__device__ float calculatePatchDistanceCUDA(
    const Pixel* image,
    int width, int height,
    int x1, int y1, int x2, int y2,
    int patchRadius
) {
    float distance = 0.0f;
    int pixelsCount = 0;
    
    for (int dy = -patchRadius; dy <= patchRadius; dy++) {
        for (int dx = -patchRadius; dx <= patchRadius; dx++) {
            int px1 = max(0, min(x1 + dx, width - 1));
            int py1 = max(0, min(y1 + dy, height - 1));
            int px2 = max(0, min(x2 + dx, width - 1));
            int py2 = max(0, min(y2 + dy, height - 1));
            
            int idx1 = py1 * width + px1;
            int idx2 = py2 * width + px2;
            
            float diffR = static_cast<float>(image[idx1].r) - static_cast<float>(image[idx2].r);
            float diffG = static_cast<float>(image[idx1].g) - static_cast<float>(image[idx2].g);
            float diffB = static_cast<float>(image[idx1].b) - static_cast<float>(image[idx2].b);
            
            distance += diffR * diffR + diffG * diffG + diffB * diffB;
            pixelsCount++;
        }
    }
    
    return distance / pixelsCount;
}

__global__ void knnDenoiseKernel(
    Pixel* input,
    Pixel* output,
    int width,
    int height,
    int patchSize,
    int searchWindowSize,
    int kNeighbors
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int patchRadius = patchSize / 2;
    int searchRadius = searchWindowSize / 2;
    
    if (x < patchRadius || x >= width - patchRadius || y < patchRadius || y >= height - patchRadius) {
        output[y * width + x] = input[y * width + x];
        return;
    }
    
    const int maxNeighbors = 100;
    float distances[maxNeighbors];
    Pixel neighbors[maxNeighbors];
    int neighborCount = 0;
    
    int startSearchY = max(0, y - searchRadius);
    int endSearchY = min(height - 1, y + searchRadius);
    int startSearchX = max(0, x - searchRadius);
    int endSearchX = min(width - 1, x + searchRadius);
    
    for (int searchY = startSearchY; searchY <= endSearchY && neighborCount < maxNeighbors; searchY++) {
        for (int searchX = startSearchX; searchX <= endSearchX && neighborCount < maxNeighbors; searchX++) {
            if (searchX == x && searchY == y) continue;
            
            if (searchX < patchRadius || searchX >= width - patchRadius ||
                searchY < patchRadius || searchY >= height - patchRadius) {
                continue;
            }
            
            float distance = calculatePatchDistanceCUDA(
                input, width, height,
                x, y, searchX, searchY,
                patchRadius
            );
            
            distances[neighborCount] = distance;
            neighbors[neighborCount] = input[searchY * width + searchX];
            neighborCount++;
        }
    }
    
    if (neighborCount < kNeighbors) {
        output[y * width + x] = input[y * width + x];
        return;
    }
    
    for (int i = 0; i < kNeighbors; i++) {
        for (int j = i + 1; j < neighborCount; j++) {
            if (distances[j] < distances[i]) {
                float tempDist = distances[i];
                distances[i] = distances[j];
                distances[j] = tempDist;
                
                Pixel tempPixel = neighbors[i];
                neighbors[i] = neighbors[j];
                neighbors[j] = tempPixel;
            }
        }
    }
    
    unsigned int sumR = input[y * width + x].r;
    unsigned int sumG = input[y * width + x].g;
    unsigned int sumB = input[y * width + x].b;
    
    for (int i = 0; i < kNeighbors && i < neighborCount; i++) {
        sumR += neighbors[i].r;
        sumG += neighbors[i].g;
        sumB += neighbors[i].b;
    }
    
    int totalPixels = kNeighbors + 1;
    
    Pixel result;
    result.r = static_cast<unsigned char>(sumR / totalPixels);
    result.g = static_cast<unsigned char>(sumG / totalPixels);
    result.b = static_cast<unsigned char>(sumB / totalPixels);
    
    output[y * width + x] = result;
}

std::vector<Pixel> knnDenoiseImageCUDA(
    const std::vector<Pixel>& input,
    int width, int height,
    const KNNParams& params
) {
    Pixel *d_input, *d_output;
    size_t size = input.size() * sizeof(Pixel);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    knnDenoiseKernel<<<numBlocks, threadsPerBlock>>>(
        d_input,
        d_output,
        width,
        height,
        params.patchSize,
        params.searchWindowSize,
        params.kNeighbors
    );
    
    cudaDeviceSynchronize();
    
    std::vector<Pixel> result(input.size());
    cudaMemcpy(result.data(), d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return result;
}

int main() {
    std::string inputPath, outputPath;
    KNNParams params;

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

    std::cout << "Enter patch size (odd number, e.g., 3, 5, 7): ";
    std::cin >> params.patchSize;

    std::cout << "Enter search window size (odd number, e.g., 11, 21, 31): ";
    std::cin >> params.searchWindowSize;

    std::cout << "Enter number of nearest neighbors (K): ";
    std::cin >> params.kNeighbors;

    if (params.patchSize % 2 == 0 || params.searchWindowSize % 2 == 0) {
        std::cerr << "Patch size and search window size must be odd numbers" << std::endl;
        return 1;
    }

    if (params.kNeighbors < 1) {
        std::cerr << "Number of neighbors must be positive" << std::endl;
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
    std::cout << "Applying KNN denoising on GPU..." << std::endl;
    std::cout << "Patch size: " << params.patchSize << "x" << params.patchSize << std::endl;
    std::cout << "Search window: " << params.searchWindowSize << "x" << params.searchWindowSize << std::endl;
    std::cout << "K neighbors: " << params.kNeighbors << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto denoised = knnDenoiseImageCUDA(image, width, height, params);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "KNN denoising completed in " << duration.count() << " ms" << std::endl;

    if (saveImage(outputPath, denoised, width, height)) {
        std::cout << "Image successfully saved!" << std::endl;
    } else {
        std::cerr << "Error saving image" << std::endl;
        return 1;
    }

    return 0;
}