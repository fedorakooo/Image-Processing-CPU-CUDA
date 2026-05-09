#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cmath>

#include "../utils/image_utils.h"

struct KNNParams {
    int patchSize;
    int searchWindowSize;
    int kNeighbors;
};

float calculatePatchDistance(
    const std::vector<Pixel>& image,
    int width, int height,
    int x1, int y1, int x2, int y2,
    int patchRadius
) {
    float distance = 0.0f;
    int pixelsCount = 0;
    
    for (int dy = -patchRadius; dy <= patchRadius; dy++) {
        for (int dx = -patchRadius; dx <= patchRadius; dx++) {
            int px1 = std::clamp(x1 + dx, 0, width - 1);
            int py1 = std::clamp(y1 + dy, 0, height - 1);
            int px2 = std::clamp(x2 + dx, 0, width - 1);
            int py2 = std::clamp(y2 + dy, 0, height - 1);
            
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

void processPixelKNN(
    int x, int y,
    int width, int height,
    const std::vector<Pixel>& input,
    std::vector<Pixel>& output,
    const KNNParams& params
) {
    int patchRadius = params.patchSize / 2;
    int searchRadius = params.searchWindowSize / 2;
    
    if (x < patchRadius || x >= width - patchRadius ||
        y < patchRadius || y >= height - patchRadius) {
        output[y * width + x] = input[y * width + x];
        return;
    }
    
    std::vector<std::pair<float, Pixel>> neighbors;
    neighbors.reserve(params.searchWindowSize * params.searchWindowSize);
    
    int startSearchY = std::max(0, y - searchRadius);
    int endSearchY = std::min(height - 1, y + searchRadius);
    int startSearchX = std::max(0, x - searchRadius);
    int endSearchX = std::min(width - 1, x + searchRadius);
    
    for (int searchY = startSearchY; searchY <= endSearchY; searchY++) {
        for (int searchX = startSearchX; searchX <= endSearchX; searchX++) {
            if (searchX == x && searchY == y) continue;
            
            if (searchX < patchRadius || searchX >= width - patchRadius ||
                searchY < patchRadius || searchY >= height - patchRadius) {
                continue;
            }
            
            float distance = calculatePatchDistance(
                input, width, height,
                x, y, searchX, searchY,
                patchRadius
            );
            
            Pixel neighborPixel = input[searchY * width + searchX];
            neighbors.emplace_back(distance, neighborPixel);
        }
    }
    
    if (neighbors.size() < params.kNeighbors) {
        output[y * width + x] = input[y * width + x];
        return;
    }
    
    std::partial_sort(
        neighbors.begin(),
        neighbors.begin() + params.kNeighbors,
        neighbors.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; }
    );
    
    unsigned long long sumR = 0, sumG = 0, sumB = 0;
    
    sumR += input[y * width + x].r;
    sumG += input[y * width + x].g;
    sumB += input[y * width + x].b;
    
    for (int i = 0; i < params.kNeighbors && i < neighbors.size(); i++) {
        sumR += neighbors[i].second.r;
        sumG += neighbors[i].second.g;
        sumB += neighbors[i].second.b;
    }
    
    int totalPixels = params.kNeighbors + 1;
    
    Pixel result;
    result.r = static_cast<unsigned char>(sumR / totalPixels);
    result.g = static_cast<unsigned char>(sumG / totalPixels);
    result.b = static_cast<unsigned char>(sumB / totalPixels);
    
    output[y * width + x] = result;
}

void processImageRangeKNN(
    int startY, int endY,
    int width, int height,
    const std::vector<Pixel>& input,
    std::vector<Pixel>& output,
    const KNNParams& params
) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            processPixelKNN(x, y, width, height, input, output, params);
        }
    }
}

std::vector<Pixel> knnDenoiseImageParallel(
    const std::vector<Pixel>& input,
    int width, int height,
    const KNNParams& params
) {
    std::vector<Pixel> output(input.size());
    std::vector<std::thread> threads;
    
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    int rowsPerThread = (height + numThreads - 1) / numThreads;
    
    for (int i = 0; i < numThreads; i++) {
        int startY = i * rowsPerThread;
        int endY = std::min((i + 1) * rowsPerThread, height);
        
        if (startY >= height) break;
        
        threads.emplace_back(
            processImageRangeKNN,
            startY, endY,
            width, height,
            std::cref(input),
            std::ref(output),
            std::cref(params)
        );
    }
    
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    return output;
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
    std::cout << "Applying KNN denoising..." << std::endl;
    std::cout << "Patch size: " << params.patchSize << "x" << params.patchSize << std::endl;
    std::cout << "Search window: " << params.searchWindowSize << "x" << params.searchWindowSize << std::endl;
    std::cout << "K neighbors: " << params.kNeighbors << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto denoised = knnDenoiseImageParallel(image, width, height, params);
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