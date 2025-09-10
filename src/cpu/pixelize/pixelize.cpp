#include <iostream>
#include <filesystem>
#include <vector>

#include "../../utils/image_utils.h"

void processPixelBlock(int startX, int startY, int blockSize, int width, int height, const std::vector<Pixel>& input, std::vector<Pixel>& result) {
    int endX = std::min(startX + blockSize, width);
    int endY = std::min(startY + blockSize, height);
    int actualBlockSize = (endX - startX) * (endY - startY);
    
    if (actualBlockSize == 0) return;
    
    unsigned long long sumR = 0, sumG = 0, sumB = 0;
    
    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            int index = y * width + x;
            sumR += input[index].r;
            sumG += input[index].g;
            sumB += input[index].b;
        }
    }
    
    Pixel averageColor = {
        static_cast<unsigned char>(sumR / actualBlockSize),
        static_cast<unsigned char>(sumG / actualBlockSize),
        static_cast<unsigned char>(sumB / actualBlockSize)
    };
    
    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            int index = y * width + x;
            result[index] = averageColor;
        }
    }
}

std::vector<Pixel> pixelizeImage(const std::vector<Pixel>& input, int width, int height, int blockSize) {
    if (blockSize <= 1) return input;
    
    std::vector<Pixel> result(input.size());
    
    for (int y = 0; y < height; y += blockSize) {
        for (int x = 0; x < width; x += blockSize) {
            processPixelBlock(x, y, blockSize, width, height, input, result);
        }
    }
    
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

    // Create output directories if it doesn't exist
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
    std::cout << "Pixelizing image with block size " << blockSize << "x" << blockSize << "..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto pixelized = pixelizeImage(image, width, height, blockSize);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Pixelization completed in " << duration.count() << " ms" << std::endl;

    if (saveImage(outputPath, pixelized, width, height)) {
        std::cout << "Image successfully saved!" << std::endl;
    } else {
        std::cerr << "Error saving image" << std::endl;
        return 1;
    }

    return 0;
}