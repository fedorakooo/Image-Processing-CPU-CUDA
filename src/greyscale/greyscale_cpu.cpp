#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <algorithm>

#include "../utils/image_utils.h"

void processGreyscaleRange(
    int startY,
    int endY,
    int width,
    const std::vector<Pixel>& input,
    std::vector<Pixel>& result
) {
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            unsigned char grey = static_cast<unsigned char>(
                0.2126 * input[index].r +
                0.7152 * input[index].g +
                0.0722 * input[index].b
            );
            result[index] = {grey, grey, grey};
        }
    }
}

std::vector<Pixel> greyscaleImageParallel(
    const std::vector<Pixel>& input,
    int width,
    int height
) {
    std::vector<Pixel> result(input.size());
    std::vector<std::thread> threads;

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    int rowsPerThread = (height + numThreads - 1) / numThreads;

    for (unsigned int i = 0; i < numThreads; ++i) {
        int startY = static_cast<int>(i * rowsPerThread);
        int endY = std::min(startY + rowsPerThread, height);
        if (startY >= height) break;

        threads.emplace_back(
            processGreyscaleRange,
            startY,
            endY,
            width,
            std::cref(input),
            std::ref(result)
        );
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
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

    std::cout << "Loading image: " << inputPath << std::endl;

    int width, height;
    auto image = loadImage(inputPath, width, height);

    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto greyImage = greyscaleImageParallel(image, width, height);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Greyscale conversion completed in " << duration.count() << " ns" << std::endl;

    if (saveImage(outputPath, greyImage, width, height)) {
        std::cout << "Image successfully saved!" << std::endl;
    } else {
        std::cerr << "Error saving image" << std::endl;
        return 1;
    }

    return 0;
}
