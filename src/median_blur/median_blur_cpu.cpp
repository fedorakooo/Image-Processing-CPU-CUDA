#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include "../utils/image_utils.h"

inline Pixel getPixelClamped(const std::vector<Pixel>& img, int x, int y, int width, int height) {
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    return img[y * width + x];
}

Pixel computeMedian(const std::vector<Pixel>& img, int cx, int cy, int width, int height, int kernelSize) {
    int radius = kernelSize / 2;
    std::vector<unsigned char> r, g, b;
    r.reserve(kernelSize * kernelSize);
    g.reserve(kernelSize * kernelSize);
    b.reserve(kernelSize * kernelSize);

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            Pixel p = getPixelClamped(img, cx + dx, cy + dy, width, height);
            r.push_back(p.r);
            g.push_back(p.g);
            b.push_back(p.b);
        }
    }

    auto median = [&](std::vector<unsigned char>& v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };

    return {
        median(r),
        median(g),
        median(b)
    };
}

void processMedianRange(
    int startY,
    int endY,
    int kernelSize,
    int width,
    int height,
    const std::vector<Pixel>& input,
    std::vector<Pixel>& result
) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            result[y * width + x] = computeMedian(input, x, y, width, height, kernelSize);
        }
    }
}

std::vector<Pixel> medianBlurParallel(const std::vector<Pixel>& input, int width, int height, int kernelSize) {
    std::vector<Pixel> result(input.size());
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<std::thread> threads;

    int rowsPerThread = (height + numThreads - 1) / numThreads;

    for (int i = 0; i < numThreads; i++) {
        int startY = i * rowsPerThread;
        int endY   = std::min((i + 1) * rowsPerThread, height);
        if (startY >= height) break;

        threads.emplace_back(
            processMedianRange,
            startY, endY,
            kernelSize,
            width, height,
            std::cref(input),
            std::ref(result)
        );
    }

    for (auto& t : threads) t.join();
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

    std::cout << "Applying median blur with kernel " << kernelSize << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    auto blurred = medianBlurParallel(image, width, height, kernelSize);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    saveImage(outputPath, blurred, width, height);
}
