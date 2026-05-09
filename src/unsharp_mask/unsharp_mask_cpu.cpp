#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "../utils/image_utils.h"

std::vector<float> generateGaussianKernel(int radius) {
    int size = radius * 2 + 1;
    std::vector<float> kernel(size * size);

    float sigma = radius / 2.0f;
    float twoSigma2 = 2.0f * sigma * sigma;
    float sum = 0.0f;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float val = std::exp(-(x*x + y*y) / twoSigma2);
            kernel[(y + radius) * size + (x + radius)] = val;
            sum += val;
        }
    }

    for (float& v : kernel) v /= sum;

    return kernel;
}

inline Pixel getPixelClamped(
    const std::vector<Pixel>& img, int x, int y, int width, int height
) {
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    return img[y * width + x];
}

Pixel applyGaussian(
    const std::vector<Pixel>& img,
    int x,
    int y,
    int width,
    int height,
    const std::vector<float>& kernel,
    int radius
) {
    int size = radius * 2 + 1;
    float r = 0, g = 0, b = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            Pixel p = getPixelClamped(img, x + dx, y + dy, width, height);
            float k = kernel[(dy + radius) * size + (dx + radius)];
            r += p.r * k;
            g += p.g * k;
            b += p.b * k;
        }
    }
    return Pixel{
        static_cast<unsigned char>(r),
        static_cast<unsigned char>(g),
        static_cast<unsigned char>(b)
    };
}

void gaussianRange(
    int startY,
    int endY,
    int width,
    int height,
    const std::vector<Pixel>& input,
    std::vector<Pixel>& blurred,
    const std::vector<float>& kernel,
    int radius
) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            blurred[y * width + x] = applyGaussian(input, x, y, width, height, kernel, radius);
        }
    }
}

void maskRange(
    int startY,
    int endY,
    int width,
    const std::vector<Pixel>& original,
    const std::vector<Pixel>& blurred,
    std::vector<Pixel>& mask
) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;

            mask[i].r = original[i].r - blurred[i].r;
            mask[i].g = original[i].g - blurred[i].g;
            mask[i].b = original[i].b - blurred[i].b;
        }
    }
}

void applyMaskRange(
    int startY,
    int endY,
    int width,
    const std::vector<Pixel>& original,
    const std::vector<Pixel>& mask,
    std::vector<Pixel>& result,
    float amount
) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;

            int r = original[i].r + mask[i].r * amount;
            int g = original[i].g + mask[i].g * amount;
            int b = original[i].b + mask[i].b * amount;

            result[i].r = std::clamp(r, 0, 255);
            result[i].g = std::clamp(g, 0, 255);
            result[i].b = std::clamp(b, 0, 255);
        }
    }
}

std::vector<Pixel> unsharpMaskCPU(
    const std::vector<Pixel>& input,
    int width,
    int height,
    int radius,
    float amount
) {
    std::vector<Pixel> blurred(input.size());
    std::vector<Pixel> mask(input.size());
    std::vector<Pixel> result(input.size());

    auto kernel = generateGaussianKernel(radius);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    int rowsPerThread = (height + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < numThreads; i++) {
        int s = i * rowsPerThread;
        int e = std::min((int)((i + 1) * rowsPerThread), height);

        threads.emplace_back(
            gaussianRange,
            s, e, width, height,
            std::cref(input), std::ref(blurred),
            std::cref(kernel), radius
        );
    }
    for (auto& t : threads) t.join();
    threads.clear();

    for (unsigned int i = 0; i < numThreads; i++) {
        int s = i * rowsPerThread;
        int e = std::min((int)((i + 1) * rowsPerThread), height);

        threads.emplace_back(
            maskRange,
            s, e, width,
            std::cref(input), std::cref(blurred),
            std::ref(mask)
        );
    }
    for (auto& t : threads) t.join();
    threads.clear();

    for (unsigned int i = 0; i < numThreads; i++) {
        int s = i * rowsPerThread;
        int e = std::min((int)((i + 1) * rowsPerThread), height);

        threads.emplace_back(
            applyMaskRange,
            s, e, width,
            std::cref(input), std::cref(mask),
            std::ref(result),
            amount
        );
    }
    for (auto& t : threads) t.join();

    return result;
}

int main() {
    std::string inPath, outPath;
    int radius;
    float amount;

    std::cout << "Enter input image path: ";
    std::getline(std::cin, inPath);

    std::cout << "Enter output image path: ";
    std::getline(std::cin, outPath);

    std::cout << "Enter blur radius: ";
    std::cin >> radius;

    std::cout << "Enter amount (e.g. 1.5): ";
    std::cin >> amount;

    int w, h;
    auto img = loadImage(inPath, w, h);
    if (img.empty()) {
        std::cerr << "Error loading image!\n";
        return 1;
    }

    auto result = unsharpMaskCPU(img, w, h, radius, amount);
    saveImage(outPath, result, w, h);

    std::cout << "Done.\n";
}
