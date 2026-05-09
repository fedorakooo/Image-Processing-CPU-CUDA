#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "../utils/image_utils.h"

struct CannyParams {
    double sigma;
    int kernelSize;
    int threshold;
};

std::vector<double> createGaussianKernel(double sigma, int size) {
    std::vector<double> kernel(size * size);
    double sum = 0.0;
    int half = size / 2;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            double value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + half) * size + (x + half)] = value;
            sum += value;
        }
    }

    for (double& v : kernel) {
        v /= sum;
    }

    return kernel;
}

void convertToGrayscale(
    const std::vector<Pixel>& input,
    std::vector<unsigned char>& output,
    int width, int height
) {
    for (int i = 0; i < width * height; i++) {
        output[i] = static_cast<unsigned char>(0.299 * input[i].r + 0.587 * input[i].g + 0.114 * input[i].b);
    }
}

void applyGaussianBlur(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int width, int height,
    const std::vector<double>& kernel,
    int kernelSize
) {
    int half = kernelSize / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum = 0.0;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int px = std::clamp(x + kx, 0, width - 1);
                    int py = std::clamp(y + ky, 0, height - 1);
                    double w = kernel[(ky + half) * kernelSize + (kx + half)];
                    sum += input[py * width + px] * w;
                }
            }

            output[y * width + x] = static_cast<unsigned char>(std::clamp(sum, 0.0, 255.0));
        }
    }
}

void processCannyRange(
    int startY, int endY,
    const std::vector<unsigned char>& image,
    std::vector<float>& magnitude,
    std::vector<float>& direction,
    std::vector<unsigned char>& suppressed,
    int width, int height
) {
    for (int y = std::max(startY, 1); y < std::min(endY, height - 1); y++) {
        for (int x = 1; x < width - 1; x++) {

            int gx = 0, gy = 0;
            gx = -image[(y - 1) * width + (x - 1)] - 2 * image[y * width + (x - 1)] - image[(y + 1) * width + (x - 1)] + image[(y - 1) * width + (x + 1)] + 2 * image[y * width + (x + 1)] + image[(y + 1) * width + (x + 1)];

            gy = -image[(y - 1) * width + (x - 1)] - 2 * image[(y - 1) * width + x] - image[(y - 1) * width + (x + 1)] + image[(y + 1) * width + (x - 1)] + 2 * image[(y + 1) * width + x] + image[(y + 1) * width + (x + 1)];

            float mag = std::sqrt(gx * gx + gy * gy);
            float ang = std::atan2(gy, gx);

            magnitude[y * width + x] = mag;
            direction[y * width + x] = ang;
        }
    }

    for (int y = std::max(startY, 1); y < std::min(endY, height - 1); y++) {
        for (int x = 1; x < width - 1; x++) {

            float angle = direction[y * width + x] * 180.0f / M_PI;
            if (angle < 0) angle += 180;

            float q = 0, r = 0;
            int idx = y * width + x;

            if ((angle < 22.5) || (angle >= 157.5)) {
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

            suppressed[idx] = (magnitude[idx] >= q && magnitude[idx] >= r) ? static_cast<unsigned char>(std::clamp(magnitude[idx], 0.0f, 255.0f)) : 0;
        }
    }
}

void singleThreshold(
    const std::vector<unsigned char>& input,
    std::vector<unsigned char>& output,
    int width, int height,
    int threshold
) {
    for (int i = 0; i < width * height; i++) {
        output[i] = (input[i] >= threshold) ? 255 : 0;
    }
}

std::vector<Pixel> cannyEdgeDetectionCPU(
    const std::vector<Pixel>& input,
    int width, int height,
    const CannyParams& params
) {
    std::vector<unsigned char> grayscale(width * height);
    convertToGrayscale(input, grayscale, width, height);

    auto kernel = createGaussianKernel(params.sigma, params.kernelSize);
    std::vector<unsigned char> blurred(width * height);
    applyGaussianBlur(grayscale, blurred, width, height, kernel, params.kernelSize);

    std::vector<float> magnitude(width * height, 0);
    std::vector<float> direction(width * height, 0);
    std::vector<unsigned char> suppressed(width * height, 0);

    unsigned threadsCount = std::thread::hardware_concurrency();
    if (!threadsCount) threadsCount = 4;

    std::vector<std::thread> threads;
    int rowsPerThread = (height + threadsCount - 1) / threadsCount;

    for (unsigned i = 0; i < threadsCount; i++) {
        int startY = i * rowsPerThread;
        int endY = std::min(height, startY + rowsPerThread);

        threads.emplace_back(
            processCannyRange,
            startY, endY,
            std::cref(blurred),
            std::ref(magnitude),
            std::ref(direction),
            std::ref(suppressed),
            width, height
        );
    }

    for (auto& t : threads) t.join();

    std::vector<unsigned char> edges(width * height);
    singleThreshold(suppressed, edges, width, height, params.threshold);

    std::vector<Pixel> result(width * height);
    for (int i = 0; i < width * height; i++) {
        result[i] = {edges[i], edges[i], edges[i]};
    }

    return result;
}

int main() {
    std::string inputPath, outputPath;
    CannyParams params;

    std::cout << "Input image path: ";
    std::getline(std::cin, inputPath);

    if (!std::filesystem::exists(inputPath)) {
        std::cerr << "File not found\n";
        return 1;
    }

    std::cout << "Output image path: ";
    std::getline(std::cin, outputPath);

    std::cout << "Gaussian sigma: ";
    std::cin >> params.sigma;

    std::cout << "Gaussian kernel size (odd): ";
    std::cin >> params.kernelSize;

    std::cout << "Threshold: ";
    std::cin >> params.threshold;

    if (params.kernelSize % 2 == 0) {
        std::cerr << "Kernel size must be odd\n";
        return 1;
    }

    int width, height;
    auto image = loadImage(inputPath, width, height);

    auto start = std::chrono::high_resolution_clock::now();
    auto edges = cannyEdgeDetectionCPU(image, width, height, params);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    saveImage(outputPath, edges, width, height);
    return 0;
}
