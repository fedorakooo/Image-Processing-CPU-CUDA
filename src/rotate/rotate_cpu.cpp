#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>

#include "../utils/image_utils.h"

struct RotationParams {
    double angle_rad;
    double center_x;
    double center_y;
    Pixel background;
};

void rotatePixelsRange(
    int startY,
    int endY,
    int width,
    int height,
    const std::vector<Pixel>& input,
    std::vector<Pixel>& output,
    const RotationParams& params
) {
    double sin_angle = sin(params.angle_rad);
    double cos_angle = cos(params.angle_rad);
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            double x_rel = x - params.center_x;
            double y_rel = y - params.center_y;
            
            double src_x = x_rel * cos_angle + y_rel * sin_angle + params.center_x;
            double src_y = -x_rel * sin_angle + y_rel * cos_angle + params.center_y;
            
            int x0 = static_cast<int>(floor(src_x));
            int y0 = static_cast<int>(floor(src_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            double dx = src_x - x0;
            double dy = src_y - y0;
            
            auto getPixelSafe = [&](int px, int py) -> Pixel {
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    return input[py * width + px];
                }
                return params.background;
            };
            
            const Pixel& p00 = getPixelSafe(x0, y0);
            const Pixel& p10 = getPixelSafe(x1, y0);
            const Pixel& p01 = getPixelSafe(x0, y1);
            const Pixel& p11 = getPixelSafe(x1, y1);
            
            Pixel result;
            result.r = static_cast<unsigned char>(p00.r * (1 - dx) * (1 - dy) + p10.r * dx * (1 - dy) + p01.r * (1 - dx) * dy + p11.r * dx * dy);
            result.g = static_cast<unsigned char>(p00.g * (1 - dx) * (1 - dy) + p10.g * dx * (1 - dy) + p01.g * (1 - dx) * dy + p11.g * dx * dy);
            result.b = static_cast<unsigned char>(p00.b * (1 - dx) * (1 - dy) + p10.b * dx * (1 - dy) + p01.b * (1 - dx) * dy + p11.b * dx * dy);
            
            output[y * width + x] = result;
        }
    }
}

std::vector<Pixel> rotateImageParallel(
    const std::vector<Pixel>& input, 
    int width, 
    int height, 
    double angle_degrees,
    Pixel background = {0, 0, 0}
) {
    std::vector<Pixel> output(input.size());
    
    double angle_rad = angle_degrees * M_PI / 180.0;
    double center_x = (width - 1) / 2.0;
    double center_y = (height - 1) / 2.0;
    
    RotationParams params{angle_rad, center_x, center_y, background};
    
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    std::vector<std::thread> threads;
    int rowsPerThread = (height + numThreads - 1) / numThreads;
    
    for (int i = 0; i < numThreads; ++i) {
        int startY = i * rowsPerThread;
        int endY = std::min((i + 1) * rowsPerThread, height);
        
        if (startY >= height) break;
        
        threads.emplace_back(
            rotatePixelsRange, 
            startY, 
            endY, 
            width, 
            height, 
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
    double angle;

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

    std::cout << "Enter rotation angle (degrees): ";
    std::cin >> angle;

    std::cout << "Loading image..." << std::endl;
    
    int width, height;
    auto image = loadImage(inputPath, width, height);
    
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }
    
    std::cout << "Image loaded: " << width << "x" << height << std::endl;
    std::cout << "Rotating image by " << angle << " degrees around center..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto rotated = rotateImageParallel(image, width, height, angle);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Rotation completed in " << duration.count() << " ms" << std::endl;

    if (saveImage(outputPath, rotated, width, height)) {
        std::cout << "Image successfully saved!" << std::endl;
    } else {
        std::cerr << "Error saving image" << std::endl;
        return 1;
    }

    return 0;
}