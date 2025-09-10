#include "image_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

std::vector<Pixel> loadImage(const std::string& filename, int& width, int& height) {
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, nullptr, 3);
    if (!data) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return {};
    }
    
    std::vector<Pixel> pixels(width * height);
    
    for (int i = 0; i < width * height; ++i) {
        int baseIndex = i * 3;
        pixels[i].r = data[baseIndex];
        pixels[i].g = data[baseIndex + 1];
        pixels[i].b = data[baseIndex + 2];
    }
    
    stbi_image_free(data);
    return pixels;
}

bool saveImage(const std::string& filename, const std::vector<Pixel>& pixels, int width, int height) {
    std::vector<unsigned char> outputData;
    outputData.reserve(width * height * 3);
    
    for (const auto& pixel : pixels) {
        outputData.push_back(pixel.r);
        outputData.push_back(pixel.g);
        outputData.push_back(pixel.b);
    }
    
    std::string ext = std::filesystem::path(filename).extension().string();
    int success = 0;
    
    if (ext == ".png") {
        success = stbi_write_png(filename.c_str(), width, height, 3, outputData.data(), width * 3);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        success = stbi_write_jpg(filename.c_str(), width, height, 3, outputData.data(), 90);
    } else if (ext == ".bmp") {
        success = stbi_write_bmp(filename.c_str(), width, height, 3, outputData.data());
    } else {
        std::cerr << "Unsupported format: " << ext << std::endl;
        return false;
    }
    
    return success != 0;
}