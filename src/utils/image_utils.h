#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>
#include <filesystem>

struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

std::vector<Pixel> loadImage(const std::string& filename, int& width, int& height);
bool saveImage(const std::string& filename, const std::vector<Pixel>& pixels, int width, int height);

#endif // IMAGE_UTILS_H