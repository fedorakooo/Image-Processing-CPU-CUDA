BUILD_DIR = build

CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17

INCLUDES = -Ilib -Isrc

UTILS_SRC = src/utils/image_utils.cpp

# CPU executables
CPU_TARGETS = $(BUILD_DIR)/greyscale_cpu \
              $(BUILD_DIR)/pixelize_cpu \
              $(BUILD_DIR)/median_blur_cpu \
              $(BUILD_DIR)/rotate_cpu \
              $(BUILD_DIR)/unsharp_mask_cpu \
              $(BUILD_DIR)/canny_edge_detector_cpu \
              $(BUILD_DIR)/knn_de_noising_cpu

# GPU executables
GPU_TARGETS = $(BUILD_DIR)/greyscale_cuda \
              $(BUILD_DIR)/pixelize_cuda \
              $(BUILD_DIR)/median_blur_cuda \
              $(BUILD_DIR)/rotate_cuda \
              $(BUILD_DIR)/unsharp_mask_cuda \
              $(BUILD_DIR)/canny_edge_detector_cuda \
              $(BUILD_DIR)/knn_de_noising_cuda

# All targets
ALL_TARGETS = $(CPU_TARGETS) $(GPU_TARGETS)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Default target
all: $(BUILD_DIR) $(ALL_TARGETS)

# CPU builds
$(BUILD_DIR)/greyscale_cpu: src/greyscale/greyscale_cpu.cpp $(UTILS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_DIR)/pixelize_cpu: src/pixelize/pixelize_cpu.cpp $(UTILS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_DIR)/median_blur_cpu: src/median_blur/median_blur_cpu.cpp $(UTILS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_DIR)/rotate_cpu: src/rotate/rotate_cpu.cpp $(UTILS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_DIR)/unsharp_mask_cpu: src/unsharp_mask/unsharp_mask_cpu.cpp $(UTILS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_DIR)/canny_edge_detector_cpu: src/canny_edge_detector/canny_edge_detector_cpu.cpp $(UTILS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_DIR)/knn_de_noising_cpu: src/knn_de_noising/knn_de_noising_cpu.cpp $(UTILS_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ $(INCLUDES) -o $@

# GPU builds
$(BUILD_DIR)/greyscale_cuda: src/greyscale/greyscale_gpu.cu $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ $(INCLUDES) -o $@ -lstdc++fs

$(BUILD_DIR)/pixelize_cuda: src/pixelize/pixelize_gpu.cu $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ $(INCLUDES) -o $@ -lstdc++fs

$(BUILD_DIR)/median_blur_cuda: src/median_blur/median_blur_gpu.cu $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ $(INCLUDES) -o $@ -lstdc++fs

$(BUILD_DIR)/rotate_cuda: src/rotate/rotate_gpu.cu $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ $(INCLUDES) -o $@ -lstdc++fs

$(BUILD_DIR)/unsharp_mask_cuda: src/unsharp_mask/unsharp_mask_gpu.cu $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ $(INCLUDES) -o $@ -lstdc++fs

$(BUILD_DIR)/canny_edge_detector_cuda: src/canny_edge_detector/canny_edge_detector_gpu.cu $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ $(INCLUDES) -o $@ -lstdc++fs

$(BUILD_DIR)/knn_de_noising_cuda: src/knn_de_noising/knn_de_noising_gpu.cu $(UTILS_SRC) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ $(INCLUDES) -o $@ -lstdc++fs

# Build only CPU targets
cpu: $(BUILD_DIR) $(CPU_TARGETS)

# Build only GPU targets
gpu: $(BUILD_DIR) $(GPU_TARGETS)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Clean all (including executables and object files)
clean-all: clean
	rm -f *.o

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all CPU and GPU executables (default)"
	@echo "  cpu          - Build only CPU executables"
	@echo "  gpu          - Build only GPU executables"
	@echo "  clean        - Remove build directory"
	@echo "  clean-all    - Remove build directory and object files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "All executables are built in the $(BUILD_DIR)/ directory"
	@echo ""
	@echo "CPU executables:"
	@echo "  - $(BUILD_DIR)/greyscale_cpu"
	@echo "  - $(BUILD_DIR)/pixelize_cpu"
	@echo "  - $(BUILD_DIR)/median_blur_cpu"
	@echo "  - $(BUILD_DIR)/rotate_cpu"
	@echo "  - $(BUILD_DIR)/unsharp_mask_cpu"
	@echo "  - $(BUILD_DIR)/canny_edge_detector_cpu"
	@echo "  - $(BUILD_DIR)/knn_de_noising_cpu"
	@echo ""
	@echo "GPU executables:"
	@echo "  - $(BUILD_DIR)/greyscale_cuda"
	@echo "  - $(BUILD_DIR)/pixelize_cuda"
	@echo "  - $(BUILD_DIR)/median_blur_cuda"
	@echo "  - $(BUILD_DIR)/rotate_cuda"
	@echo "  - $(BUILD_DIR)/unsharp_mask_cuda"
	@echo "  - $(BUILD_DIR)/canny_edge_detector_cuda"
	@echo "  - $(BUILD_DIR)/knn_de_noising_cuda"

.PHONY: all cpu gpu clean clean-all help
