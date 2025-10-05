NVCC = nvcc
CFLAGS = -O3 -Wno-deprecated-gpu-targets
ARCH = -arch=sm_90
LIBS = -lcudart -lcuda


TARGET = matmul_wmma_tiling
SRC = matmul_wmma_tiling.cu

.PHONY: all run profile clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(ARCH) $< -o $@ $(LIBS)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	nsys profile --trace=cuda,nvtx --output=$(TARGET) --stats=true --force-overwrite true ./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).nsys-rep $(TARGET).sqlite *.qdrep *.qdstrm
