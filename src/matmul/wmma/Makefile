CC = nvcc
CFLAGS = -O3 -Wno-deprecated-gpu-targets
ARCH = -arch=sm_90
LIBS =

TARGET = matmul_wmma
SRC = matmul_wmma.cu

.PHONY: all run profile clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(ARCH) $< -o $@ $(LIBS)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	nsys profile --trace=cuda,nvtx --output=$(TARGET) --stats=true --force-overwrite true ./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).nsys-rep $(TARGET).sqlite *.qdrep *.qdstrm
