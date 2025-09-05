# Define the compiler
CXX = g++

# Define the source file
SRC = example.cpp

# Define the include directories
RKNPU_INC_DIR = -I/usr/local/include/rknpu/

OPENCV_INC_DIR = -I/usr/local/include/opencv2/

CURRENT_INC_DIR = -I./include

LIB_DIR = -L/usr/local/lib

# Define the libraries to link against
LIBS = -lrknnrt -fopenmp -lopencv_core -pthread

# Define the compilation flags
CXX_INCLUDE_FLAGS = $(RKNPU_INC_DIR) $(OPENCV_INC_DIR) $(CURRENT_INC_DIR)

CXX_LIB_FLAGS =  $(LIB_DIR) $(LIBS) 

CXXFLAGS = -Wall -O3 -std=c++20

# Define the rule to build the target
example: example.cpp
	$(CXX) example.cpp -o example $(CXX_INCLUDE_FLAGS) $(CXX_LIB_FLAGS) $(CXXFLAGS)

test: test.cpp
	$(CXX) test.cpp -o test $(CXX_INCLUDE_FLAGS) $(CXX_LIB_FLAGS) $(CXXFLAGS)

opencv: example_opencv.cpp
	$(CXX) example_opencv.cpp -o example_opencv $(CXX_INCLUDE_FLAGS) $(CXX_LIB_FLAGS) $(CXXFLAGS)
	
matmul: matmul.cpp
	$(CXX) matmul.cpp -o matmul $(CXX_INCLUDE_FLAGS) $(CXX_LIB_FLAGS) $(CXXFLAGS)
	
matmul_C: matmul.cpp
	$(CXX) -c matmul_C.cpp -o matmul_C.o $(CXX_INCLUDE_FLAGS) $(CXX_LIB_FLAGS) $(CXXFLAGS)
# Define the rule to clean up generated files
.PHONY: clean
clean:
	rm -f example test opencv matmul matmul_C
