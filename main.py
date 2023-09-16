import ctypes
import numpy as np
import time

# Load the C++ shared library
lib = ctypes.CDLL('./fast_proximal_grad/matmul.so')  # Replace with the path to your compiled shared library

# Define the matrix multiplication function signature
matmul = lib.simd_matmul
matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # Result matrix
    ctypes.POINTER(ctypes.c_float),  # Matrix 1
    ctypes.POINTER(ctypes.c_float),  # Matrix 2
    ctypes.c_int,                    # Rows of Matrix 1
    ctypes.c_int,                    # Columns of Matrix 1
    ctypes.c_int                     # Columns of Matrix 2
]

# Create random matricies of given dimensions aligning memory to 32 byte boundaries
rows1, cols1 = (6, 6)
rows2, cols2 = (6, 6)
matrix1 = np.random.rand(rows1, cols1).astype(np.float32)
matrix2 = np.random.rand(rows2, cols2).astype(np.float32)

def aligned_array(shape, dtype, alignment=32):
    buffer = np.empty(shape, dtype)
    if buffer.ctypes.data % alignment != 0:
        offset = alignment - (buffer.ctypes.data % alignment)
        buffer = buffer[offset:]
    return buffer


start = time.time()
C = matrix1 @ matrix2
end = time.time()
print("Numpy time: ", end - start)

# Create a result matrix
result = np.zeros((rows1, cols2), dtype=np.float32)

# Call the C++ function
start = time.time()
matmul(result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                matrix1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                matrix2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rows1, cols1, cols2)
end = time.time()
print("C++ time: ", end - start)

#print (C)
#print (result)

for i in range(rows1):
    for j in range(cols2):
        if C[i][j] != result[i][j]:
            print("i: ", i, " j: ", j, " C: ", C[i][j], " result: ", result[i][j])
            print("Error")
            exit(1)
    
