#include <iostream>
#include <immintrin.h>
#include <vector>
#include <omp.h>

extern "C" {

    void matmul(float* result, const float* matrix1, const float* matrix2, int rows1, int cols1, int cols2) {
        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                result[i * cols2 + j] = 0.0f;
                for (int k = 0; k < cols1; k++) {
                    result[i * cols2 + j] += matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
                }
            }
        }
    }

    static inline float _mm256_reduce_add_ps(__m256 x) {
        const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        return _mm_cvtss_f32(x32);
    }

    void simd_matmul(float *result, const float *matrix1, const float *matrix2, int rows1, int cols1, int cols2) {

        for (int i = 0; i < rows1; ++i) {
            for (int j = 0; j < cols2; ++j) {
                __m256 sum = _mm256_setzero_ps();

                for (int k = 0; k < cols1; k += 8) {
                    __m256 a = _mm256_loadu_ps(matrix1 + i * cols1 + k);
                    __m256 b = _mm256_loadu_ps(matrix2 + k * cols2 + j);
                    sum = _mm256_fmadd_ps(a, b, sum);

                }

                result[i * cols2 + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
            }
        }
    }
}