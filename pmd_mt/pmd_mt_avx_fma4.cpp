#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   0
#define USE_AVX2  0
#define USE_FMA3  0
#define USE_FMA4  1

#include "pmd_mt_simd.h"

void pmd_mt_avx_fma4(int thread_id, int thread_num, void *param1, void *param2) {
    pmd_mt_simd(thread_id, thread_num, param1, param2);
}

void anisotropic_mt_avx_fma4(int thread_id, int thread_num, void *param1, void *param2) {
    anisotropic_mt_simd(thread_id, thread_num, param1, param2);
}
