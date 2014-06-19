#define USE_SSE2  1
#define USE_SSSE3 0
#define USE_SSE41 0
#define USE_AVX   0
#define USE_AVX2  0

#include "pmd_mt_simd.h"

void gaussianH_sse2(int thread_id, int thread_num, void *param1, void *param2) {
	gaussianH_simd(thread_id, thread_num, param1, param2);
}

void gaussianV_sse2(int thread_id, int thread_num, void *param1, void *param2) {
	gaussianV_simd(thread_id, thread_num, param1, param2);
}

void pmd_mt_sse2(int thread_id, int thread_num, void *param1, void *param2) {
	pmd_mt_simd(thread_id, thread_num, param1, param2);
}

void anisotropic_mt_sse2(int thread_id, int thread_num, void *param1, void *param2) {
	anisotropic_mt_simd(thread_id, thread_num, param1, param2);
}

void pmd_mt_exp_sse2(int thread_id, int thread_num, void *param1, void *param2) {
	pmd_mt_exp_simd(thread_id, thread_num, param1, param2);
}

void anisotropic_mt_exp_sse2(int thread_id, int thread_num, void *param1, void *param2) {
	anisotropic_mt_exp_simd(thread_id, thread_num, param1, param2);
}
