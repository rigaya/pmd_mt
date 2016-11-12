#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   1
#define USE_AVX2  1
#define USE_FMA3  1
#define USE_FMA4  0
#define USE_VPGATHER 1 //Broadwell以降ではvpgatherを使用したほうが高速
#define USE_FMATH 0    //expのほうはfmathでexp計算をするよりも表引きのほうが高速

#if USE_FMATH
//ここではxbyakを使用しないほうが高速
//#define FMATH_USE_XBYAK
#include <fmath.hpp>
#endif

#include <cstdint>
#include <cmath>
#include "pmd_mt.h"
#include "filter.h"
#include "simd_util.h"
#include "pmd_mt_avx2.h"

void pmd_mt_exp_avx2_gather(int thread_id, int thread_num, void *param1, void *param2) {
    pmd_mt_exp_avx2_base(thread_id, thread_num, param1, param2);
}

void anisotropic_mt_exp_avx2_gather(int thread_id, int thread_num, void *param1, void *param2) {
    anisotropic_mt_exp_avx2_base(thread_id, thread_num, param1, param2);
}
