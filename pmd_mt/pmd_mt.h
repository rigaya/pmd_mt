#pragma once

#include <windows.h>
#include "filter.h"

#define SIMD_DEBUG 0
#define CHECK_PERFORMANCE 0

typedef struct PMD_MT_PRM {
    PIXEL_YC *gauss;
    int strength, threshold;
    int *pmd;
} PMD_MT_PRM;

typedef struct PMD_MT_FUNC {
    MULTI_THREAD_FUNC gaussianH;
    MULTI_THREAD_FUNC gaussianV;
    MULTI_THREAD_FUNC gaussianHV;
    MULTI_THREAD_FUNC main_func[2][2];
    DWORD simd;
} PMD_MT_FUNC;

static const int PMD_TABLE_SIZE = 4500;

const PMD_MT_FUNC *get_pmd_func_list();

void gaussianH(int thread_id, int thread_num, void *param1, void *param2);
void gaussianH_sse2(int thread_id, int thread_num, void *param1, void *param2);
void gaussianH_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void gaussianH_sse41(int thread_id, int thread_num, void *param1, void *param2);
void gaussianH_avx(int thread_id, int thread_num, void *param1, void *param2);
void gaussianH_avx2(int thread_id, int thread_num, void *param1, void *param2);

void gaussianV(int thread_id, int thread_num, void *param1, void *param2);
void gaussianV_sse2(int thread_id, int thread_num, void *param1, void *param2);
void gaussianV_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void gaussianV_sse41(int thread_id, int thread_num, void *param1, void *param2);
void gaussianV_avx(int thread_id, int thread_num, void *param1, void *param2);
void gaussianV_avx2(int thread_id, int thread_num, void *param1, void *param2);

void gaussianHV_avx2(int thread_id, int thread_num, void *param1, void *param2);

void pmd_mt(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt(int thread_id, int thread_num, void *param1, void *param2);

void pmd_mt_sse2(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_sse41(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_avx(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_avx_fma3(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_avx_fma4(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_avx2_fma3(int thread_id, int thread_num, void *param1, void *param2);

void pmd_mt_exp_sse2(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_exp_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_exp_sse41(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_exp_avx(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_exp_avx2(int thread_id, int thread_num, void *param1, void *param2);
void pmd_mt_exp_avx2_gather(int thread_id, int thread_num, void *param1, void *param2);

void anisotropic_mt_sse2(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_sse41(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_avx(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_avx_fma3(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_avx_fma4(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_avx2_fma3(int thread_id, int thread_num, void *param1, void *param2);

void anisotropic_mt_exp_sse2(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_exp_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_exp_sse41(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_exp_avx(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_exp_avx2(int thread_id, int thread_num, void *param1, void *param2);
void anisotropic_mt_exp_avx2_gather(int thread_id, int thread_num, void *param1, void *param2);


typedef struct {
    int frame_count;
    __int64 tmp[16];
    __int64 value[15];
    __int64 freq;
} PERFORMANCE_CHECKER;

#pragma warning (push)
#pragma warning (disable: 4100)
static __forceinline void get_qp_counter(__int64 *qpc) {
#if CHECK_PERFORMANCE
    QueryPerformanceCounter((LARGE_INTEGER *)qpc);
#endif
}

static __forceinline void add_qpctime(__int64 *qpc, __int64 add) {
#if CHECK_PERFORMANCE
    *qpc += add;
#endif
}
#pragma warning (pop)
