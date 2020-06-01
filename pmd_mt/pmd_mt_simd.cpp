//----------------------------------------------------------------------------------
//        PMD_MT
//----------------------------------------------------------------------------------

#include <intrin.h>
#include <cstdlib>
#include "pmd_mt.h"

enum {
    NONE        = 0x0000,
    SSE2        = 0x0001,
    SSE3        = 0x0002,
    SSSE3       = 0x0004,
    SSE41       = 0x0008,
    SSE42       = 0x0010,
    POPCNT      = 0x0020,
    XOP         = 0x0040,
    AVX         = 0x0080,
    AVX2        = 0x0100,
    FMA3        = 0x0200,
    FMA4        = 0x0400,
    FAST_GATHER = 0x00800,
    AVX2FAST    = 0x001000,
    AVX512F     = 0x002000,
    AVX512DQ    = 0x004000,
    AVX512IFMA  = 0x008000,
    AVX512PF    = 0x010000,
    AVX512ER    = 0x020000,
    AVX512CD    = 0x040000,
    AVX512BW    = 0x080000,
    AVX512VL    = 0x100000,
    AVX512VBMI  = 0x200000,
    AVX512VNNI  = 0x400000,
};

static DWORD get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    DWORD simd = NONE;
    if (CPUInfo[3] & 0x04000000) simd |= SSE2;
    if (CPUInfo[2] & 0x00000001) simd |= SSE3;
    if (CPUInfo[2] & 0x00000200) simd |= SSSE3;
    if (CPUInfo[2] & 0x00080000) simd |= SSE41;
    if (CPUInfo[2] & 0x00100000) simd |= SSE42;
    if (CPUInfo[2] & 0x00800000) simd |= POPCNT;
#if (_MSC_VER >= 1600)
    UINT64 xgetbv = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        xgetbv = _xgetbv(0);
        if ((xgetbv & 0x06) == 0x06)
            simd |= AVX;
#if (_MSC_VER >= 1700)
        if(CPUInfo[2] & 0x00001000 )
            simd |= FMA3;
#endif //(_MSC_VER >= 1700)
    }
#endif
#if (_MSC_VER >= 1700)
    __cpuid(CPUInfo, 7);
    if (simd & AVX) {
        if (CPUInfo[1] & 0x00000020)
            simd |= AVX2;
        if (CPUInfo[1] & (1<<18)) //rdseed -> Broadwell
            simd |= FAST_GATHER;
        if ((simd & AVX) && ((xgetbv >> 5) & 7) == 7) {
            if (CPUInfo[1] & (1u << 16)) simd |= AVX512F;
            if (simd & AVX512F) {
                if (CPUInfo[1] & (1u << 17)) simd |= AVX512DQ;
                if (CPUInfo[1] & (1u << 21)) simd |= AVX512IFMA;
                if (CPUInfo[1] & (1u << 26)) simd |= AVX512PF;
                if (CPUInfo[1] & (1u << 27)) simd |= AVX512ER;
                if (CPUInfo[1] & (1u << 28)) simd |= AVX512CD;
                if (CPUInfo[1] & (1u << 30)) simd |= AVX512BW;
                if (CPUInfo[1] & (1u << 31)) simd |= AVX512VL;
                if (CPUInfo[2] & (1u <<  1)) simd |= AVX512VBMI;
                if (CPUInfo[2] & (1u << 11)) simd |= AVX512VNNI;
            }
        }
        __cpuid(CPUInfo, 0x80000001);
        if (CPUInfo[2] & 0x00000800)
            simd |= XOP;
        if (CPUInfo[2] & 0x00010000)
            simd |= FMA4;
    }
#endif
    return simd;
}

static const PMD_MT_FUNC FUNC_LIST[] = {
    { gaussianH_avx2,  gaussianV_avx2,  gaussianHV_avx512vbmivnni, { { anisotropic_mt_avx512,     anisotropic_mt_exp_avx512      }, { pmd_mt_avx512,    pmd_mt_exp_avx512vnni  } }, AVX512VNNI|AVX512VBMI|AVX512BW|AVX512DQ|AVX512F|AVX2|FMA3|AVX },
    { gaussianH_avx2,  gaussianV_avx2,  gaussianHV_avx512,         { { anisotropic_mt_avx512,     anisotropic_mt_exp_avx512      }, { pmd_mt_avx512,    pmd_mt_exp_avx512      } }, AVX512BW|AVX512DQ|AVX512F|AVX2|FMA3|AVX },
    { gaussianH_avx2,  gaussianV_avx2,  gaussianHV_avx2,           { { anisotropic_mt_avx2_fma3,  anisotropic_mt_exp_avx2_gather }, { pmd_mt_avx2_fma3, pmd_mt_exp_avx2_gather } }, FAST_GATHER|AVX2|FMA3|AVX },
    { gaussianH_avx2,  gaussianV_avx2,  gaussianHV_avx2,           { { anisotropic_mt_avx2_fma3,  anisotropic_mt_exp_avx2        }, { pmd_mt_avx2_fma3, pmd_mt_exp_avx2        } }, AVX2|FMA3|AVX },
    { gaussianH_avx,   gaussianV_avx,   NULL,                      { { anisotropic_mt_avx_fma3,   anisotropic_mt_exp_avx         }, { pmd_mt_avx_fma3,  pmd_mt_exp_avx         } }, FMA3|AVX },
    { gaussianH_avx,   gaussianV_avx,   NULL,                      { { anisotropic_mt_avx_fma4,   anisotropic_mt_exp_avx         }, { pmd_mt_avx_fma4,  pmd_mt_exp_avx         } }, FMA4|AVX },
    { gaussianH_avx,   gaussianV_avx,   NULL,                      { { anisotropic_mt_avx,        anisotropic_mt_exp_avx         }, { pmd_mt_avx,       pmd_mt_exp_avx         } }, AVX|SSE41|SSSE3|SSE2 },
    { gaussianH_sse41, gaussianV_sse41, NULL,                      { { anisotropic_mt_sse41,      anisotropic_mt_exp_sse41       }, { pmd_mt_sse41,     pmd_mt_exp_sse41       } }, SSE41|SSSE3|SSE2 },
    { gaussianH_ssse3, gaussianV_ssse3, NULL,                      { { anisotropic_mt_ssse3,      anisotropic_mt_exp_ssse3       }, { pmd_mt_ssse3,     pmd_mt_exp_ssse3       } }, SSSE3|SSE2 },
    { gaussianH_sse2,  gaussianV_sse2,  NULL,                      { { anisotropic_mt_sse2,       anisotropic_mt_exp_sse2        }, { pmd_mt_sse2,      pmd_mt_exp_sse2        } }, SSE2 },
    { gaussianH,       gaussianV,       NULL,                      { { anisotropic_mt,            anisotropic_mt                 }, { pmd_mt,           pmd_mt                 } }, NONE },
};

const PMD_MT_FUNC *get_pmd_func_list() {
    const DWORD simd_avail = get_availableSIMD();
    for (int i = 0; i < _countof(FUNC_LIST); i++) {
        if ((FUNC_LIST[i].simd & simd_avail) == FUNC_LIST[i].simd) {
            return &FUNC_LIST[i];
        }
    }
    return NULL;
};

void avx2_dummy() {
    __asm {
        vpxor ymm0, ymm0, ymm0
    };
}

