//----------------------------------------------------------------------------------
//		PMD_MT
//----------------------------------------------------------------------------------

#include <intrin.h>
#include "pmd_mt.h"

enum {
    NONE   = 0x0000,
    SSE2   = 0x0001,
    SSE3   = 0x0002,
    SSSE3  = 0x0004,
    SSE41  = 0x0008,
    SSE42  = 0x0010,
	POPCNT = 0x0020,
    AVX    = 0x0040,
    AVX2   = 0x0080,
	FMA3   = 0x0100,
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
	if ((simd & AVX) && (CPUInfo[1] & 0x00000020))
		simd |= AVX2;
#endif
	return simd;
}

static const PMD_MT_FUNC FUNC_LIST[] = {
	{ gaussianH_avx2,  gaussianV_avx2,  { { anisotropic_mt_avx2_fma3,  anisotropic_mt_exp_avx2  }, { pmd_mt_avx2_fma3, pmd_mt_exp_avx2  } }, FMA3|AVX2|AVX },
	{ gaussianH_avx,   gaussianV_avx,   { { anisotropic_mt_avx_fma3,   anisotropic_mt_exp_avx   }, { pmd_mt_avx_fma3,  pmd_mt_exp_avx   } }, FMA3|AVX },
	{ gaussianH_avx,   gaussianV_avx,   { { anisotropic_mt_avx,        anisotropic_mt_exp_avx   }, { pmd_mt_avx,       pmd_mt_exp_avx   } }, AVX|SSE41|SSSE3|SSE2 },
	{ gaussianH_sse41, gaussianV_sse41, { { anisotropic_mt_sse41,      anisotropic_mt_exp_sse41 }, { pmd_mt_sse41,     pmd_mt_exp_sse41 } }, SSE41|SSSE3|SSE2 },
	{ gaussianH_ssse3, gaussianV_ssse3, { { anisotropic_mt_ssse3,      anisotropic_mt_exp_ssse3 }, { pmd_mt_ssse3,     pmd_mt_exp_ssse3 } }, SSSE3|SSE2 },
	{ gaussianH_sse2,  gaussianV_sse2,  { { anisotropic_mt_sse2,       anisotropic_mt_exp_sse2  }, { pmd_mt_sse2,      pmd_mt_exp_sse2  } }, SSE2 },
	{ gaussianH,       gaussianV,       { { anisotropic_mt,            anisotropic_mt           }, { pmd_mt,           pmd_mt           } }, NONE },
};

const PMD_MT_FUNC *get_pmd_func_list() {
	const DWORD simd_avail = get_availableSIMD() & ~FMA3;
	for (int i = 0; i < _countof(FUNC_LIST); i++) {
		if ((FUNC_LIST[i].simd & simd_avail) == FUNC_LIST[i].simd) {
			return &FUNC_LIST[i];
		}
	}
	return NULL;
};
