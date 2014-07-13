#pragma once

#include <emmintrin.h> //SSE2
#if USE_SSSE3
#include <tmmintrin.h> //SSSE3
#endif
#if USE_SSE41
#include <smmintrin.h> //SSE4.1
#endif
#if USE_AVX || USE_FMA3
#include <immintrin.h> //AVX
#endif
#if USE_FMA4
#include <intrin.h>
#endif

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です
template<bool dst_aligned>
static void __forceinline memcpy_sse(uint8_t *dst, const uint8_t *src, int size) {
	if (size < 64) {
		memcpy(dst, src, size);
		return;
	}
	uint8_t *dst_fin = dst + size;
	uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 15) & ~15) - 64);
	__m128 x0, x1, x2, x3;
	if (!dst_aligned) {
		const int start_align_diff = (int)((size_t)dst & 15);
		if (start_align_diff) {
			x0 = _mm_loadu_ps((float*)src);
			_mm_storeu_ps((float*)dst, x0);
			dst += 16 - start_align_diff;
			src += 16 - start_align_diff;
		}
	}
	for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
		x0 = _mm_loadu_ps((float*)(src +  0));
		x1 = _mm_loadu_ps((float*)(src + 16));
		x2 = _mm_loadu_ps((float*)(src + 32));
		x3 = _mm_loadu_ps((float*)(src + 48));
		_mm_store_ps((float*)(dst +  0), x0);
		_mm_store_ps((float*)(dst + 16), x1);
		_mm_store_ps((float*)(dst + 32), x2);
		_mm_store_ps((float*)(dst + 48), x3);
	}
	uint8_t *dst_tmp = dst_fin - 64;
	src -= (dst - dst_tmp);
	x0 = _mm_loadu_ps((float*)(src +  0));
	x1 = _mm_loadu_ps((float*)(src + 16));
	x2 = _mm_loadu_ps((float*)(src + 32));
	x3 = _mm_loadu_ps((float*)(src + 48));
	_mm_storeu_ps((float*)(dst_tmp +  0), x0);
	_mm_storeu_ps((float*)(dst_tmp + 16), x1);
	_mm_storeu_ps((float*)(dst_tmp + 32), x2);
	_mm_storeu_ps((float*)(dst_tmp + 48), x3);
}
#pragma warning (pop)

//r0 := (mask0 & 0x80) ? b0 : a0
//SSE4.1の_mm_blendv_epi8(__m128i a, __m128i b, __m128i mask) のSSE2版のようなもの
static inline __m128i select_by_mask(__m128i a, __m128i b, __m128i mask) {
	return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
}
//SSSE3のpalignrもどき
#define palignr_sse2(a,b,i) _mm_or_si128( _mm_slli_si128(a, 16-i), _mm_srli_si128(b, i) )

static __forceinline __m128i blendv_epi8_simd(__m128i a, __m128i b, __m128i mask) {
#if USE_SSE41
	return _mm_blendv_epi8(a, b, mask);
#else
	return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
#endif
}

#if USE_SSSE3
#define _mm_alignr_epi8_simd _mm_alignr_epi8
#else
#define _mm_alignr_epi8_simd palignr_sse2
#endif

static inline __m128i _mm_abs_epi16_simd(__m128i x0) {
#if USE_SSSE3
	x0 = _mm_abs_epi16(x0);
#else
	__m128i x1;
	x1 = _mm_setzero_si128();
	x1 = _mm_cmpgt_epi16(x1, x0);
	x0 = _mm_xor_si128(x0, x1);
	x0 = _mm_subs_epi16(x0, x1);
#endif
	return x0;
}

static __forceinline __m128i _mm_packus_epi32_simd(__m128i a, __m128i b) {
#if USE_SSE41
	return _mm_packus_epi32(a, b);
#else
	static const _declspec(align(64)) unsigned int VAL[2][4] = {
		{ 0x00008000, 0x00008000, 0x00008000, 0x00008000 },
		{ 0x80008000, 0x80008000, 0x80008000, 0x80008000 }
	};
#define LOAD_32BIT_0x8000 _mm_load_si128((__m128i *)VAL[0])
#define LOAD_16BIT_0x8000 _mm_load_si128((__m128i *)VAL[1])
	a = _mm_sub_epi32(a, LOAD_32BIT_0x8000);
	b = _mm_sub_epi32(b, LOAD_32BIT_0x8000);
	a = _mm_packs_epi32(a, b);
	return _mm_add_epi16(a, LOAD_16BIT_0x8000);
#undef LOAD_32BIT_0x8000
#undef LOAD_16BIT_0x8000
#endif
}

static __forceinline __m128i cvtlo_epi16_epi32(__m128i x0) {
#if USE_SSE41
	return _mm_cvtepi16_epi32(x0);
#else
	static const _declspec(align(64)) unsigned int VAL[2][4] = {
		{ 0x80008000, 0x80008000, 0x80008000, 0x80008000 },
		{ 0x00008000, 0x00008000, 0x00008000, 0x00008000 },
	};
	return _mm_sub_epi32(_mm_unpacklo_epi16(_mm_add_epi16(x0, _mm_load_si128((__m128i *)VAL[0])), _mm_setzero_si128()), _mm_load_si128((__m128i *)VAL[1]));
#endif
}

static __forceinline __m128i cvthi_epi16_epi32(__m128i x0) {
	return cvtlo_epi16_epi32(_mm_srli_si128(x0, 8));
}

static __forceinline __m128i _mm_mullo_epi32_simd(__m128i x0, __m128i x1) {
#if USE_SSE41
	return _mm_mullo_epi32(x0, x1);
#else
	__m128i xResult;
	xResult.m128i_i32[0] = x0.m128i_i32[0] * x1.m128i_i32[0];
	xResult.m128i_i32[1] = x0.m128i_i32[1] * x1.m128i_i32[1];
	xResult.m128i_i32[2] = x0.m128i_i32[2] * x1.m128i_i32[2];
	xResult.m128i_i32[3] = x0.m128i_i32[3] * x1.m128i_i32[3];
	return xResult;
#endif
}

static __forceinline __m128 _mm_madd_ps(__m128 x0, __m128 x1, __m128 x2) {
#if USE_FMA4
	return _mm_macc_ps(x0, x1, x2);
#elif USE_FMA3
	return _mm_fmadd_ps(x0, x1, x2);
#else
	return _mm_add_ps(_mm_mul_ps(x0, x1), x2);
#endif
}

static __forceinline __m128 _mm_rcp_ps_hp(__m128 x0) {
	__m128 x1, x2;
	x1 = _mm_rcp_ps(x0);
	x0 = _mm_mul_ps(x0, x1);
	x2 = _mm_add_ps(x1, x1);
#if USE_FMA4
	x2 = _mm_nmacc_ps(x0, x1, x2);
#elif USE_FMA3
	x2 = _mm_fnmadd_ps(x0, x1, x2);
#else
	x0 = _mm_mul_ps(x0, x1);
	x2 = _mm_sub_ps(x2, x0);
#endif
	return x2;
}

#if USE_AVX
static __forceinline __m256 _mm256_madd_ps(__m256 y0, __m256 y1, __m256 y2) {
#if USE_FMA3
	return _mm256_fmadd_ps(y0, y1, y2);
#else
	return _mm256_add_ps(_mm256_mul_ps(y0, y1), y2);
#endif
}
#endif

#if USE_AVX
static __forceinline __m256 _mm256_rcp_ps_hp(__m256 y0) {
	__m256 y1, y2;
	y1 = _mm256_rcp_ps(y0);
	y0 = _mm256_mul_ps(y0, y1);
	y2 = _mm256_add_ps(y1, y1);
#if USE_FMA3
	y2 = _mm256_fnmadd_ps(y0, y1, y2);
#else
	y0 = _mm256_mul_ps(y0, y1);
	y2 = _mm256_sub_ps(y2, y0);
#endif
	return y2;
}
#endif

#if USE_AVX2

#include <immintrin.h>

//本来の256bit alignr
#define MM_ABS(x) (((x) < 0) ? -(x) : (x))
#define _mm256_alignr256_epi8(a, b, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), b, i) : _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), MM_ABS(i-16)))

//_mm256_srli_si256, _mm256_slli_si256は
//単に128bitシフト×2をするだけの命令である
#define _mm256_bsrli_epi128 _mm256_srli_si256
#define _mm256_bslli_epi128 _mm256_slli_si256

//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))

static __forceinline __m256i cvtlo256_epi16_epi32(__m256i y0) {
	static const _declspec(align(64)) unsigned int VAL[2][8] = {
		{ 0x80008000, 0x80008000, 0x80008000, 0x80008000, 0x80008000, 0x80008000, 0x80008000, 0x80008000 },
		{ 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000 },
	};
	return _mm256_sub_epi32(_mm256_unpacklo_epi16(_mm256_add_epi16(y0, _mm256_load_si256((__m256i *)VAL[0])), _mm256_setzero_si256()), _mm256_load_si256((__m256i *)VAL[1]));
}

static __forceinline __m256i cvthi256_epi16_epi32(__m256i y0) {
	return cvtlo256_epi16_epi32(_mm256_srli_si256(y0, 8));
}


#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です
template<bool dst_aligned, bool use_stream, bool zeroupper>
static void __forceinline memcpy_avx2(uint8_t *dst, const uint8_t *src, int size) {
	if (size < 128) {
		memcpy(dst, src, size);
		return;
	}
	uint8_t *dst_fin = dst + size;
	uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 31) & ~31) - 128);
	__m256i y0, y1, y2, y3;
	if (!dst_aligned) {
		const int start_align_diff = (int)((size_t)dst & 31);
		if (start_align_diff) {
			y0 = _mm256_loadu_si256((__m256i*)src);
			_mm256_storeu_si256((__m256i*)dst, y0);
			dst += 32 - start_align_diff;
			src += 32 - start_align_diff;
		}
	}
#define _mm256_stream_switch_si256(x, ymm) ((use_stream) ? _mm256_stream_si256((x), (ymm)) : _mm256_store_si256((x), (ymm)))
	for ( ; dst < dst_aligned_fin; dst += 128, src += 128) {
		y0 = _mm256_loadu_si256((const __m256i*)(src +  0));
		y1 = _mm256_loadu_si256((const __m256i*)(src + 32));
		y2 = _mm256_loadu_si256((const __m256i*)(src + 64));
		y3 = _mm256_loadu_si256((const __m256i*)(src + 96));
		_mm256_stream_switch_si256((__m256i*)(dst +  0), y0);
		_mm256_stream_switch_si256((__m256i*)(dst + 32), y1);
		_mm256_stream_switch_si256((__m256i*)(dst + 64), y2);
		_mm256_stream_switch_si256((__m256i*)(dst + 96), y3);
	}
#undef _mm256_stream_switch_si256
	uint8_t *dst_tmp = dst_fin - 128;
	src -= (dst - dst_tmp);
	y0 = _mm256_loadu_si256((const __m256i*)(src +  0));
	y1 = _mm256_loadu_si256((const __m256i*)(src + 32));
	y2 = _mm256_loadu_si256((const __m256i*)(src + 64));
	y3 = _mm256_loadu_si256((const __m256i*)(src + 96));
	_mm256_storeu_si256((__m256i*)(dst_tmp +  0), y0);
	_mm256_storeu_si256((__m256i*)(dst_tmp + 32), y1);
	_mm256_storeu_si256((__m256i*)(dst_tmp + 64), y2);
	_mm256_storeu_si256((__m256i*)(dst_tmp + 96), y3);
	if (zeroupper) {
		_mm256_zeroupper();
	}
}
#pragma warning (pop)

#endif
