#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   1
#define USE_AVX2  1
#define USE_FMA3  1
#define USE_FMA4  0
#define USE_VPGATHER 1 //Haswellではvpgatherを使用したほうが遅い

#define USE_FMATH 0 //やはり表引きしたほうが速いので無効化

#include <cstdint>
#include <algorithm>
#include <cmath>
#include "pmd_mt.h"
#include "filter.h"
#include "simd_util.h"

#define _mm512_stream_switch_si512(x, zmm) ((aligned_store) ? _mm512_stream_si512((x), (zmm)) : _mm512_storeu_si512((x), (zmm)))

template<bool aligned_store>
static void __forceinline memcpy_avx512(void *_dst, void *_src, int size) {
    uint8_t *dst = (uint8_t *)_dst;
    uint8_t *src = (uint8_t *)_src;
    if (size < 256) {
        memcpy(dst, src, size);
        return;
    }
    uint8_t *dst_fin = dst + size;
    uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 63) & ~63) - 256);
    __m512i z0, z1, z2, z3;
    const int start_align_diff = (int)((size_t)dst & 63);
    if (start_align_diff) {
        z0 = _mm512_loadu_si512((__m512i*)src);
        _mm512_storeu_si512((__m512i*)dst, z0);
        dst += 64 - start_align_diff;
        src += 64 - start_align_diff;
    }
    for (; dst < dst_aligned_fin; dst += 256, src += 256) {
        z0 = _mm512_loadu_si512((__m512i*)(src +   0));
        z1 = _mm512_loadu_si512((__m512i*)(src +  64));
        z2 = _mm512_loadu_si512((__m512i*)(src + 128));
        z3 = _mm512_loadu_si512((__m512i*)(src + 192));
        _mm512_stream_switch_si512((__m512i*)(dst +   0), z0);
        _mm512_stream_switch_si512((__m512i*)(dst +  64), z1);
        _mm512_stream_switch_si512((__m512i*)(dst + 128), z2);
        _mm512_stream_switch_si512((__m512i*)(dst + 192), z3);
    }
    uint8_t *dst_tmp = dst_fin - 256;
    src -= (dst - dst_tmp);
    z0 = _mm512_loadu_si512((__m512i*)(src +   0));
    z1 = _mm512_loadu_si512((__m512i*)(src +  64));
    z2 = _mm512_loadu_si512((__m512i*)(src + 128));
    z3 = _mm512_loadu_si512((__m512i*)(src + 192));
    _mm512_storeu_si512((__m512i*)(dst_tmp +   0), z0);
    _mm512_storeu_si512((__m512i*)(dst_tmp +  64), z1);
    _mm512_storeu_si512((__m512i*)(dst_tmp + 128), z2);
    _mm512_storeu_si512((__m512i*)(dst_tmp + 192), z3);
}

static __forceinline __m512i cvtlo512_epi16_epi32(__m512i z0) {
    __mmask32 mWords = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), z0);
    return _mm512_unpacklo_epi16(z0, _mm512_movm_epi16(mWords));
}

static __forceinline __m512i cvthi512_epi16_epi32(__m512i z0) {
    __mmask32 mWords = _mm512_cmpgt_epi16_mask(_mm512_setzero_si512(), z0);
    return _mm512_unpackhi_epi16(z0, _mm512_movm_epi16(mWords));
}

// z0*1 + z1*4 + z2*6 + z3*4 + z4*1
template<bool avx512vnni>
static __forceinline __m512i gaussian_1_4_6_4_1(__m512i z0, __m512i z1, __m512i z2, const __m512i& z3, const __m512i& z4) {
    z0 = _mm512_adds_epi16(z0, z4);
    z1 = _mm512_adds_epi16(z1, z3);
    alignas(64) static const int16_t MUL[] = {
        4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6,
        4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6 };

    __m512i y0_lower = cvtlo512_epi16_epi32(z0);
    __m512i y0_upper = cvthi512_epi16_epi32(z0);
    if (avx512vnni) {
        y0_lower = _mm512_dpwssd_epi32(y0_lower, _mm512_unpacklo_epi16(z1, z2), _mm512_load_si512((__m512i *)MUL));
        y0_upper = _mm512_dpwssd_epi32(y0_upper, _mm512_unpackhi_epi16(z1, z2), _mm512_load_si512((__m512i *)MUL));
    } else {
        __m512i y1_lower = _mm512_madd_epi16(_mm512_unpacklo_epi16(z1, z2), _mm512_load_si512((__m512i *)MUL));
        __m512i y1_upper = _mm512_madd_epi16(_mm512_unpackhi_epi16(z1, z2), _mm512_load_si512((__m512i *)MUL));
        y0_lower = _mm512_add_epi32(y0_lower, y1_lower);
        y0_upper = _mm512_add_epi32(y0_upper, y1_upper);
    }
    y0_lower = _mm512_srai_epi32(y0_lower, 4);
    y0_upper = _mm512_srai_epi32(y0_upper, 4);
    return _mm512_packs_epi32(y0_lower, y0_upper);
}

template<int line_size>
static __forceinline void copy_bufline_avx512(void *dst, const void *src) {
    static_assert(line_size % 256 == 0, "line_size % 256");
    int n = line_size / 256;
    const char *srcptr = (const char *)src;
    char *dstptr = (char *)dst;
    do {
        __m512i z0 = _mm512_load_si512(srcptr + 0);
        __m512i z1 = _mm512_load_si512(srcptr + 64);
        __m512i z2 = _mm512_load_si512(srcptr + 128);
        __m512i z3 = _mm512_load_si512(srcptr + 192);
        _mm512_store_si512((__m512i *)(dstptr + 0), z0);
        _mm512_store_si512((__m512i *)(dstptr + 64), z1);
        _mm512_store_si512((__m512i *)(dstptr + 128), z2);
        _mm512_store_si512((__m512i *)(dstptr + 192), z3);
        srcptr += 256;
        dstptr += 256;
        n--;
    } while (n);
}

template<bool avx512vbmi>
static __forceinline void gather_y_u_v_to_yc48(__m512i& zY, __m512i& zU, __m512i& zV) {
    __m512i z0, z1, z2;
    if (avx512vbmi) {
        alignas(64) static const uint8_t shuffle_yc48_vbmi[] = {
              0,   1,  64,  65, 128, 129,   2,   3,  66,  67, 130, 131,   4,   5,  68,  69, 132, 133,   6,   7,  70,  71, 134, 135,   8,   9,  72,  73, 136, 137,  10,  11,
             74,  75, 138, 139,  12,  13,  76,  77, 140, 141,  14,  15,  78,  79, 142, 143,  16,  17,  80,  81, 144, 145,  18,  19,  82,  83, 146, 147,  20,  21,  84,  85
        };
        alignas(64) static const int8_t OFFSET20[] = {
            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20 };
        alignas(16) static const int8_t OFFSET42[] = {
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42 };
        __m512i zShuffle = _mm512_load_si512((__m512i *)shuffle_yc48_vbmi);
#if 0
        __mmask64 mask1 = _cvtu64_mask64(0xF3CF3CF3CF3CF3CF);
        __mmask64 mask2 = _cvtu64_mask64(0xCF3CF3CF3CF3CF3C);
        __mmask64 mask1not = _knot_mask64(mask1);
        __mmask64 mask2not = _knot_mask64(mask2);
        z0 = _mm512_mask_permutex2var_epi8(zY/*a*/, mask1, zShuffle/*idx*/, zU/*b*/);
        z0 = _mm512_mask_permutexvar_epi8(z0/*src*/, mask1not, zShuffle/*idx*/, zV);

        __m512i zShuffleM42 = _mm512_sub_epi8(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET42)));
        __m512i zShuffleP20 = _mm512_add_epi8(zShuffle, _mm512_load_si512((__m512i *)OFFSET20));
        z1 = _mm512_mask_permutex2var_epi8(zY/*a*/, mask2, zShuffleM42/*idx*/, zU/*b*/);
        z1 = _mm512_mask_permutexvar_epi8(z1/*src*/, mask2not, zShuffleP20/*idx*/, zV);

        __m512i zShuffleP42 = _mm512_add_epi8(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET42)));
        __m512i zShuffleM84 = _mm512_sub_epi8(zShuffleM42, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET42)));
        z2 = _mm512_mask_permutex2var_epi8(zU/*a*/, mask1, zShuffleP42/*idx*/, zV/*b*/);
        z2 = _mm512_mask_permutexvar_epi8(z2/*src*/, mask1not, zShuffleM84/*idx*/, zY);
#else
        //clangの提案により、以下のほうが速そう
        __mmask32 mask1 = 0xDB6DB6DBu;
        __mmask32 mask2 = 0xB6DB6DB6u;
        __mmask32 mask1not = ~mask1;
        __mmask32 mask2not = ~mask2;
        z0 = _mm512_permutex2var_epi8(zY/*a*/, zShuffle/*idx*/, zU/*b*/);
        z0 = _mm512_mask_mov_epi16(z0, mask1not, _mm512_permutexvar_epi8(zShuffle/*idx*/, zV));

        __m512i zShuffleM42 = _mm512_sub_epi8(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET42)));
        __m512i zShuffleP20 = _mm512_add_epi8(zShuffle, _mm512_load_si512((__m512i *)OFFSET20));
        z1 = _mm512_permutex2var_epi8(zY/*a*/, zShuffleM42/*idx*/, zU/*b*/);
        z1 = _mm512_mask_mov_epi16(z1, mask2not, _mm512_permutexvar_epi8(zShuffleP20/*idx*/, zV));

        __m512i zShuffleP42 = _mm512_add_epi8(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET42)));
        __m512i zShuffleM84 = _mm512_sub_epi8(zShuffleM42, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET42)));
        z2 = _mm512_permutex2var_epi8(zU/*a*/, zShuffleP42/*idx*/, zV/*b*/);
        z2 = _mm512_mask_mov_epi16(z2, mask1not, _mm512_permutexvar_epi8(zShuffleM84/*idx*/, zY));
#endif
    } else {
        alignas(64) static const uint16_t shuffle_yc48[] = {
            0, 32, 64,  1, 33, 65,  2, 34, 66,  3, 35, 67,  4, 36, 68,  5,
           37, 69,  6, 38, 70,  7, 39, 71,  8, 40, 72,  9, 41, 73, 10, 42
        };
        alignas(64) static const int16_t OFFSET10[] = {
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };
        alignas(16) static const int16_t OFFSET21[] = { 21, 21, 21, 21, 21, 21, 21, 21 };
        __mmask32 mask1 = 0xDB6DB6DBu;
        __mmask32 mask2 = 0xB6DB6DB6u;
        __mmask32 mask1not = ~mask1;
        __mmask32 mask2not = ~mask2;
        __m512i zShuffle = _mm512_load_si512((__m512i *)shuffle_yc48);
#if 0 //どちらでもあまり速度は変わらない
        z0 = _mm512_mask_permutex2var_epi16(zY/*a*/, mask1, zShuffle/*idx*/, zU/*b*/);
        z0 = _mm512_mask_permutexvar_epi16(z0/*src*/, mask1not, zShuffle/*idx*/, zV);

        __m512i zShuffleM21 = _mm512_sub_epi16(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET21)));
        __m512i zShuffleP10 = _mm512_add_epi16(zShuffle, _mm512_load_si512((__m512i *)OFFSET10));
        z1 = _mm512_mask_permutex2var_epi16(zY/*a*/, mask2, zShuffleM21/*idx*/, zU/*b*/);
        z1 = _mm512_mask_permutexvar_epi16(z1/*src*/, mask2not, zShuffleP10/*idx*/, zV);

        __m512i zShuffleP21 = _mm512_add_epi16(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET21)));
        __m512i zShuffleM42 = _mm512_sub_epi16(zShuffleM21, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET21)));
        z2 = _mm512_mask_permutex2var_epi16(zU/*a*/, mask1, zShuffleP21/*idx*/, zV/*b*/);
        z2 = _mm512_mask_permutexvar_epi16(z2/*src*/, mask1not, zShuffleM42/*idx*/, zY);
#else
        z0 = _mm512_permutex2var_epi16(zY/*a*/, zShuffle/*idx*/, zU/*b*/);
        z0 = _mm512_mask_mov_epi16(z0, mask1not, _mm512_permutexvar_epi16(zShuffle/*idx*/, zV));

        __m512i zShuffleM21 = _mm512_sub_epi16(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET21)));
        __m512i zShuffleP10 = _mm512_add_epi16(zShuffle, _mm512_load_si512((__m512i *)OFFSET10));
        z1 = _mm512_permutex2var_epi16(zY/*a*/, zShuffleM21/*idx*/, zU/*b*/);
        z1 = _mm512_mask_mov_epi16(z1, mask2not, _mm512_permutexvar_epi16(zShuffleP10/*idx*/, zV));

        __m512i zShuffleP21 = _mm512_add_epi16(zShuffle, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET21)));
        __m512i zShuffleM42 = _mm512_sub_epi16(zShuffleM21, _mm512_broadcast_i64x2(_mm_load_si128((const __m128i *)OFFSET21)));
        z2 = _mm512_permutex2var_epi16(zU/*a*/, zShuffleP21/*idx*/, zV/*b*/);
        z2 = _mm512_mask_mov_epi16(z2, mask1not, _mm512_permutexvar_epi16(zShuffleM42/*idx*/, zY));
#endif
    }

    zY = z0;
    zU = z1;
    zV = z2;
}

template<bool avx512vbmi>
static __forceinline void store_y_u_v_to_yc48(char *ptr, __m512i zY, __m512i zU, __m512i zV, bool store_per_pix, int n) {
    gather_y_u_v_to_yc48<avx512vbmi>(zY, zU, zV);
    if (store_per_pix) {
        __mmask32 mask = (1 << n) - 1;
        _mm512_mask_storeu_epi16((__m512i *)(ptr + 0), mask, zY);
        _mm512_mask_storeu_epi16((__m512i *)(ptr + 64), mask, zU);
        _mm512_mask_storeu_epi16((__m512i *)(ptr + 128), mask, zV);
    } else {
        _mm512_storeu_si512((__m512i *)(ptr + 0), zY);
        _mm512_storeu_si512((__m512i *)(ptr + 64), zU);
        _mm512_storeu_si512((__m512i *)(ptr + 128), zV);
    }
}

template<bool aligned, bool avx512vbmi>
void __forceinline afs_load_yc48(__m512i& y, __m512i& cb, __m512i& cr, const char *src) {
    alignas(64) static const uint16_t PACK_YC48_SHUFFLE_AVX512[32] = {
         0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
        48, 51, 54, 57, 60, 63,  2,  5,  8, 11, 14, 17, 20, 23, 26, 29
    };
    alignas(64) static const uint8_t PACK_YC48_SHUFFLE_AVX512_VBMI[64] = {
         0,   1,   6,   7,  12,  13,  18,  19,  24,  25,  30,  31, 36, 37, 42, 43, 48, 49, 54, 55, 60, 61, 66, 67, 72, 73, 78, 79, 84, 85, 90, 91,
        96,  97, 102, 103, 108, 109, 114, 115, 120, 121, 126, 127,  4,  5, 10, 11, 16, 17, 22, 23, 28, 29, 34, 35, 40, 41, 46, 47, 52, 53, 58, 59
    };
    __m512i z0 = _mm512_load_si512(avx512vbmi ? (__m512i *)PACK_YC48_SHUFFLE_AVX512_VBMI : (__m512i *)PACK_YC48_SHUFFLE_AVX512);
    __m512i z5 = (aligned) ? _mm512_load_si512((__m512i *)(src +   0)) : _mm512_loadu_si512((__m512i *)(src +   0));
    __m512i z4 = (aligned) ? _mm512_load_si512((__m512i *)(src +  64)) : _mm512_loadu_si512((__m512i *)(src +  64));
    __m512i z3 = (aligned) ? _mm512_load_si512((__m512i *)(src + 128)) : _mm512_loadu_si512((__m512i *)(src + 128));

    __m512i z1, z2;
    __m512i z6 = _mm512_ternarylogic_epi64(_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), 0xff);
    if (avx512vbmi) {
        z6 = _mm512_add_epi8(z6, z6);
#if 0
        __mmask64 k6 = _cvtu64_mask64(0xFFFFFC0000000000);
        __mmask64 k7 = _cvtu64_mask64(0xFFFFF00000000000);

        z1 = z0;
        z1 = _mm512_permutex2var_epi8(z5/*a*/, z1/*idx*/, z4/*b*/);
        z1 = _mm512_mask_permutexvar_epi8(z1/*src*/, k7, z0/*idx*/, z3);
        z0 = _mm512_sub_epi16(z0, z6);

        z2 = z0;
        z2 = _mm512_permutex2var_epi8(z5/*a*/, z2/*idx*/, z4/*b*/);
        z2 = _mm512_mask_permutexvar_epi8(z2/*src*/, k6, z0/*idx*/, z3);
        z0 = _mm512_sub_epi16(z0, z6);

        z6 = z0;
        z6 = _mm512_permutex2var_epi8(z5/*a*/, z6/*idx*/, z4/*b*/);
        z6 = _mm512_mask_permutexvar_epi8(z6/*src*/, k6, z0/*idx*/, z3);
#else
        __mmask32 k7 = 0xffc00000;
        __mmask32 k6 = 0xffe00000;

        z1 = z0;
        z1 = _mm512_permutex2var_epi8(z5/*a*/, z1/*idx*/, z4/*b*/);
        z1 = _mm512_mask_mov_epi16(z1, k7, _mm512_permutexvar_epi8(z0/*idx*/, z3));
        z0 = _mm512_sub_epi8(z0, z6);

        z2 = z0;
        z2 = _mm512_permutex2var_epi8(z5/*a*/, z2/*idx*/, z4/*b*/);
        z2 = _mm512_mask_mov_epi16(z2, k6, _mm512_permutexvar_epi8(z0/*idx*/, z3));
        z0 = _mm512_sub_epi8(z0, z6);

        z6 = z0;
        z6 = _mm512_permutex2var_epi8(z5/*a*/, z6/*idx*/, z4/*b*/);
        z6 = _mm512_mask_mov_epi16(z6, k6, _mm512_permutexvar_epi8(z0/*idx*/, z3));
#endif
    } else {
        __mmask32 k7 = 0xffc00000;
        __mmask32 k6 = 0xffe00000;

        z1 = z0;
        z1 = _mm512_permutex2var_epi16(z5/*a*/, z1/*idx*/, z4/*b*/);
#if 1 //どちらでもあまり速度は変わらない
        z1 = _mm512_mask_permutexvar_epi16(z1/*src*/, k7, z0/*idx*/, z3);
#else
        z1 = _mm512_mask_mov_epi16(z1, k7, _mm512_permutexvar_epi16(z0/*idx*/, z3));
#endif
        z0 = _mm512_sub_epi16(z0, z6);

        z2 = z0;
        z2 = _mm512_permutex2var_epi16(z5/*a*/, z2/*idx*/, z4/*b*/);
#if 1 //どちらでもあまり速度は変わらない
        z2 = _mm512_mask_permutexvar_epi16(z2/*src*/, k6, z0/*idx*/, z3);
#else
        z1 = _mm512_mask_mov_epi16(z2, k6, _mm512_permutexvar_epi16(z0/*idx*/, z3));
#endif
        z0 = _mm512_sub_epi16(z0, z6);

        z6 = z0;
        z6 = _mm512_permutex2var_epi16(z5/*a*/, z6/*idx*/, z4/*b*/);
#if 1 //どちらでもあまり速度は変わらない
        z6 = _mm512_mask_permutexvar_epi16(z6/*src*/, k6, z0/*idx*/, z3);
#else
        z6 = _mm512_mask_mov_epi16(z6, k6, _mm512_permutexvar_epi16(z0/*idx*/, z3));
#endif
    }
    y = z1;
    cb = z2;
    cr = z6;
}

#pragma warning (push)
#pragma warning (disable:4127) //warning  C4127: 条件式が定数です。
template<int shift>
static __forceinline __m512i _mm512_alignr512_epi8(const __m512i &z1, const __m512i &z0) {
    static_assert(0 <= shift && shift <= 64, "0 <= shift && shift <= 64");
    if (shift == 0) {
        return z0;
    } else if (shift == 64) {
        return z1;
    } else if (shift % 4 == 0) {
        return _mm512_alignr_epi32(z1, z0, shift / 4);
    } else if (shift <= 16) {
        __m512i z01 = _mm512_alignr_epi32(z1, z0, 4);
        return _mm512_alignr_epi8(z01, z0, shift);
    } else if (shift <= 32) {
        __m512i z010 = _mm512_alignr_epi32(z1, z0, 4);
        __m512i z011 = _mm512_alignr_epi32(z1, z0, 8);
        return _mm512_alignr_epi8(z011, z010, std::max(shift - 16, 0));
    } else if (shift <= 48) {
        __m512i z010 = _mm512_alignr_epi32(z1, z0, 8);
        __m512i z011 = _mm512_alignr_epi32(z1, z0, 12);
        return _mm512_alignr_epi8(z011, z010, std::max(shift - 32, 0));
    } else { //shift <= 64
        __m512i z01 = _mm512_alignr_epi32(z1, z0, 12);
        return _mm512_alignr_epi8(z1, z01, std::max(shift - 48, 0));
    }
}
#pragma warning (pop)

//1,2,1加算を行う
static __forceinline __m512i smooth_3x3_vertical(const __m512i &z0, const __m512i &z1, const __m512i &z2) {
    __m512i ySum = _mm512_add_epi16(_mm512_add_epi16(z1, z1), _mm512_set1_epi16(2));
    ySum = _mm512_add_epi16(ySum, _mm512_add_epi16(z0, z2));
    return _mm512_srai_epi16(ySum, 2);
}
//1,4,6,4,1加算
template<bool avx512vnni>
static __forceinline __m512i smooth_5x5_vertical(const __m512i& z0, const __m512i &z1, const __m512i &z2, const __m512i &z3, const __m512i &z4) {
    return gaussian_1_4_6_4_1<avx512vnni>(z0, z1, z2, z3, z4);
}
#pragma warning (push)
#pragma warning (disable:4100) //warning C4100: 引数は関数の本体部で 1 度も参照されません。
//1,6,15,20,15,6,1加算
static __forceinline __m512i smooth_7x7_vertical(const __m512i &z0, const __m512i &z1, const __m512i &z2, const __m512i &z3, const __m512i &z4, const __m512i &z5, const __m512i &z6) {
    // ###################
    //   !!!! 未実装 !!!!!
    // ###################
    return z3;
}
#pragma warning (pop)


//1,2,1加算を行う
static __forceinline __m512i smooth_3x3_horizontal(__m512i z0, __m512i z1, __m512i z2) {
    return smooth_3x3_vertical(
        _mm512_alignr512_epi8<64-2>(z1, z0),
        z1,
        _mm512_alignr512_epi8<2>(z2, z1)
    );
}
//1,4,6,4,1加算
template<bool avx512vnni>
static __forceinline __m512i smooth_5x5_horizontal(__m512i z0, __m512i z1, __m512i z2) {
    return smooth_5x5_vertical<avx512vnni>(
        _mm512_alignr512_epi8<64-4>(z1, z0),
        _mm512_alignr512_epi8<64-2>(z1, z0),
        z1,
        _mm512_alignr512_epi8<2>(z2, z1),
        _mm512_alignr512_epi8<4>(z2, z1)
    );
}
//1,6,15,20,15,6,1加算
static __forceinline __m512i smooth_7x7_horizontal(__m512i z0, __m512i z1, __m512i z2) {
    __m512i p0 = _mm512_alignr512_epi8<64-6>(z1, z0);
    __m512i p1 = _mm512_alignr512_epi8<64-4>(z1, z0);
    __m512i p2 = _mm512_alignr512_epi8<64-2>(z1, z0);
    __m512i p3 = z1;
    __m512i p4 = _mm512_alignr512_epi8<2>(z2, z1);
    __m512i p5 = _mm512_alignr512_epi8<4>(z2, z1);
    __m512i p6 = _mm512_alignr512_epi8<6>(z2, z1);
    // ###################
    //   !!!! 未実装 !!!!!
    // ###################
    return smooth_7x7_vertical(p0, p1, p2, p3, p4, p5, p6);
}

//rangeに応じてスムージング用の水平加算を行う
template<int range, bool avx512vnni>
static __forceinline __m512i smooth_horizontal(const __m512i &z0, const __m512i &z1, const __m512i &z2) {
    static_assert(0 < range && range <= 2, "range >= 3 not implemeted!");
    switch (range) {
    case 3: return smooth_7x7_horizontal(z0, z1, z2);
    case 2: return smooth_5x5_horizontal<avx512vnni>(z0, z1, z2);
    case 1:
    default:return smooth_3x3_horizontal(z0, z1, z2);
    }
}

//スムージングでは、まず水平方向の加算結果をバッファに格納していく
//この関数は1ラインぶんの水平方向の加算 + バッファへの格納のみを行う
template<int range, bool avx512vbmi, bool avx512vnni>
static __forceinline void smooth_fill_buffer_yc48(char *buf_ptr, const char *src_ptr, int x_start, int x_fin, int width, const __mmask32 &smooth_mask) {
    __m512i zY0, zU0, zV0;
    if (x_start == 0) {
        const PIXEL_YC *firstpix = (const PIXEL_YC *)src_ptr;
        zY0 = _mm512_broadcastw_epi16(_mm_loadu_si16(&firstpix->y));
        zU0 = _mm512_broadcastw_epi16(_mm_loadu_si16(&firstpix->cb));
        zV0 = _mm512_broadcastw_epi16(_mm_loadu_si16(&firstpix->cr));
    } else {
        src_ptr += x_start * sizeof(PIXEL_YC);
        afs_load_yc48<false, avx512vbmi>(zY0, zU0, zV0, src_ptr - 192);
    }
    //横方向のループ数は、AVX2(256bit)か128bitかによって異なる (logo_pitchとは異なる)
    const int x_fin_align = (((x_fin - x_start) + 31) & ~31) - 32;
    __m512i zY1, zU1, zV1;
    afs_load_yc48<false, avx512vbmi>(zY1, zU1, zV1, src_ptr);
    __m512i zY2, zU2, zV2;
    for (int x = x_fin_align; x; x -= 32, src_ptr += 192, buf_ptr += 192) {
        afs_load_yc48<false, avx512vbmi>(zY2, zU2, zV2, src_ptr + 192);
        _mm512_storeu_si512((__m512i *)(buf_ptr +   0), smooth_horizontal<range, avx512vnni>(zY0, zY1, zY2));
        _mm512_storeu_si512((__m512i *)(buf_ptr +  64), smooth_horizontal<range, avx512vnni>(zU0, zU1, zU2));
        _mm512_storeu_si512((__m512i *)(buf_ptr + 128), smooth_horizontal<range, avx512vnni>(zV0, zV1, zV2));
        zY0 = zY1; zY1 = zY2;
        zU0 = zU1; zU1 = zU2;
        zV0 = zV1; zV1 = zV2;
    }
    if (x_fin >= width) {
        const PIXEL_YC *lastpix = ((const PIXEL_YC *)src_ptr) + width - x_fin_align - x_start - 1;
        zY2 = _mm512_broadcastw_epi16(_mm_loadu_si16(&lastpix->y));
        zU2 = _mm512_broadcastw_epi16(_mm_loadu_si16(&lastpix->cb));
        zV2 = _mm512_broadcastw_epi16(_mm_loadu_si16(&lastpix->cr));
        zY1 = _mm512_mask_mov_epi16(zY2, smooth_mask, zY1);
        zU1 = _mm512_mask_mov_epi16(zU2, smooth_mask, zU1);
        zV1 = _mm512_mask_mov_epi16(zV2, smooth_mask, zV1);
    } else {
        afs_load_yc48<false, avx512vbmi>(zY2, zU2, zV2, src_ptr + 192);
    }
    _mm512_storeu_si512((__m512i *)(buf_ptr +   0), smooth_horizontal<range, avx512vnni>(zY0, zY1, zY2));
    _mm512_storeu_si512((__m512i *)(buf_ptr +  64), smooth_horizontal<range, avx512vnni>(zU0, zU1, zU2));
    _mm512_storeu_si512((__m512i *)(buf_ptr + 128), smooth_horizontal<range, avx512vnni>(zV0, zV1, zV2));
}

//バッファのライン数によるオフセットを計算する
#define BUF_LINE_OFFSET(x) ((((x) & (buf_line - 1)) * line_size) * sizeof(int16_t))

#pragma warning (push)
#pragma warning (disable:4127) //warning C4127: 条件式が定数です。
//yNewLineResultの最新のラインの水平加算結果と、バッファに格納済みの水平加算結果を用いて、
//縦方向の加算を行い、スムージング結果を16bit整数に格納して返す。
//yNewLineResultの値は、新たにバッファに格納される
template<unsigned int range, int buf_line, int line_size, bool avx512vnni>
static __forceinline void smooth_vertical(char *buf_ptr, __m512i& zResultY, __m512i &zResultU, __m512i &zResultV, int y) {
    __m512i zResultYOrg = zResultY;
    __m512i zResultUOrg = zResultU;
    __m512i zResultVOrg = zResultV;

    if (range == 1) {
        zResultY = smooth_3x3_vertical(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            zResultY
        );
        zResultU = smooth_3x3_vertical(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 64)),
            zResultU
        );
        zResultV = smooth_3x3_vertical(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 128)),
            zResultV
        );
    } else if (range == 2) {
        zResultY = smooth_5x5_vertical<avx512vnni>(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 0)),
            zResultY
        );
        zResultU = smooth_5x5_vertical<avx512vnni>(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 64)),
            zResultU
        );
        zResultV = smooth_5x5_vertical<avx512vnni>(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 128)),
            zResultV
        );
    } else if (range == 3) {
        zResultY = smooth_7x7_vertical(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 0)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 0)),
            zResultY
        );
        zResultU = smooth_7x7_vertical(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 64)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 64)),
            zResultU
        );
        zResultV = smooth_7x7_vertical(
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 128)),
            _mm512_loadu_si512((const __m512i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 128)),
            zResultV
        );
    }
    _mm512_storeu_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) +   0), zResultYOrg);
    _mm512_storeu_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) +  64), zResultUOrg);
    _mm512_storeu_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) + 128), zResultVOrg);
}
#pragma warning (pop)

//line_sizeはpixel数x3
template<int range, int line_size, bool avx512vbmi, bool avx512vnni>
void gaussHV_yc48_avx512_base(char *dst, int dst_pitch, const char *src, int src_pitch, int x_start, int x_fin, int y_start, int y_fin, int width, int height) {
    static_assert(0 < range && range <= 2, "0 < range && range <= 2");
    static_assert(((line_size/3) & ((line_size/3)-1)) == 0 && line_size % 3 == 0, "((line_size/3) & ((line_size/3)-1)) == 0 && line_size % 3 == 0");
    //最低限必要なバッファのライン数の決定、計算上2の乗数を使用する
    //最後の1ラインは一時的に重複して保持させる(必要な物を読んだところに上書きしていく)ため、
    //range4 (9x9)なら8ラインあればよい
    const int buf_line = (range >= 3 ? 8 : (range >= 2 ? 4 : 2));
    //水平方向の加算結果を保持するバッファ
    int16_t __declspec(align(64)) buffer[buf_line * line_size];
    memset(buffer, 0, sizeof(buffer));
    const __mmask32 smooth_mask = (((x_fin - x_start) & 31) == 0) ? 0xffffffff : 0xffffffff >> (32 - ((x_fin - x_start) & 31));
    const bool store_per_pix_on_edge = dst_pitch < ((x_start + (((x_fin - x_start) + 31) & ~31)) * (int)sizeof(PIXEL_YC));

    src += y_start * src_pitch;
    dst += y_start * dst_pitch;

    //バッファのrange*2-1行目までを埋める (range*2行目はメインループ内でロードする)
    for (int i = (y_start == 0) ? 0 : -range; i < range; i++)
        smooth_fill_buffer_yc48<range, avx512vbmi, avx512vnni>((char *)(buffer + (i + range) * line_size), src + i * src_pitch, x_start, x_fin, width, smooth_mask);

    if (y_start == 0) {
        //range行目と同じものでバッファの1行目～range行目まで埋める
        for (int i = 0; i < range; i++) {
            copy_bufline_avx512<line_size * sizeof(int16_t)>(buffer + i * line_size, buffer + range * line_size);
        }
    }

    //メインループ
    __m512i yDiff2Sum = _mm512_setzero_si512();
    int y = 0; //バッファのライン数のもととなるため、y=0で始めることは重要
    const int y_fin_loop = y_fin - y_start - ((y_fin >= height) ? range : 0); //水平加算用に先読みするため、rangeに配慮してループの終わりを決める
    for (; y < y_fin_loop; y++, dst += dst_pitch, src += src_pitch) {
        const char *src_ptr = src;
        char *dst_ptr = dst;
        char *buf_ptr = (char *)buffer;
        const int range_offset = range * src_pitch; //水平加算用に先読みする位置のオフセット YC48モードではsrc_pitchを使用する
        __m512i zY0, zU0, zV0;
        if (x_start == 0) {
            const PIXEL_YC *firstpix = (const PIXEL_YC *)(src_ptr + range_offset);
            zY0 = _mm512_broadcastw_epi16(_mm_loadu_si16(&firstpix->y));
            zU0 = _mm512_broadcastw_epi16(_mm_loadu_si16(&firstpix->cb));
            zV0 = _mm512_broadcastw_epi16(_mm_loadu_si16(&firstpix->cr));
        } else {
            src_ptr += x_start * sizeof(PIXEL_YC);
            dst_ptr += x_start * sizeof(PIXEL_YC);
            afs_load_yc48<false, avx512vbmi>(zY0, zU0, zV0, src_ptr + range_offset - 192);
        }
        __m512i zY1, zU1, zV1;
        afs_load_yc48<false, avx512vbmi>(zY1, zU1, zV1, src_ptr + range_offset);

        //横方向のループ数は、AVX2(256bit)か128bitかによって異なる (logo_pitchとは異なる)
        const int x_fin_align = (((x_fin - x_start) + 31) & ~31) - 32;
        for (int x = x_fin_align; x; x -= 32, src_ptr += 192, dst_ptr += 192, buf_ptr += 192) {
            __m512i zY2, zU2, zV2;
            afs_load_yc48<false, avx512vbmi>(zY2, zU2, zV2, src_ptr + range_offset + 192);
            //連続するデータz0, z1, z2を使って水平方向の加算を行う
            __m512i zResultY = smooth_horizontal<range, avx512vnni>(zY0, zY1, zY2);
            __m512i zResultU = smooth_horizontal<range, avx512vnni>(zU0, zU1, zU2);
            __m512i zResultV = smooth_horizontal<range, avx512vnni>(zV0, zV1, zV2);
            //zResultとバッファに格納されている水平方向の加算結果を合わせて
            //垂直方向の加算を行い、スムージングを完成させる
            //このループで得た水平加算結果はバッファに新たに格納される (不要になったものを上書き)
            smooth_vertical<range, buf_line, line_size, avx512vnni>(buf_ptr, zResultY, zResultU, zResultV, y);

            store_y_u_v_to_yc48<avx512vbmi>(dst_ptr, zResultY, zResultU, zResultV, false, 32);

            zY0 = zY1; zY1 = zY2;
            zU0 = zU1; zU1 = zU2;
            zV0 = zV1; zV1 = zV2;
        }

        __m512i zY2, zU2, zV2;
        if (x_fin >= width) {
            const PIXEL_YC *lastpix = ((const PIXEL_YC *)(src + range_offset)) + width - 1;
            zY2 = _mm512_broadcastw_epi16(_mm_loadu_si16(&lastpix->y));
            zU2 = _mm512_broadcastw_epi16(_mm_loadu_si16(&lastpix->cb));
            zV2 = _mm512_broadcastw_epi16(_mm_loadu_si16(&lastpix->cr));
            zY1 = _mm512_mask_mov_epi16(zY2, smooth_mask, zY1);
            zU1 = _mm512_mask_mov_epi16(zU2, smooth_mask, zU1);
            zV1 = _mm512_mask_mov_epi16(zV2, smooth_mask, zV1);
        } else {
            afs_load_yc48<false, avx512vbmi>(zY2, zU2, zV2, src_ptr + range_offset + 192);
        }

        __m512i zResultY = smooth_horizontal<range, avx512vnni>(zY0, zY1, zY2);
        __m512i zResultU = smooth_horizontal<range, avx512vnni>(zU0, zU1, zU2);
        __m512i zResultV = smooth_horizontal<range, avx512vnni>(zV0, zV1, zV2);
        smooth_vertical<range, buf_line, line_size, avx512vnni>(buf_ptr, zResultY, zResultU, zResultV, y);

        store_y_u_v_to_yc48<avx512vbmi>(dst_ptr, zResultY, zResultU, zResultV, store_per_pix_on_edge, (x_fin - x_start) & 31);
    }
    if (y_fin >= height) {
        //先読みできる分が終了したら、あとはバッファから読み込んで処理する
        //yとiの値に注意する
        for (int i = 1; i <= range; y++, i++, src += src_pitch, dst += dst_pitch) {
            const char *src_ptr = src + x_start * sizeof(PIXEL_YC);
            char *dst_ptr = dst + x_start * sizeof(PIXEL_YC);
            char *buf_ptr = (char *)buffer;
            const int x_fin_align = (((x_fin - x_start) + 31) & ~31) - 32;
            for (int x = x_fin_align; x; x -= 32, src_ptr += 192, dst_ptr += 192, buf_ptr += 192) {
                __m512i zResultY = _mm512_load_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 0));
                __m512i zResultU = _mm512_load_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 64));
                __m512i zResultV = _mm512_load_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 128));
                smooth_vertical<range, buf_line, line_size, avx512vnni>(buf_ptr, zResultY, zResultU, zResultV, y);

                store_y_u_v_to_yc48<avx512vbmi>(dst_ptr, zResultY, zResultU, zResultV, false, 32);
            }
            __m512i zResultY = _mm512_load_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 0));
            __m512i zResultU = _mm512_load_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 64));
            __m512i zResultV = _mm512_load_si512((__m512i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 128));
            smooth_vertical<range, buf_line, line_size, avx512vnni>(buf_ptr, zResultY, zResultU, zResultV, y);

            store_y_u_v_to_yc48<avx512vbmi>(dst_ptr, zResultY, zResultU, zResultV, store_per_pix_on_edge, (x_fin - x_start) & 31);
        }
    }
}

template<bool avx512vbmi, bool avx512vnni>
void gaussianHV_avx512_proc(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip = (FILTER_PROC_INFO *)param1;
    const int max_w = fpip->max_w;
    const int w = fpip->w;
    const int h = fpip->h;
    PIXEL_YC *ycp_buf = (PIXEL_YC *)fpip->ycp_edit;
    PIXEL_YC *ycp_dst = (PIXEL_YC *)param2;

    //ブロックサイズの決定
    const int BLOCK_SIZE_YCP = 256;
    const int max_block_size = BLOCK_SIZE_YCP;
    const int min_analyze_cycle = 64;
    const int scan_worker_x_limit_lower = std::min(thread_num, std::max(1, (w + BLOCK_SIZE_YCP - 1) / BLOCK_SIZE_YCP));
    const int scan_worker_x_limit_upper = std::max(1, w / 64);
    int scan_worker_x, scan_worker_y;
    for (int scan_worker_active = thread_num; ; scan_worker_active--) {
        for (scan_worker_x = scan_worker_x_limit_lower; scan_worker_x <= scan_worker_x_limit_upper; scan_worker_x++) {
            scan_worker_y = scan_worker_active / scan_worker_x;
            if (scan_worker_active - scan_worker_y * scan_worker_x == 0) {
                goto block_size_set; //二重ループを抜ける
            }
        }
    }
    block_size_set:
    if (thread_id >= scan_worker_x * scan_worker_y) {
        return;
    }
    int id_y = thread_id / scan_worker_x;
    int id_x = thread_id - id_y * scan_worker_x;
    int pos_y = ((int)(h * id_y / (double)scan_worker_y + 0.5)) & ~1;
    int y_fin = (id_y == scan_worker_y - 1) ? h : ((int)(h * (id_y+1) / (double)scan_worker_y + 0.5)) & ~1;
    int pos_x = ((int)(w * id_x / (double)scan_worker_x + 0.5) + (min_analyze_cycle -1)) & ~(min_analyze_cycle -1);
    int x_fin = (id_x == scan_worker_x - 1) ? w : ((int)(w * (id_x+1) / (double)scan_worker_x + 0.5) + (min_analyze_cycle -1)) & ~(min_analyze_cycle -1);
    if (pos_y == y_fin || pos_x == x_fin) {
        return; //念のため
    }
    int analyze_block = BLOCK_SIZE_YCP;
    if (id_x < scan_worker_x - 1) {
        for (; pos_x < x_fin; pos_x += analyze_block) {
            analyze_block = std::min(x_fin - pos_x, max_block_size);
            gaussHV_yc48_avx512_base<2, BLOCK_SIZE_YCP * 3, avx512vbmi, avx512vnni>((char *)ycp_dst, max_w * (int)sizeof(PIXEL_YC), (const char *)ycp_buf, max_w * (int)sizeof(PIXEL_YC), pos_x, pos_x + analyze_block, pos_y, y_fin, w, h);
        }
    } else {
        for (; x_fin - pos_x > max_block_size; pos_x += analyze_block) {
            analyze_block = std::min(x_fin - pos_x, max_block_size);
            gaussHV_yc48_avx512_base<2, BLOCK_SIZE_YCP * 3, avx512vbmi, avx512vnni>((char *)ycp_dst, max_w * (int)sizeof(PIXEL_YC), (const char *)ycp_buf, max_w * (int)sizeof(PIXEL_YC), pos_x, pos_x + analyze_block, pos_y, y_fin, w, h);
        }
        if (pos_x < w) {
            analyze_block = ((w - pos_x) + (min_analyze_cycle - 1)) & ~(min_analyze_cycle - 1);
            pos_x = w - analyze_block;
            gaussHV_yc48_avx512_base<2, BLOCK_SIZE_YCP * 3, avx512vbmi, avx512vnni>((char *)ycp_dst, max_w * (int)sizeof(PIXEL_YC), (const char *)ycp_buf, max_w * (int)sizeof(PIXEL_YC), pos_x, pos_x + analyze_block, pos_y, y_fin, w, h);
        }
    }
}

void gaussianHV_avx512(int thread_id, int thread_num, void *param1, void *param2) {
    gaussianHV_avx512_proc<false, false>(thread_id, thread_num, param1, param2);
}

void gaussianHV_avx512vbmivnni(int thread_id, int thread_num, void *param1, void *param2) {
    gaussianHV_avx512_proc<true, true>(thread_id, thread_num, param1, param2);
}

//---------------------------------------------------------------------
//        修正PDMマルチスレッド関数
//---------------------------------------------------------------------


#if USE_FMATH
__m512 __forceinline exp_ps512(__m512 z0) {
    static const uint32_t exptable[5] = {
        0x3f800000,
        0x3effff12,
        0x3e2aaa56,
        0x3d2b89cc,
        0x3c091331,
    };
#define BROADCAST512(ptr) _mm512_broadcastss_ps(_mm_castsi128_ps(_mm_cvtsi32_si128(*(int *)ptr)))
#define expCoeff(x) BROADCAST512(&exptable[x])
    static const float log2 = 0.6931471805599453f; //std::log(2.0f);
    static const float log2_e = 1.4426950408889634f; //1.0f / log2
    __m512 z1, z2;
    // xは - 87.3 <= x <= 88.72
    //static const float expMin = cvt(0xc2aeac50);
    //static const float expMax = cvt(0x42b17218);
    //z0 = _mm512_min_ps(z0, expMax);
    //z0 = _mm512_max_ps(z0, expMin);
    z0 = _mm512_mul_ps(z0, BROADCAST512(&log2_e));
    z1 = _mm512_roundscale_ps(z0, 0); // n = round(x)
    z0 = _mm512_sub_ps(z0, z1); // a
    z0 = _mm512_mul_ps(z0, BROADCAST512(&log2));
    z2 = expCoeff(4);
    z2 = _mm512_fmadd_ps(z2, z0, expCoeff(3));
    z2 = _mm512_fmadd_ps(z2, z0, expCoeff(2));
    z2 = _mm512_fmadd_ps(z2, z0, expCoeff(1));
    z2 = _mm512_fmadd_ps(z2, z0, expCoeff(0));
    z2 = _mm512_fmadd_ps(z2, z0, expCoeff(0));
    return _mm512_scalef_ps(z2, z1); // zm2 * 2^zm1
#undef expCoeff
#undef BROADCAST512
}
#endif //#if USE_FMATH


static __forceinline void getDiff(uint8_t *src, int max_w, __m512i& xUpper, __m512i& xLower, __m512i& xLeft, __m512i& xRight) {
    __m512i zSrc0, zSrc1;
    zSrc0 = _mm512_loadu_si512((__m128i *)(src - sizeof(PIXEL_YC) +  0));
    zSrc1 = _mm512_castsi128_si512(_mm_loadu_si128((__m128i *)(src - sizeof(PIXEL_YC) + 64)));

    __m512i zSrc = _mm512_alignr512_epi8<6>(zSrc1, zSrc0);

    xUpper = _mm512_sub_epi16(_mm512_loadu_si512((__m512i *)(src - max_w * sizeof(PIXEL_YC))), zSrc);
    xLower = _mm512_sub_epi16(_mm512_loadu_si512((__m512i *)(src + max_w * sizeof(PIXEL_YC))), zSrc);
    xLeft  = _mm512_sub_epi16(zSrc0, zSrc);
    xRight = _mm512_sub_epi16(_mm512_alignr512_epi8<12>(zSrc1, zSrc0), zSrc);
}


template <bool use_stream>
static __forceinline void pmd_mt_exp_avx512_line(uint8_t *dst, uint8_t *src, uint8_t *gau, int process_size_in_byte, int max_w, const __m512i &yPMDBufLimit, const int *pmdp) {
    uint8_t *src_fin = src + process_size_in_byte;

#if !USE_VPGATHER && !USE_FMATH
    __declspec(align(64)) int16_t diffBuf[64];
    __declspec(align(64)) int expBuf[64];
#endif

}

#pragma warning(push)
#pragma warning(disable: 4127) // C4127: 条件式が定数です。
template<int pmdc>
static __forceinline void lut_pmdp16(const int16_t *pmdp16, __m512i& z0, __m512i &z1, __m512i &z2, __m512i &z3) {
    if (pmdc <= 1) {
        __m512i zpmdp16_0 = _mm512_load_si512((__m512i *)(pmdp16 +  0));
        __m512i zpmdp16_1 = _mm512_load_si512((__m512i *)(pmdp16 + 32));
        z0 = _mm512_permutex2var_epi16(zpmdp16_0, z0, zpmdp16_1);
        z1 = _mm512_permutex2var_epi16(zpmdp16_0, z1, zpmdp16_1);
        z2 = _mm512_permutex2var_epi16(zpmdp16_0, z2, zpmdp16_1);
        z3 = _mm512_permutex2var_epi16(zpmdp16_0, z3, zpmdp16_1);
    } else if (pmdc <= 2) {
        __m512i zpmdp16_0 = _mm512_load_si512((__m512i *)(pmdp16 +  0));
        __m512i zpmdp16_1 = _mm512_load_si512((__m512i *)(pmdp16 + 32));
        __mmask32 m0 = _mm512_cmplt_epi16_mask(z0, _mm512_set1_epi16(64));
        __mmask32 m1 = _mm512_cmplt_epi16_mask(z1, _mm512_set1_epi16(64));
        __mmask32 m2 = _mm512_cmplt_epi16_mask(z2, _mm512_set1_epi16(64));
        __mmask32 m3 = _mm512_cmplt_epi16_mask(z3, _mm512_set1_epi16(64));
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z0, m0, zpmdp16_1);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z1, m1, zpmdp16_1);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z2, m2, zpmdp16_1);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z3, m3, zpmdp16_1);
        __m512i zpmdp16_2 = _mm512_load_si512((__m512i *)(pmdp16 + 64));
        __m512i zpmdp16_3 = _mm512_load_si512((__m512i *)(pmdp16 + 96));
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z0, _knot_mask32(m0), zpmdp16_3);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z1, _knot_mask32(m1), zpmdp16_3);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z2, _knot_mask32(m2), zpmdp16_3);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z3, _knot_mask32(m3), zpmdp16_3);
    } else if (pmdc <= 3) {
        __m512i zpmdp16_0 = _mm512_load_si512((__m512i *)(pmdp16 +  0));
        __m512i zpmdp16_1 = _mm512_load_si512((__m512i *)(pmdp16 + 32));
        __mmask32 m00 = _mm512_cmplt_epi16_mask(z0, _mm512_set1_epi16(64)); // x < 64
        __mmask32 m10 = _mm512_cmplt_epi16_mask(z1, _mm512_set1_epi16(64));
        __mmask32 m20 = _mm512_cmplt_epi16_mask(z2, _mm512_set1_epi16(64));
        __mmask32 m30 = _mm512_cmplt_epi16_mask(z3, _mm512_set1_epi16(64));
        __mmask32 m01 = _mm512_cmpge_epi16_mask(z0, _mm512_set1_epi16(128)); // x >= 128
        __mmask32 m11 = _mm512_cmpge_epi16_mask(z1, _mm512_set1_epi16(128));
        __mmask32 m21 = _mm512_cmpge_epi16_mask(z2, _mm512_set1_epi16(128));
        __mmask32 m31 = _mm512_cmpge_epi16_mask(z3, _mm512_set1_epi16(128));
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z0, m00, zpmdp16_1);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z1, m10, zpmdp16_1);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z2, m20, zpmdp16_1);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z3, m30, zpmdp16_1);
        __m512i zpmdp16_4 = _mm512_load_si512((__m512i *)(pmdp16 + 128));
        __m512i zpmdp16_5 = _mm512_load_si512((__m512i *)(pmdp16 + 160));
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z0, m01, zpmdp16_5);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z1, m11, zpmdp16_5);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z2, m21, zpmdp16_5);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z3, m31, zpmdp16_5);
        __m512i zpmdp16_2 = _mm512_load_si512((__m512i *)(pmdp16 + 64));
        __m512i zpmdp16_3 = _mm512_load_si512((__m512i *)(pmdp16 + 96));
        m01 = _kor_mask32(m01, m00);
        m11 = _kor_mask32(m11, m10);
        m21 = _kor_mask32(m21, m20);
        m31 = _kor_mask32(m31, m30);
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z0, _knot_mask32(m01), zpmdp16_3);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z1, _knot_mask32(m11), zpmdp16_3);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z2, _knot_mask32(m21), zpmdp16_3);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z3, _knot_mask32(m31), zpmdp16_3);
    } else if (pmdc <= 4) {
        __m512i zpmdp16_0 = _mm512_load_si512((__m512i *)(pmdp16 +  0));
        __m512i zpmdp16_1 = _mm512_load_si512((__m512i *)(pmdp16 + 32));
        __mmask32 m00 = _mm512_cmplt_epi16_mask(z0, _mm512_set1_epi16(64));
        __mmask32 m10 = _mm512_cmplt_epi16_mask(z1, _mm512_set1_epi16(64));
        __mmask32 m20 = _mm512_cmplt_epi16_mask(z2, _mm512_set1_epi16(64));
        __mmask32 m30 = _mm512_cmplt_epi16_mask(z3, _mm512_set1_epi16(64));
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z0, m00, zpmdp16_1);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z1, m10, zpmdp16_1);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z2, m20, zpmdp16_1);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_0, z3, m30, zpmdp16_1);
        __m512i zpmdp16_2 = _mm512_load_si512((__m512i *)(pmdp16 + 64));
        __m512i zpmdp16_3 = _mm512_load_si512((__m512i *)(pmdp16 + 96));
        __mmask32 m01 = _mm512_cmplt_epi16_mask(z0, _mm512_set1_epi16(128));
        __mmask32 m11 = _mm512_cmplt_epi16_mask(z1, _mm512_set1_epi16(128));
        __mmask32 m21 = _mm512_cmplt_epi16_mask(z2, _mm512_set1_epi16(128));
        __mmask32 m31 = _mm512_cmplt_epi16_mask(z3, _mm512_set1_epi16(128));
        m01 = _kandn_mask32(m00, m01);
        m11 = _kandn_mask32(m10, m11);
        m21 = _kandn_mask32(m20, m21);
        m31 = _kandn_mask32(m30, m31);
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z0, m01, zpmdp16_3);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z1, m11, zpmdp16_3);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z2, m21, zpmdp16_3);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_2, z3, m31, zpmdp16_3);
        __m512i zpmdp16_4 = _mm512_load_si512((__m512i *)(pmdp16 + 128));
        __m512i zpmdp16_5 = _mm512_load_si512((__m512i *)(pmdp16 + 160));
        m01 = _kor_mask32(m01, m00);
        m11 = _kor_mask32(m11, m10);
        m21 = _kor_mask32(m21, m20);
        m31 = _kor_mask32(m31, m30);
        __mmask32 m02 = _mm512_cmplt_epi16_mask(z0, _mm512_set1_epi16(192));
        __mmask32 m12 = _mm512_cmplt_epi16_mask(z1, _mm512_set1_epi16(192));
        __mmask32 m22 = _mm512_cmplt_epi16_mask(z2, _mm512_set1_epi16(192));
        __mmask32 m32 = _mm512_cmplt_epi16_mask(z3, _mm512_set1_epi16(192));
        m02 = _kandn_mask32(m01, m02);
        m12 = _kandn_mask32(m11, m12);
        m22 = _kandn_mask32(m21, m22);
        m32 = _kandn_mask32(m31, m32);
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z0, m02, zpmdp16_5);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z1, m12, zpmdp16_5);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z2, m22, zpmdp16_5);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_4, z3, m32, zpmdp16_5);
        __m512i zpmdp16_6 = _mm512_load_si512((__m512i *)(pmdp16 + 192));
        __m512i zpmdp16_7 = _mm512_load_si512((__m512i *)(pmdp16 + 224));
        m02 = _kor_mask32(m02, m01);
        m12 = _kor_mask32(m12, m11);
        m22 = _kor_mask32(m22, m21);
        m32 = _kor_mask32(m32, m31);
        z0 = _mm512_mask2_permutex2var_epi16(zpmdp16_6, z0, _knot_mask32(m02), zpmdp16_7);
        z1 = _mm512_mask2_permutex2var_epi16(zpmdp16_6, z1, _knot_mask32(m12), zpmdp16_7);
        z2 = _mm512_mask2_permutex2var_epi16(zpmdp16_6, z2, _knot_mask32(m22), zpmdp16_7);
        z3 = _mm512_mask2_permutex2var_epi16(zpmdp16_6, z3, _knot_mask32(m32), zpmdp16_7);
    }
}

template<bool avx512vnni, int pmdc>
static __forceinline void pmd_mt_exp_avx512_base(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    PIXEL_YC *gauss    = ((PMD_MT_PRM *)param2)->gauss;
    const int w = fpip->w;
    const int h = fpip->h;
    const int max_w = fpip->max_w;
    int y_start = h *  thread_id    / thread_num;
    int y_fin = h * (thread_id + 1) / thread_num;
#if USE_FMATH
    const int strength =  ((PMD_MT_PRM *)param2)->strength;
    const int threshold = ((PMD_MT_PRM *)param2)->threshold;

    const float range = 4.0f;
    const float threshold2 = pow(2.0f, threshold * 0.1f);
    const float strength2 = strength * 0.01f;
    //閾値の設定を変えた方が使いやすいです
    const float inv_threshold2 = (float)(1.0 / threshold2);

    __m512 yTempStrength2 = _mm512_set1_ps(strength2 * (1.0f / range));
    __m512 yMinusInvThreshold2 = _mm512_set1_ps(-1.0f * inv_threshold2);
#else
    int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;
#endif
    __declspec(align(64)) int16_t pmdp16[(pmdc > 4) ? 1 : pmdc*64];
    if (pmdc <= 4) {
        for (int i = 0; i < pmdc; i++) {
            __m256i y0 = _mm512_cvtepi32_epi16(_mm512_loadu_si512((__m512i *)(pmdp + i*64 +  0)));
            __m256i y1 = _mm512_cvtepi32_epi16(_mm512_loadu_si512((__m512i *)(pmdp + i*64 + 16)));
            __m256i y2 = _mm512_cvtepi32_epi16(_mm512_loadu_si512((__m512i *)(pmdp + i*64 + 32)));
            __m256i y3 = _mm512_cvtepi32_epi16(_mm512_loadu_si512((__m512i *)(pmdp + i*64 + 48)));
            _mm256_store_si256((__m256i *)(pmdp16 + i*64 +  0), y0);
            _mm256_store_si256((__m256i *)(pmdp16 + i*64 + 16), y1);
            _mm256_store_si256((__m256i *)(pmdp16 + i*64 + 32), y2);
            _mm256_store_si256((__m256i *)(pmdp16 + i*64 + 48), y3);
        }
    }

    //最初の行はそのままコピー
    if (0 == y_start) {
        memcpy_avx512<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
        y_start++;
    }
    //最後の行はそのままコピー
    y_fin -= (h == y_fin);

#if !USE_VPGATHER && !USE_FMATH
    __declspec(align(64)) int16_t diffBuf[64];
    __declspec(align(64)) int expBuf[64];
#endif

    uint8_t *src_line = (uint8_t *)(fpip->ycp_edit + y_start * max_w);
    uint8_t *dst_line = (uint8_t *)(fpip->ycp_temp + y_start * max_w);
    uint8_t *gau_line = (uint8_t *)(gauss          + y_start * max_w);

    __m512i yPMDBufLimit = _mm512_set1_epi16((pmdc > 4) ? (PMD_TABLE_SIZE-1) : (pmdc*64-1));

    for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC), gau_line += max_w * sizeof(PIXEL_YC)) {
        uint8_t *src = src_line;
        uint8_t *dst = dst_line;
        uint8_t *gau = gau_line;

        //まずは、先端終端ピクセルを気にせず普通に処理してしまう
        //先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
        //最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
        //先端終端ピクセルは後から上書きコピーする
        uint8_t *src_fin = src + w * sizeof(PIXEL_YC);;
        for ( ; src < src_fin; src += 64, dst += 64, gau += 64) {
            __m512i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
            __m512i yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff;
            getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);
            getDiff(gau, max_w, yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff);
    #if USE_FMATH
            __m512 yGUpperlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauUpperDiff));
            __m512 yGUpperhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauUpperDiff));
            __m512 yGLowerlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauLowerDiff));
            __m512 yGLowerhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauLowerDiff));
            __m512 yGLeftlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauLeftDiff));
            __m512 yGLefthi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauLeftDiff));
            __m512 yGRightlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauRightDiff));
            __m512 yGRighthi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauRightDiff));

            yGUpperlo = _mm512_mul_ps(yGUpperlo, yGUpperlo);
            yGUpperhi = _mm512_mul_ps(yGUpperhi, yGUpperhi);
            yGLowerlo = _mm512_mul_ps(yGLowerlo, yGLowerlo);
            yGLowerhi = _mm512_mul_ps(yGLowerhi, yGLowerhi);
            yGLeftlo = _mm512_mul_ps(yGLeftlo, yGLeftlo);
            yGLefthi = _mm512_mul_ps(yGLefthi, yGLefthi);
            yGRightlo = _mm512_mul_ps(yGRightlo, yGRightlo);
            yGRighthi = _mm512_mul_ps(yGRighthi, yGRighthi);

            yGUpperlo = _mm512_mul_ps(yGUpperlo, yMinusInvThreshold2);
            yGUpperhi = _mm512_mul_ps(yGUpperhi, yMinusInvThreshold2);
            yGLowerlo = _mm512_mul_ps(yGLowerlo, yMinusInvThreshold2);
            yGLowerhi = _mm512_mul_ps(yGLowerhi, yMinusInvThreshold2);
            yGLeftlo = _mm512_mul_ps(yGLeftlo, yMinusInvThreshold2);
            yGLefthi = _mm512_mul_ps(yGLefthi, yMinusInvThreshold2);
            yGRightlo = _mm512_mul_ps(yGRightlo, yMinusInvThreshold2);
            yGRighthi = _mm512_mul_ps(yGRighthi, yMinusInvThreshold2);

            yGUpperlo = exp_ps512(yGUpperlo);
            yGUpperhi = exp_ps512(yGUpperhi);
            yGLowerlo = exp_ps512(yGLowerlo);
            yGLowerhi = exp_ps512(yGLowerhi);
            yGLeftlo = exp_ps512(yGLeftlo);
            yGLefthi = exp_ps512(yGLefthi);
            yGRightlo = exp_ps512(yGRightlo);
            yGRighthi = exp_ps512(yGRighthi);

            yGUpperlo = _mm512_mul_ps(yGUpperlo, yTempStrength2);
            yGUpperhi = _mm512_mul_ps(yGUpperhi, yTempStrength2);
            yGLowerlo = _mm512_mul_ps(yGLowerlo, yTempStrength2);
            yGLowerhi = _mm512_mul_ps(yGLowerhi, yTempStrength2);
            yGLeftlo = _mm512_mul_ps(yGLeftlo, yTempStrength2);
            yGLefthi = _mm512_mul_ps(yGLefthi, yTempStrength2);
            yGRightlo = _mm512_mul_ps(yGRightlo, yTempStrength2);
            yGRighthi = _mm512_mul_ps(yGRighthi, yTempStrength2);

            __m512 yAddLo0, yAddHi0, yAddLo1, yAddHi1;
            yGUpperlo = _mm512_mul_ps(yGUpperlo, _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcUpperDiff)));
            yGUpperhi = _mm512_mul_ps(yGUpperhi, _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcUpperDiff)));
            yGLeftlo = _mm512_mul_ps(yGLeftlo, _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLeftDiff)));
            yGLefthi = _mm512_mul_ps(yGLefthi, _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLeftDiff)));

            yAddLo0 = _mm512_fmadd_ps(yGLowerlo, _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLowerDiff)), yGUpperlo);
            yAddHi0 = _mm512_fmadd_ps(yGLowerhi, _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLowerDiff)), yGUpperhi);
            yAddLo1 = _mm512_fmadd_ps(yGRightlo, _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcRightDiff)), yGLeftlo);
            yAddHi1 = _mm512_fmadd_ps(yGRighthi, _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcRightDiff)), yGLefthi);

            yAddLo0 = _mm512_add_ps(yAddLo0, yAddLo1);
            yAddHi0 = _mm512_add_ps(yAddHi0, yAddHi1);

            __m512i ySrc = _mm512_loadu_si512((__m512i *)(src));
            _mm512_storeu_si512((__m512i *)(dst), _mm512_add_epi16(ySrc, _mm512_packs_epi32(_mm512_cvtps_epi32(yAddLo0), _mm512_cvtps_epi32(yAddHi0))));
    #else
            yGauUpperDiff = _mm512_abs_epi16(yGauUpperDiff);
            yGauLowerDiff = _mm512_abs_epi16(yGauLowerDiff);
            yGauLeftDiff  = _mm512_abs_epi16(yGauLeftDiff);
            yGauRightDiff = _mm512_abs_epi16(yGauRightDiff);

            yGauUpperDiff = _mm512_min_epi16(yGauUpperDiff, yPMDBufLimit);
            yGauLowerDiff = _mm512_min_epi16(yGauLowerDiff, yPMDBufLimit);
            yGauLeftDiff  = _mm512_min_epi16(yGauLeftDiff,  yPMDBufLimit);
            yGauRightDiff = _mm512_min_epi16(yGauRightDiff, yPMDBufLimit);

            __m512i yAddLo, yAddHi;
            if (pmdc <= 4) {
                lut_pmdp16<pmdc>(pmdp16, yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff);

                __m512i yELULo = _mm512_unpacklo_epi16(yGauLowerDiff, yGauUpperDiff);
                __m512i yELUHi = _mm512_unpackhi_epi16(yGauLowerDiff, yGauUpperDiff);
                __m512i yERLLo = _mm512_unpacklo_epi16(yGauRightDiff, yGauLeftDiff);
                __m512i yERLHi = _mm512_unpackhi_epi16(yGauRightDiff, yGauLeftDiff);

                yAddLo = _mm512_madd_epi16(yELULo, _mm512_unpacklo_epi16(ySrcLowerDiff, ySrcUpperDiff));
                yAddHi = _mm512_madd_epi16(yELUHi, _mm512_unpackhi_epi16(ySrcLowerDiff, ySrcUpperDiff));
                if (avx512vnni) {
                    yAddLo = _mm512_dpwssd_epi32(yAddLo, yERLLo, _mm512_unpacklo_epi16(ySrcRightDiff, ySrcLeftDiff));
                    yAddHi = _mm512_dpwssd_epi32(yAddHi, yERLHi, _mm512_unpackhi_epi16(ySrcRightDiff, ySrcLeftDiff));
                } else {
                    __m512i yAddLo1 = _mm512_madd_epi16(yERLLo, _mm512_unpacklo_epi16(ySrcRightDiff, ySrcLeftDiff));
                    __m512i yAddHi1 = _mm512_madd_epi16(yERLHi, _mm512_unpackhi_epi16(ySrcRightDiff, ySrcLeftDiff));
                    yAddLo = _mm512_add_epi32(yAddLo, yAddLo1);
                    yAddHi = _mm512_add_epi32(yAddHi, yAddHi1);
                }
            } else {
#if USE_VPGATHER

                __m512i yEUpperlo = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(yGauUpperDiff), pmdp, 4);
                __m512i yEUpperhi = _mm512_i32gather_epi32(cvthi512_epi16_epi32(yGauUpperDiff), pmdp, 4);
                __m512i yELowerlo = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(yGauLowerDiff), pmdp, 4);
                __m512i yELowerhi = _mm512_i32gather_epi32(cvthi512_epi16_epi32(yGauLowerDiff), pmdp, 4);
                __m512i yELeftlo = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(yGauLeftDiff), pmdp, 4);
                __m512i yELefthi = _mm512_i32gather_epi32(cvthi512_epi16_epi32(yGauLeftDiff), pmdp, 4);
                __m512i yERightlo = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(yGauRightDiff), pmdp, 4);
                __m512i yERighthi = _mm512_i32gather_epi32(cvthi512_epi16_epi32(yGauRightDiff), pmdp, 4);
#else
                _mm512_store_si512((__m512i *)(diffBuf + 0), yGauUpperDiff);
                _mm512_store_si512((__m512i *)(diffBuf + 32), yGauLowerDiff);
                _mm512_store_si512((__m512i *)(diffBuf + 64), yGauLeftDiff);
                _mm512_store_si512((__m512i *)(diffBuf + 96), yGauRightDiff);

                for (int i = 0; i < _countof(expBuf); i += 16) {
                    expBuf[i + 0] = pmdp[diffBuf[i + 0]];
                    expBuf[i + 1] = pmdp[diffBuf[i + 1]];
                    expBuf[i + 2] = pmdp[diffBuf[i + 2]];
                    expBuf[i + 3] = pmdp[diffBuf[i + 3]];
                    expBuf[i + 4] = pmdp[diffBuf[i + 8]];
                    expBuf[i + 5] = pmdp[diffBuf[i + 9]];
                    expBuf[i + 6] = pmdp[diffBuf[i + 10]];
                    expBuf[i + 7] = pmdp[diffBuf[i + 11]];
                    expBuf[i + 8] = pmdp[diffBuf[i + 4]];
                    expBuf[i + 9] = pmdp[diffBuf[i + 5]];
                    expBuf[i + 10] = pmdp[diffBuf[i + 6]];
                    expBuf[i + 11] = pmdp[diffBuf[i + 7]];
                    expBuf[i + 12] = pmdp[diffBuf[i + 12]];
                    expBuf[i + 13] = pmdp[diffBuf[i + 13]];
                    expBuf[i + 14] = pmdp[diffBuf[i + 14]];
                    expBuf[i + 15] = pmdp[diffBuf[i + 15]];
                }

                __m512i yEUpperlo = _mm512_load_si512((__m512i *)(expBuf + 0));
                __m512i yEUpperhi = _mm512_load_si512((__m512i *)(expBuf + 16));
                __m512i yELowerlo = _mm512_load_si512((__m512i *)(expBuf + 32));
                __m512i yELowerhi = _mm512_load_si512((__m512i *)(expBuf + 48));
                __m512i yELeftlo = _mm512_load_si512((__m512i *)(expBuf + 64));
                __m512i yELefthi = _mm512_load_si512((__m512i *)(expBuf + 80));
                __m512i yERightlo = _mm512_load_si512((__m512i *)(expBuf + 96));
                __m512i yERighthi = _mm512_load_si512((__m512i *)(expBuf + 112));
#endif
#if 1 //こちらの積算の少ないほうが高速
                __mmask32 maskUpper = 0xAAAAAAAA;
                __m512i yELULo = _mm512_mask_mov_epi16(yELowerlo, maskUpper, _mm512_slli_epi32(yEUpperlo, 16));
                __m512i yELUHi = _mm512_mask_mov_epi16(yELowerhi, maskUpper, _mm512_slli_epi32(yEUpperhi, 16));
                __m512i yERLLo = _mm512_mask_mov_epi16(yERightlo, maskUpper, _mm512_slli_epi32(yELeftlo, 16));
                __m512i yERLHi = _mm512_mask_mov_epi16(yERighthi, maskUpper, _mm512_slli_epi32(yELefthi, 16));

                yAddLo = _mm512_madd_epi16(yELULo, _mm512_unpacklo_epi16(ySrcLowerDiff, ySrcUpperDiff));
                yAddHi = _mm512_madd_epi16(yELUHi, _mm512_unpackhi_epi16(ySrcLowerDiff, ySrcUpperDiff));
                if (avx512vnni) {
                    yAddLo = _mm512_dpwssd_epi32(yAddLo, yERLLo, _mm512_unpacklo_epi16(ySrcRightDiff, ySrcLeftDiff));
                    yAddHi = _mm512_dpwssd_epi32(yAddHi, yERLHi, _mm512_unpackhi_epi16(ySrcRightDiff, ySrcLeftDiff));
                } else {
                    __m512i yAddLo1 = _mm512_madd_epi16(yERLLo, _mm512_unpacklo_epi16(ySrcRightDiff, ySrcLeftDiff));
                    __m512i yAddHi1 = _mm512_madd_epi16(yERLHi, _mm512_unpackhi_epi16(ySrcRightDiff, ySrcLeftDiff));
                    yAddLo = _mm512_add_epi32(yAddLo, yAddLo1);
                    yAddHi = _mm512_add_epi32(yAddHi, yAddHi1);
                }
    #else
                yEUpperlo = _mm512_mullo_epi32(yEUpperlo, cvtlo512_epi16_epi32(ySrcUpperDiff));
                yEUpperhi = _mm512_mullo_epi32(yEUpperhi, cvthi512_epi16_epi32(ySrcUpperDiff));
                yELowerlo = _mm512_mullo_epi32(yELowerlo, cvtlo512_epi16_epi32(ySrcLowerDiff));
                yELowerhi = _mm512_mullo_epi32(yELowerhi, cvthi512_epi16_epi32(ySrcLowerDiff));
                yELeftlo = _mm512_mullo_epi32(yELeftlo, cvtlo512_epi16_epi32(ySrcLeftDiff));
                yELefthi = _mm512_mullo_epi32(yELefthi, cvthi512_epi16_epi32(ySrcLeftDiff));
                yERightlo = _mm512_mullo_epi32(yERightlo, cvtlo512_epi16_epi32(ySrcRightDiff));
                yERighthi = _mm512_mullo_epi32(yERighthi, cvthi512_epi16_epi32(ySrcRightDiff));

                yAddLo = yEUpperlo;
                yAddHi = yEUpperhi;
                yAddLo = _mm512_add_epi32(yAddLo, yELowerlo);
                yAddHi = _mm512_add_epi32(yAddHi, yELowerhi);
                yAddLo = _mm512_add_epi32(yAddLo, yELeftlo);
                yAddHi = _mm512_add_epi32(yAddHi, yELefthi);
                yAddLo = _mm512_add_epi32(yAddLo, yERightlo);
                yAddHi = _mm512_add_epi32(yAddHi, yERighthi);
#endif
            }

            __m512i ySrc = _mm512_loadu_si512((__m512i *)(src));
            _mm512_storeu_si512((__m512i *)(dst), _mm512_add_epi16(ySrc, _mm512_packs_epi32(_mm512_srai_epi32(yAddLo, 16), _mm512_srai_epi32(yAddHi, 16))));
    #endif
        }

        //先端と終端をそのままコピー
        *(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
        *(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
    }
    //最後の行はそのままコピー
    if (h-1 == y_fin) {
        memcpy_avx512<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
    }
    _mm256_zeroupper();
}
#pragma warning(pop)

static __forceinline void anisotropic_mt_exp_avx512_base(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    const int w = fpip->w;
    const int h = fpip->h;
    const int max_w = fpip->max_w;
    int y_start = h *  thread_id    / thread_num;
    int y_fin   = h * (thread_id+1) / thread_num;

    //最初の行はそのままコピー
    if (0 == y_start) {
        memcpy_avx512<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
        y_start++;
    }
    //最後の行はそのままコピー
    y_fin -= (h == y_fin);

    uint8_t *src_line = (uint8_t *)(fpip->ycp_edit + y_start * max_w);
    uint8_t *dst_line = (uint8_t *)(fpip->ycp_temp + y_start * max_w);
#if USE_FMATH
    const int strength =  ((PMD_MT_PRM *)param2)->strength;
    const int threshold = ((PMD_MT_PRM *)param2)->threshold;

    const float range = 4.0f;
    const float strength2 = strength/100.0f;
    //閾値の設定を変えた方が使いやすいです
    const float inv_threshold2 = (float)(1.0 / (threshold*16/10.0*threshold*16/10.0));

    // = (1.0 / range) * (   (1.0/ (1.0 + (  x*x / threshold2 )) )  * strength2 )
    // = (1.0 / range) * (   (1.0/ (1.0 + (  x*x * inv_threshold2 )) )  * strength2 )

    __m512 yMinusInvThreshold2 = _mm512_set1_ps(-1.0f * inv_threshold2);
    __m512 zStrength2 = _mm512_set1_ps(strength2 / range);
#else
    int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;
#endif

#if !USE_VPGATHER && !USE_FMATH
    __declspec(align(32)) int16_t diffBuf[64];
    __declspec(align(32)) int expBuf[64];
#endif
    __m512i yPMDBufMaxLimit = _mm512_set1_epi16(PMD_TABLE_SIZE-1);
    __m512i yPMDBufMinLimit = _mm512_set1_epi16(-PMD_TABLE_SIZE);

    for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC)) {
        uint8_t *src = src_line;
        uint8_t *dst = dst_line;

        //まずは、先端終端ピクセルを気にせず普通に処理してしまう
        //先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
        //最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
        //先端終端ピクセルは後から上書きコピーする
        uint8_t *src_fin = src + w * sizeof(PIXEL_YC);
        for ( ; src < src_fin; src += 64, dst += 64) {
            __m512i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
            getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);
#if USE_FMATH
            __m512 ySUpperlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcUpperDiff));
            __m512 ySUpperhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcUpperDiff));
            __m512 ySLowerlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLowerDiff));
            __m512 ySLowerhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLowerDiff));
            __m512 ySLeftlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLeftDiff));
            __m512 ySLefthi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLeftDiff));
            __m512 ySRightlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcRightDiff));
            __m512 ySRighthi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcRightDiff));

            __m512 yTUpperlo = _mm512_mul_ps(ySUpperlo, ySUpperlo);
            __m512 yTUpperhi = _mm512_mul_ps(ySUpperhi, ySUpperhi);
            __m512 yTLowerlo = _mm512_mul_ps(ySLowerlo, ySLowerlo);
            __m512 yTLowerhi = _mm512_mul_ps(ySLowerhi, ySLowerhi);
            __m512 yTLeftlo = _mm512_mul_ps(ySLeftlo, ySLeftlo);
            __m512 yTLefthi = _mm512_mul_ps(ySLefthi, ySLefthi);
            __m512 yTRightlo = _mm512_mul_ps(ySRightlo, ySRightlo);
            __m512 yTRighthi = _mm512_mul_ps(ySRighthi, ySRighthi);

            yTUpperlo = _mm512_mul_ps(ySUpperlo, yMinusInvThreshold2);
            yTUpperhi = _mm512_mul_ps(ySUpperhi, yMinusInvThreshold2);
            yTLowerlo = _mm512_mul_ps(ySLowerlo, yMinusInvThreshold2);
            yTLowerhi = _mm512_mul_ps(ySLowerhi, yMinusInvThreshold2);
            yTLeftlo = _mm512_mul_ps(ySLeftlo, yMinusInvThreshold2);
            yTLefthi = _mm512_mul_ps(ySLefthi, yMinusInvThreshold2);
            yTRightlo = _mm512_mul_ps(ySRightlo, yMinusInvThreshold2);
            yTRighthi = _mm512_mul_ps(ySRighthi, yMinusInvThreshold2);

            yTUpperlo = exp_ps512(yTUpperlo);
            yTUpperhi = exp_ps512(yTUpperhi);
            yTLowerlo = exp_ps512(yTLowerlo);
            yTLowerhi = exp_ps512(yTLowerhi);
            yTLeftlo = exp_ps512(yTLeftlo);
            yTLefthi = exp_ps512(yTLefthi);
            yTRightlo = exp_ps512(yTRightlo);
            yTRighthi = exp_ps512(yTRighthi);

            yTUpperlo = _mm512_mul_ps(zStrength2, yTUpperlo);
            yTUpperhi = _mm512_mul_ps(zStrength2, yTUpperhi);
            yTLowerlo = _mm512_mul_ps(zStrength2, yTLowerlo);
            yTLowerhi = _mm512_mul_ps(zStrength2, yTLowerhi);
            yTLeftlo = _mm512_mul_ps(zStrength2, yTLeftlo);
            yTLefthi = _mm512_mul_ps(zStrength2, yTLefthi);
            yTRightlo = _mm512_mul_ps(zStrength2, yTRightlo);
            yTRighthi = _mm512_mul_ps(zStrength2, yTRighthi);

            __m512 yAddLo0, yAddHi0, yAddLo1, yAddHi1;
            yAddLo0 = _mm512_fmadd_ps(ySLowerlo, yTLowerlo, _mm512_mul_ps(ySUpperlo, yTUpperlo));
            yAddHi0 = _mm512_fmadd_ps(ySLowerhi, yTLowerhi, _mm512_mul_ps(ySUpperhi, yTUpperhi));
            yAddLo1 = _mm512_fmadd_ps(ySRightlo, yTRightlo, _mm512_mul_ps(ySLeftlo, yTLeftlo));
            yAddHi1 = _mm512_fmadd_ps(ySRighthi, yTRighthi, _mm512_mul_ps(ySLefthi, yTLefthi));

            yAddLo0 = _mm512_add_ps(yAddLo0, yAddLo1);
            yAddHi0 = _mm512_add_ps(yAddHi0, yAddHi1);

            __m512i ySrc = _mm512_loadu_si512((__m512i *)(src));
            _mm512_storeu_si512((__m512i *)(dst), _mm512_add_epi16(ySrc, _mm512_packs_epi32(_mm512_cvtps_epi32(yAddLo0), _mm512_cvtps_epi32(yAddHi0))));
#else
            ySrcUpperDiff = _mm512_max_epi16(ySrcUpperDiff, yPMDBufMinLimit);
            ySrcLowerDiff = _mm512_max_epi16(ySrcLowerDiff, yPMDBufMinLimit);
            ySrcLeftDiff  = _mm512_max_epi16(ySrcLeftDiff,  yPMDBufMinLimit);
            ySrcRightDiff = _mm512_max_epi16(ySrcRightDiff, yPMDBufMinLimit);

            ySrcUpperDiff = _mm512_min_epi16(ySrcUpperDiff, yPMDBufMaxLimit);
            ySrcLowerDiff = _mm512_min_epi16(ySrcLowerDiff, yPMDBufMaxLimit);
            ySrcLeftDiff  = _mm512_min_epi16(ySrcLeftDiff,  yPMDBufMaxLimit);
            ySrcRightDiff = _mm512_min_epi16(ySrcRightDiff, yPMDBufMaxLimit);
#if USE_VPGATHER
            __m512i yEUpperlo = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(ySrcUpperDiff), pmdp, 4);
            __m512i yEUpperhi = _mm512_i32gather_epi32(cvthi512_epi16_epi32(ySrcUpperDiff), pmdp, 4);
            __m512i yELowerlo = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(ySrcLowerDiff), pmdp, 4);
            __m512i yELowerhi = _mm512_i32gather_epi32(cvthi512_epi16_epi32(ySrcLowerDiff), pmdp, 4);
            __m512i yELeftlo  = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(ySrcLeftDiff),  pmdp, 4);
            __m512i yELefthi  = _mm512_i32gather_epi32(cvthi512_epi16_epi32(ySrcLeftDiff),  pmdp, 4);
            __m512i yERightlo = _mm512_i32gather_epi32(cvtlo512_epi16_epi32(ySrcRightDiff), pmdp, 4);
            __m512i yERighthi = _mm512_i32gather_epi32(cvthi512_epi16_epi32(ySrcRightDiff), pmdp, 4);
#else
            _mm512_store_si512((__m512i *)(diffBuf +  0), ySrcUpperDiff);
            _mm512_store_si512((__m512i *)(diffBuf + 32), ySrcLowerDiff);
            _mm512_store_si512((__m512i *)(diffBuf + 64), ySrcLeftDiff);
            _mm512_store_si512((__m512i *)(diffBuf + 96), ySrcRightDiff);

            for (int i = 0; i < _countof(expBuf); i += 16) {
                expBuf[i+ 0] = pmdp[diffBuf[i+ 0]];
                expBuf[i+ 1] = pmdp[diffBuf[i+ 1]];
                expBuf[i+ 2] = pmdp[diffBuf[i+ 2]];
                expBuf[i+ 3] = pmdp[diffBuf[i+ 3]];
                expBuf[i+ 4] = pmdp[diffBuf[i+ 8]];
                expBuf[i+ 5] = pmdp[diffBuf[i+ 9]];
                expBuf[i+ 6] = pmdp[diffBuf[i+10]];
                expBuf[i+ 7] = pmdp[diffBuf[i+11]];
                expBuf[i+ 8] = pmdp[diffBuf[i+ 4]];
                expBuf[i+ 9] = pmdp[diffBuf[i+ 5]];
                expBuf[i+10] = pmdp[diffBuf[i+ 6]];
                expBuf[i+11] = pmdp[diffBuf[i+ 7]];
                expBuf[i+12] = pmdp[diffBuf[i+12]];
                expBuf[i+13] = pmdp[diffBuf[i+13]];
                expBuf[i+14] = pmdp[diffBuf[i+14]];
                expBuf[i+15] = pmdp[diffBuf[i+15]];
            }

            __m512i yEUpperlo = _mm512_load_si512((__m512i *)(expBuf +   0));
            __m512i yEUpperhi = _mm512_load_si512((__m512i *)(expBuf +  16));
            __m512i yELowerlo = _mm512_load_si512((__m512i *)(expBuf +  32));
            __m512i yELowerhi = _mm512_load_si512((__m512i *)(expBuf +  48));
            __m512i yELeftlo  = _mm512_load_si512((__m512i *)(expBuf +  64));
            __m512i yELefthi  = _mm512_load_si512((__m512i *)(expBuf +  80));
            __m512i yERightlo = _mm512_load_si512((__m512i *)(expBuf +  96));
            __m512i yERighthi = _mm512_load_si512((__m512i *)(expBuf + 112));
#endif
            __m512i yAddLo, yAddHi;
            yAddLo = yEUpperlo;
            yAddHi = yEUpperhi;
            yAddLo = _mm512_add_epi32(yAddLo, yELowerlo);
            yAddHi = _mm512_add_epi32(yAddHi, yELowerhi);
            yAddLo = _mm512_add_epi32(yAddLo, yELeftlo);
            yAddHi = _mm512_add_epi32(yAddHi, yELefthi);
            yAddLo = _mm512_add_epi32(yAddLo, yERightlo);
            yAddHi = _mm512_add_epi32(yAddHi, yERighthi);

            __m512i ySrc = _mm512_loadu_si512((__m512i *)(src));
            _mm512_storeu_si512((__m512i *)(dst), _mm512_add_epi16(ySrc, _mm512_packs_epi32(yAddLo, yAddHi)));
#endif
        }
        //先端と終端をそのままコピー
        *(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
        *(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
    }
    //最後の行はそのままコピー
    if (h-1 == y_fin) {
        memcpy_avx512<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
    }
    _mm256_zeroupper();
}

template <bool use_stream>
static __forceinline void pmd_mt_avx512_line(uint8_t *dst, uint8_t *src, uint8_t *gau, int process_size_in_byte, int max_w, const __m512& zInvThreshold2, const __m512& zStrength2, const __m512& zOnef) {
    uint8_t *src_fin = src + process_size_in_byte;
    for (; src < src_fin; src += 64, dst += 64, gau += 64) {
        __m512i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
        __m512i yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff;
        getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);
        getDiff(gau, max_w, yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff);

        __m512 yGUpperlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauUpperDiff));
        __m512 yGUpperhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauUpperDiff));
        __m512 yGLowerlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauLowerDiff));
        __m512 yGLowerhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauLowerDiff));
        __m512 yGLeftlo  = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauLeftDiff));
        __m512 yGLefthi  = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauLeftDiff));
        __m512 yGRightlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(yGauRightDiff));
        __m512 yGRighthi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(yGauRightDiff));

        yGUpperlo = _mm512_mul_ps(yGUpperlo, yGUpperlo);
        yGUpperhi = _mm512_mul_ps(yGUpperhi, yGUpperhi);
        yGLowerlo = _mm512_mul_ps(yGLowerlo, yGLowerlo);
        yGLowerhi = _mm512_mul_ps(yGLowerhi, yGLowerhi);
        yGLeftlo  = _mm512_mul_ps(yGLeftlo,  yGLeftlo);
        yGLefthi  = _mm512_mul_ps(yGLefthi,  yGLefthi);
        yGRightlo = _mm512_mul_ps(yGRightlo, yGRightlo);
        yGRighthi = _mm512_mul_ps(yGRighthi, yGRighthi);

        yGUpperlo = _mm512_fmadd_ps(yGUpperlo, zInvThreshold2, zOnef);
        yGUpperhi = _mm512_fmadd_ps(yGUpperhi, zInvThreshold2, zOnef);
        yGLowerlo = _mm512_fmadd_ps(yGLowerlo, zInvThreshold2, zOnef);
        yGLowerhi = _mm512_fmadd_ps(yGLowerhi, zInvThreshold2, zOnef);
        yGLeftlo  = _mm512_fmadd_ps(yGLeftlo,  zInvThreshold2, zOnef);
        yGLefthi  = _mm512_fmadd_ps(yGLefthi,  zInvThreshold2, zOnef);
        yGRightlo = _mm512_fmadd_ps(yGRightlo, zInvThreshold2, zOnef);
        yGRighthi = _mm512_fmadd_ps(yGRighthi, zInvThreshold2, zOnef);

#if 1
        yGUpperlo = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGUpperlo));
        yGUpperhi = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGUpperhi));
        yGLowerlo = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGLowerlo));
        yGLowerhi = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGLowerhi));
        yGLeftlo  = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGLeftlo));
        yGLefthi  = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGLefthi));
        yGRightlo = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGRightlo));
        yGRighthi = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yGRighthi));
#else
        yGUpperlo = _mm512_div_ps(zStrength2, yGUpperlo);
        yGUpperhi = _mm512_div_ps(zStrength2, yGUpperhi);
        yGLowerlo = _mm512_div_ps(zStrength2, yGLowerlo);
        yGLowerhi = _mm512_div_ps(zStrength2, yGLowerhi);
        yGLeftlo  = _mm512_div_ps(zStrength2, yGLeftlo);
        yGLefthi  = _mm512_div_ps(zStrength2, yGLefthi);
        yGRightlo = _mm512_div_ps(zStrength2, yGRightlo);
        yGRighthi = _mm512_div_ps(zStrength2, yGRighthi);
#endif

        __m512 yAddLo0, yAddHi0, yAddLo1, yAddHi1;
        yGUpperlo = _mm512_mul_ps(yGUpperlo, _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcUpperDiff)));
        yGUpperhi = _mm512_mul_ps(yGUpperhi, _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcUpperDiff)));
        yGLeftlo  = _mm512_mul_ps(yGLeftlo,  _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLeftDiff)));
        yGLefthi  = _mm512_mul_ps(yGLefthi,  _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLeftDiff)));

        yAddLo0   = _mm512_fmadd_ps(yGLowerlo, _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLowerDiff)), yGUpperlo);
        yAddHi0   = _mm512_fmadd_ps(yGLowerhi, _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLowerDiff)), yGUpperhi);
        yAddLo1   = _mm512_fmadd_ps(yGRightlo, _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcRightDiff)), yGLeftlo);
        yAddHi1   = _mm512_fmadd_ps(yGRighthi, _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcRightDiff)), yGLefthi);

        yAddLo0 = _mm512_add_ps(yAddLo0, yAddLo1);
        yAddHi0 = _mm512_add_ps(yAddHi0, yAddHi1);

        __m512i ySrc = _mm512_loadu_si512((__m512i *)(src));
        _mm512_storeu_si512((__m512i *)(dst), _mm512_add_epi16(ySrc, _mm512_packs_epi32(_mm512_cvtps_epi32(yAddLo0), _mm512_cvtps_epi32(yAddHi0))));
    }
}

void pmd_mt_avx512(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    PIXEL_YC *gauss    = ((PMD_MT_PRM *)param2)->gauss;
    const int w = fpip->w;
    const int h = fpip->h;
    const int max_w = fpip->max_w;
    int y_start = h *  thread_id    / thread_num;
    int y_fin   = h * (thread_id+1) / thread_num;

    //以下、修正PMD法によるノイズ除去
    const int strength =  ((PMD_MT_PRM *)param2)->strength;
    const int threshold = ((PMD_MT_PRM *)param2)->threshold;

    const float range = 4.0f;
    const float strength2 = strength/100.0f;
    //閾値の設定を変えた方が使いやすいです
    const float inv_threshold2 = (float)(1.0 / pow(2.0, threshold/10.0));

    // = (1.0 / range) * (   (1.0/ (1.0 + (  x*x / threshold2 )) )  * strength2 )
    // = (1.0 / range) * (   (1.0/ (1.0 + (  x*x * inv_threshold2 )) )  * strength2 )

    __m512 zInvThreshold2 = _mm512_set1_ps(inv_threshold2);
    __m512 zStrength2 = _mm512_set1_ps(strength2 / range);
    __m512 zOnef = _mm512_set1_ps(1.0f);

    //最初の行はそのままコピー
    if (0 == y_start) {
        memcpy_avx512<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
        y_start++;
    }
    //最後の行はそのままコピー
    y_fin -= (h == y_fin);

    uint8_t *src_line = (uint8_t *)(fpip->ycp_edit + y_start * max_w);
    uint8_t *dst_line = (uint8_t *)(fpip->ycp_temp + y_start * max_w);
    uint8_t *gau_line = (uint8_t *)(gauss          + y_start * max_w);

    for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC), gau_line += max_w * sizeof(PIXEL_YC)) {
        uint8_t *src = src_line;
        uint8_t *dst = dst_line;
        uint8_t *gau = gau_line;

        //まずは、先端終端ピクセルを気にせず普通に処理してしまう
        //先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
        //最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
        //先端終端ピクセルは後から上書きコピーする
        size_t process_size_in_byte = w * sizeof(PIXEL_YC);
        const size_t dst_mod64 = (int)((size_t)dst & 0x3f);
        if (dst_mod64) {
            int dw = 64 - dst_mod64;
            pmd_mt_avx512_line<false>(dst, src, gau, dw, max_w, zInvThreshold2, zStrength2, zOnef);
            src += dw; dst += dw; gau += dw; process_size_in_byte -= dw;
        }
        pmd_mt_avx512_line<true>(dst, src, gau, process_size_in_byte & (~0x3f), max_w, zInvThreshold2, zStrength2, zOnef);
        if (process_size_in_byte & 0x3f) {
            src += process_size_in_byte - 64;
            dst += process_size_in_byte - 64;
            gau += process_size_in_byte - 64;
            pmd_mt_avx512_line<false>(dst, src, gau, 64, max_w, zInvThreshold2, zStrength2, zOnef);
        }
        //先端と終端のピクセルをそのままコピー
        *(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
        *(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
    }
    //最後の行はそのままコピー
    if (h-1 == y_fin) {
        memcpy_avx512<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
    }
    _mm256_zeroupper();
}

void pmd_mt_exp_avx512(int thread_id, int thread_num, void *param1, void *param2) {
    const int pmd_nzsize = ((PMD_MT_PRM *)param2)->nzsize;
    if (pmd_nzsize < 63) {
        pmd_mt_exp_avx512_base<false, 1>(thread_id, thread_num, param1, param2);
    } else if (pmd_nzsize < 127) {
        pmd_mt_exp_avx512_base<false, 2>(thread_id, thread_num, param1, param2);
    } else if (pmd_nzsize < 191) {
        pmd_mt_exp_avx512_base<false, 3>(thread_id, thread_num, param1, param2);
    } else if (pmd_nzsize < 255) {
        pmd_mt_exp_avx512_base<false, 4>(thread_id, thread_num, param1, param2);
    } else {
        pmd_mt_exp_avx512_base<false, 5>(thread_id, thread_num, param1, param2);
    }
}

void pmd_mt_exp_avx512vnni(int thread_id, int thread_num, void *param1, void *param2) {
    const int pmd_nzsize = ((PMD_MT_PRM *)param2)->nzsize;
    if (pmd_nzsize < 63) {
        pmd_mt_exp_avx512_base<true, 1>(thread_id, thread_num, param1, param2);
    } else if (pmd_nzsize < 127) {
        pmd_mt_exp_avx512_base<true, 2>(thread_id, thread_num, param1, param2);
    } else if (pmd_nzsize < 191) {
        pmd_mt_exp_avx512_base<true, 3>(thread_id, thread_num, param1, param2);
    } else if (pmd_nzsize < 255) {
        pmd_mt_exp_avx512_base<true, 4>(thread_id, thread_num, param1, param2);
    } else {
        pmd_mt_exp_avx512_base<true, 5>(thread_id, thread_num, param1, param2);
    }
}

template <bool use_stream>
static __forceinline void anisotropic_mt_avx512_line(uint8_t *dst, uint8_t *src, int process_size_in_byte, int max_w, const __m512& zInvThreshold2, const __m512& zStrength2, const __m512& zOnef) {
    uint8_t *src_fin = src + process_size_in_byte;
    for ( ; src < src_fin; src += 64, dst += 64) {
        __m512i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
        getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);

        __m512 xSUpperlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcUpperDiff));
        __m512 xSUpperhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcUpperDiff));
        __m512 xSLowerlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLowerDiff));
        __m512 xSLowerhi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLowerDiff));
        __m512 xSLeftlo  = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcLeftDiff));
        __m512 xSLefthi  = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcLeftDiff));
        __m512 xSRightlo = _mm512_cvtepi32_ps(cvtlo512_epi16_epi32(ySrcRightDiff));
        __m512 xSRighthi = _mm512_cvtepi32_ps(cvthi512_epi16_epi32(ySrcRightDiff));

        __m512 yTUpperlo = _mm512_mul_ps(xSUpperlo, xSUpperlo);
        __m512 yTUpperhi = _mm512_mul_ps(xSUpperhi, xSUpperhi);
        __m512 yTLowerlo = _mm512_mul_ps(xSLowerlo, xSLowerlo);
        __m512 yTLowerhi = _mm512_mul_ps(xSLowerhi, xSLowerhi);
        __m512 yTLeftlo  = _mm512_mul_ps(xSLeftlo,  xSLeftlo);
        __m512 yTLefthi  = _mm512_mul_ps(xSLefthi,  xSLefthi);
        __m512 yTRightlo = _mm512_mul_ps(xSRightlo, xSRightlo);
        __m512 yTRighthi = _mm512_mul_ps(xSRighthi, xSRighthi);

        yTUpperlo = _mm512_fmadd_ps(yTUpperlo, zInvThreshold2, zOnef);
        yTUpperhi = _mm512_fmadd_ps(yTUpperhi, zInvThreshold2, zOnef);
        yTLowerlo = _mm512_fmadd_ps(yTLowerlo, zInvThreshold2, zOnef);
        yTLowerhi = _mm512_fmadd_ps(yTLowerhi, zInvThreshold2, zOnef);
        yTLeftlo  = _mm512_fmadd_ps(yTLeftlo,  zInvThreshold2, zOnef);
        yTLefthi  = _mm512_fmadd_ps(yTLefthi,  zInvThreshold2, zOnef);
        yTRightlo = _mm512_fmadd_ps(yTRightlo, zInvThreshold2, zOnef);
        yTRighthi = _mm512_fmadd_ps(yTRighthi, zInvThreshold2, zOnef);

#if 1
        yTUpperlo = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTUpperlo));
        yTUpperhi = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTUpperhi));
        yTLowerlo = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTLowerlo));
        yTLowerhi = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTLowerhi));
        yTLeftlo  = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTLeftlo));
        yTLefthi  = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTLefthi));
        yTRightlo = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTRightlo));
        yTRighthi = _mm512_mul_ps(zStrength2, _mm512_rcp14_ps(yTRighthi));
#else
        yTUpperlo = _mm512_div_ps(zStrength2, yTUpperlo);
        yTUpperhi = _mm512_div_ps(zStrength2, yTUpperhi);
        yTLowerlo = _mm512_div_ps(zStrength2, yTLowerlo);
        yTLowerhi = _mm512_div_ps(zStrength2, yTLowerhi);
        yTLeftlo  = _mm512_div_ps(zStrength2, yTLeftlo);
        yTLefthi  = _mm512_div_ps(zStrength2, yTLefthi);
        yTRightlo = _mm512_div_ps(zStrength2, yTRightlo);
        yTRighthi = _mm512_div_ps(zStrength2, yTRighthi);
#endif

        __m512 yAddLo0, yAddHi0, yAddLo1, yAddHi1;
        yAddLo0   = _mm512_fmadd_ps(xSLowerlo, yTLowerlo, _mm512_mul_ps(xSUpperlo, yTUpperlo));
        yAddHi0   = _mm512_fmadd_ps(xSLowerhi, yTLowerhi, _mm512_mul_ps(xSUpperhi, yTUpperhi));
        yAddLo1   = _mm512_fmadd_ps(xSRightlo, yTRightlo, _mm512_mul_ps(xSLeftlo, yTLeftlo));
        yAddHi1   = _mm512_fmadd_ps(xSRighthi, yTRighthi, _mm512_mul_ps(xSLefthi, yTLefthi));

        yAddLo0 = _mm512_add_ps(yAddLo0, yAddLo1);
        yAddHi0 = _mm512_add_ps(yAddHi0, yAddHi1);

        __m512i ySrc = _mm512_loadu_si512((__m512i *)(src));
        _mm512_storeu_si512((__m512i *)(dst), _mm512_add_epi16(ySrc, _mm512_packs_epi32(_mm512_cvtps_epi32(yAddLo0), _mm512_cvtps_epi32(yAddHi0))));
    }
}

void anisotropic_mt_avx512(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    const int w = fpip->w;
    const int h = fpip->h;
    const int max_w = fpip->max_w;
    int y_start = h *  thread_id    / thread_num;
    int y_fin   = h * (thread_id+1) / thread_num;

    const int strength =  ((PMD_MT_PRM *)param2)->strength;
    const int threshold = ((PMD_MT_PRM *)param2)->threshold;

    const float range = 4.0f;
    const float strength2 = strength/100.0f;
    //閾値の設定を変えた方が使いやすいです
    const float inv_threshold2 = (float)(1.0 / (threshold*16/10.0*threshold*16/10.0));

    // = (1.0 / range) * (   (1.0/ (1.0 + (  x*x / threshold2 )) )  * strength2 )
    // = (1.0 / range) * (   (1.0/ (1.0 + (  x*x * inv_threshold2 )) )  * strength2 )

    __m512 zInvThreshold2 = _mm512_set1_ps(inv_threshold2);
    __m512 zStrength2 = _mm512_set1_ps(strength2 / range);
    __m512 zOnef = _mm512_set1_ps(1.0f);

    //最初の行はそのままコピー
    if (0 == y_start) {
        memcpy_avx512<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
        y_start++;
    }
    //最後の行はそのままコピー
    y_fin -= (h == y_fin);

    uint8_t *src_line = (uint8_t *)(fpip->ycp_edit + y_start * max_w);
    uint8_t *dst_line = (uint8_t *)(fpip->ycp_temp + y_start * max_w);

    for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC)) {
        uint8_t *src = src_line;
        uint8_t *dst = dst_line;

        //まずは、先端終端ピクセルを気にせず普通に処理してしまう
        //先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
        //最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
        //先端終端ピクセルは後から上書きコピーする
        uint32_t process_size_in_byte = w * sizeof(PIXEL_YC);
        const size_t dst_mod64 = (int)((size_t)dst & 0x3f);
        if (dst_mod64) {
            int dw = 64 - dst_mod64;
            anisotropic_mt_avx512_line<false>(dst, src, dw, max_w, zInvThreshold2, zStrength2, zOnef);
            src += dw; dst += dw; process_size_in_byte -= dw;
        }
        anisotropic_mt_avx512_line<true>(dst, src, process_size_in_byte & (~0x3f), max_w, zInvThreshold2, zStrength2, zOnef);
        if (process_size_in_byte & 0x3f) {
            src += process_size_in_byte - 64;
            dst += process_size_in_byte - 64;
            anisotropic_mt_avx512_line<false>(dst, src, 64, max_w, zInvThreshold2, zStrength2, zOnef);
        }
        //先端と終端をそのままコピー
        *(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
        *(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
    }
    //最後の行はそのままコピー
    if (h-1 == y_fin) {
        memcpy_avx512<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
    }
    _mm256_zeroupper();
}

void anisotropic_mt_exp_avx512(int thread_id, int thread_num, void *param1, void *param2) {
    anisotropic_mt_exp_avx512_base(thread_id, thread_num, param1, param2);
}

