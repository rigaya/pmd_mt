#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   1
#define USE_AVX2  1
#define USE_FMA3  1
#define USE_FMA4  0
#define USE_VPGATHER 0 //Haswellではvpgatherを使用したほうが遅い
#define USE_FMATH 0    //expのほうはfmathでexp計算をするよりも表引きのほうが高速

#if USE_FMATH
//#define FMATH_USE_XBYAK
#include <fmath.hpp>
#endif

#include <cstdint>
#include <algorithm>
#include <cmath>
#include "pmd_mt.h"
#include "filter.h"
#include "simd_util.h"
#include "pmd_mt_avx2.h"

// y0*1 + y1*4 + y2*6 + y3*4 + y4*1
static __forceinline __m256i gaussian_1_4_6_4_1(__m256i y0, __m256i y1, __m256i y2, const __m256i& y3, const __m256i& y4) {
    static const __declspec(align(32)) short MULTIPLIZER[16] = { 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6, 4, 6 };
    y0 = _mm256_adds_epi16(y0, y4);
    y1 = _mm256_adds_epi16(y1, y3);

    __m256i y0_lower = cvtlo256_epi16_epi32(y0);
    __m256i y0_upper = cvthi256_epi16_epi32(y0);
    __m256i y1_lower = _mm256_madd_epi16(_mm256_unpacklo_epi16(y1, y2), _mm256_load_si256((__m256i *)MULTIPLIZER));
    __m256i y1_upper = _mm256_madd_epi16(_mm256_unpackhi_epi16(y1, y2), _mm256_load_si256((__m256i *)MULTIPLIZER));

    y0_lower = _mm256_add_epi32(y0_lower, y1_lower);
    y0_upper = _mm256_add_epi32(y0_upper, y1_upper);
    y0_lower = _mm256_srai_epi32(y0_lower, 4);
    y0_upper = _mm256_srai_epi32(y0_upper, 4);

    return _mm256_packs_epi32(y0_lower, y0_upper);
}

static __forceinline void set_temp_buffer(int i_dst, int i_src, uint8_t *src_top, uint8_t *temp, int analyze_block, int max_w, int tmp_line_size) {
    uint8_t *src_ptr = src_top + i_src * max_w * sizeof(PIXEL_YC);
    uint8_t *tmp_ptr = temp + i_dst * tmp_line_size;
    uint8_t *temp_fin = tmp_ptr + analyze_block * sizeof(PIXEL_YC);
    __m256i y0, y1, y2, y3;
    for (; tmp_ptr < temp_fin; tmp_ptr += 128, src_ptr += 128) {
        y0 = _mm256_loadu_si256((const __m256i*)(src_ptr +  0));
        y1 = _mm256_loadu_si256((const __m256i*)(src_ptr + 32));
        y2 = _mm256_loadu_si256((const __m256i*)(src_ptr + 64));
        y3 = _mm256_loadu_si256((const __m256i*)(src_ptr + 96));
        _mm256_store_si256((__m256i*)(tmp_ptr +  0), y0);
        _mm256_store_si256((__m256i*)(tmp_ptr + 32), y1);
        _mm256_store_si256((__m256i*)(tmp_ptr + 64), y2);
        _mm256_store_si256((__m256i*)(tmp_ptr + 96), y3);
    }
}

void gaussianH_avx2(int thread_id, int thread_num, void *param1, void *param2) {
    avx2_dummy();
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    const int max_w = fpip->max_w;
    const int w     = fpip->w;
    const int h     = fpip->h;
    PIXEL_YC *ycp_src = (fpip->ycp_temp == param2) ? fpip->ycp_edit : fpip->ycp_temp;
    PIXEL_YC *ycp_buf = (PIXEL_YC *)param2;
    const int min_analyze_cycle = 16;
    const int max_block_size = 256;
    const int tmp_line_size = max_block_size * sizeof(PIXEL_YC);
    const int x_start = ((int)(w * thread_id / (double)thread_num + 0.5) + (min_analyze_cycle-1)) & ~(min_analyze_cycle-1);
    const int x_fin = ((int)(w * (thread_id+1) / (double)thread_num + 0.5) + (min_analyze_cycle-1)) & ~(min_analyze_cycle-1);
    __declspec(align(32)) uint8_t temp[tmp_line_size * 4];
    int analyze_block = max_block_size;
    uint8_t *src_top = (uint8_t *)ycp_src + x_start * sizeof(PIXEL_YC);
    uint8_t *buf_top = (uint8_t *)ycp_buf + x_start * sizeof(PIXEL_YC);

    for (int pos_x = x_start; pos_x < x_fin; pos_x += analyze_block, src_top += analyze_block * sizeof(PIXEL_YC), buf_top += analyze_block * sizeof(PIXEL_YC)) {
        analyze_block = std::min(x_fin - pos_x, max_block_size);
        //一時領域にデータを満たす
        set_temp_buffer(0, 0, src_top, temp, analyze_block, max_w, tmp_line_size);
        set_temp_buffer(1, 0, src_top, temp, analyze_block, max_w, tmp_line_size);
        set_temp_buffer(2, 0, src_top, temp, analyze_block, max_w, tmp_line_size);
        set_temp_buffer(3, 1, src_top, temp, analyze_block, max_w, tmp_line_size);

        uint8_t *buf_line = buf_top;
        for (int y = 0; y < h; y++, buf_line += max_w * sizeof(PIXEL_YC)) {
            uint8_t *src = src_top + std::min(y, h-3) * max_w * sizeof(PIXEL_YC);
            uint8_t *buf = buf_line;
            uint8_t *tmp = temp;
            for (int x = 0; x < analyze_block; x += 16, src += 96, buf += 96, tmp += 96) {
                auto tmp_ptr = [tmp, tmp_line_size](int y) { return tmp + (y&3)*tmp_line_size; };
                auto guassianH_process_internal = [&](int i) {
                    __m256i ySrc0 = _mm256_loadu_si256((__m256i *)(src + 2 * max_w * sizeof(PIXEL_YC) + (i<<5)));

                    __m256i y0 = _mm256_load_si256((__m256i *)(tmp_ptr(y+0) + (i<<5)));
                    __m256i y1 = _mm256_load_si256((__m256i *)(tmp_ptr(y+1) + (i<<5)));
                    __m256i y2 = _mm256_load_si256((__m256i *)(tmp_ptr(y+2) + (i<<5)));
                    __m256i y3 = _mm256_load_si256((__m256i *)(tmp_ptr(y+3) + (i<<5)));
                    __m256i y4 = ySrc0;
                    _mm256_storeu_si256((__m256i *)(buf + (i<<5)), gaussian_1_4_6_4_1(y0, y1, y2, y3, y4));

                    _mm256_store_si256((__m256i *)(tmp_ptr(y+4) + (i<<5)), ySrc0);
                };
                _mm_prefetch((char *)src + 3 * max_w * sizeof(PIXEL_YC) +  0, _MM_HINT_T0);
                _mm_prefetch((char *)src + 3 * max_w * sizeof(PIXEL_YC) + 64, _MM_HINT_T0);
                guassianH_process_internal(0);
                guassianH_process_internal(1);
                guassianH_process_internal(2);
            }
        }
    }
    _mm256_zeroupper();
}

// ySrc0   v-1, u-1, y-1, v-2, u-2, y-2,   x,   x
// ySrc1    u2,  y2,  v1,  u1,  y1,  v0,  u0,  y0
// ySrc2    y5,  v4,  u4,  y4,  v3,  u3,  y3,  v2
// ySrc3    v7,  u7,  y7,  v6,  u6,  y6,  v5,  u5
// ySrc4     x,   x,  v9,  u9,  y9,  v8,  u8,  y8

//dst <= 横方向にガウシアンのかかった YC48_0 --- YC48_7
static __forceinline void guassianV_process(uint8_t *dst, __m256i ySrc0, __m256i ySrc1, __m256i ySrc2, const __m256i& ySrc3, const __m256i& ySrc4) {
    __m256i ySum0, ySum1, ySum2, ySum3, ySum4;
    ySum0 = _mm256_alignr256_epi8(ySrc1, ySrc0, (32-12));
    ySum1 = _mm256_alignr256_epi8(ySrc1, ySrc0, (32- 6));
    ySum2 = ySrc1;
    ySum3 = _mm256_alignr256_epi8(ySrc2, ySrc1,  6);
    ySum4 = _mm256_alignr256_epi8(ySrc2, ySrc1, 12);

    _mm256_storeu_si256((__m256i *)(dst +  0), gaussian_1_4_6_4_1(ySum0, ySum1, ySum2, ySum3, ySum4));

    ySum0 = _mm256_alignr256_epi8(ySrc2, ySrc1, (32-12));
    ySum1 = _mm256_alignr256_epi8(ySrc2, ySrc1, (32- 6));
    ySum2 = ySrc2;
    ySum3 = _mm256_alignr256_epi8(ySrc3, ySrc2,  6);
    ySum4 = _mm256_alignr256_epi8(ySrc3, ySrc2, 12);

    _mm256_storeu_si256((__m256i *)(dst + 32), gaussian_1_4_6_4_1(ySum0, ySum1, ySum2, ySum3, ySum4));

    ySum0 = _mm256_alignr256_epi8(ySrc3, ySrc2, (32-12));
    ySum1 = _mm256_alignr256_epi8(ySrc3, ySrc2, (32- 6));
    ySum2 = ySrc3;
    ySum3 = _mm256_alignr256_epi8(ySrc4, ySrc3,  6);
    ySum4 = _mm256_alignr256_epi8(ySrc4, ySrc3, 12);

    _mm256_storeu_si256((__m256i *)(dst + 64), gaussian_1_4_6_4_1(ySum0, ySum1, ySum2, ySum3, ySum4));
}

void gaussianV_avx2(int thread_id, int thread_num, void *param1, void *param2) {
    avx2_dummy();
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    const int max_w = fpip->max_w;
    const int w     = fpip->w;
    const int h     = fpip->h;
    PIXEL_YC *ycp_buf = (fpip->ycp_temp == param2) ? fpip->ycp_edit : fpip->ycp_temp;
    PIXEL_YC *ycp_dst = (PIXEL_YC *)param2;
    const int y_start = (h *  thread_id   ) / thread_num;
    const int y_fin   = (h * (thread_id+1)) / thread_num;
    uint8_t *buf_line = (uint8_t *)ycp_buf + max_w * y_start * sizeof(PIXEL_YC);
    uint8_t *dst_line = (uint8_t *)ycp_dst + max_w * y_start * sizeof(PIXEL_YC);

    for (int y = y_start; y < y_fin; y++, dst_line += max_w * sizeof(PIXEL_YC), buf_line += max_w * sizeof(PIXEL_YC)) {
        uint8_t *buf = buf_line;
        uint8_t *dst = dst_line;
        __m256i ySrc0, ySrc1, ySrc2, ySrc3, ySrc4;
        ySrc1 = _mm256_loadu_si256((__m256i *)(buf +  0));
        ySrc0 = _mm256_slli256_si256(ySrc1, (32-6));
        ySrc0 = _mm256_or_si256(ySrc0, _mm256_srli_si256(ySrc0, 6));
        int x_cycles = 0;
        const int x_cycle_count = (w - 1) >> 4;
        for (; x_cycles < x_cycle_count; x_cycles++, dst += 96, buf += 96) {
            ySrc2 = _mm256_loadu_si256((__m256i *)(buf + 32));
            ySrc3 = _mm256_loadu_si256((__m256i *)(buf + 64));
            ySrc4 = _mm256_loadu_si256((__m256i *)(buf + 96));

            guassianV_process(dst, ySrc0, ySrc1, ySrc2, ySrc3, ySrc4);

            ySrc0 = ySrc3;
            ySrc1 = ySrc4;
        }
        //終端
        const int last_dispalcement = (x_cycles << 4) + 16 - w;
        if (last_dispalcement) {
            dst -= last_dispalcement * sizeof(PIXEL_YC);
            buf -= last_dispalcement * sizeof(PIXEL_YC);
            ySrc0 = _mm256_loadu_si256((__m256i *)(buf - 32));
            ySrc1 = _mm256_loadu_si256((__m256i *)(buf +  0));
        }
        ySrc2 = _mm256_loadu_si256((__m256i *)(buf + 32));
        ySrc3 = _mm256_loadu_si256((__m256i *)(buf + 64));
        ySrc4 = _mm256_srli256_si256(ySrc3, (32-6));
        ySrc4 = _mm256_or_si256(ySrc4, _mm256_slli_si256(ySrc4, 6));
        guassianV_process(dst, ySrc0, ySrc1, ySrc2, ySrc3, ySrc4);
    }
    _mm256_zeroupper();
}


template<int line_size>
static __forceinline void copy_bufline_avx2(void *dst, const void *src) {
    static_assert(line_size % 128 == 0, "line_size % 128");
    int n = line_size / 128;
    const char *srcptr = (const char *)src;
    char *dstptr = (char *)dst;
    do {
        __m256i y0 = _mm256_load_si256((const __m256i *)(srcptr + 0));
        __m256i y1 = _mm256_load_si256((const __m256i *)(srcptr + 32));
        __m256i y2 = _mm256_load_si256((const __m256i *)(srcptr + 64));
        __m256i y3 = _mm256_load_si256((const __m256i *)(srcptr + 96));
        _mm256_store_si256((__m256i *)(dstptr + 0), y0);
        _mm256_store_si256((__m256i *)(dstptr + 32), y1);
        _mm256_store_si256((__m256i *)(dstptr + 64), y2);
        _mm256_store_si256((__m256i *)(dstptr + 96), y3);
        srcptr += 128;
        dstptr += 128;
        n--;
    } while (n);
}


static __forceinline void gather_y_u_v_to_yc48(__m256i& y0, __m256i& y1, __m256i& y2) {
    __m256i y3, y4, y5;

    alignas(16) static const uint8_t shuffle_yc48[32] = {
        0x00, 0x01, 0x06, 0x07, 0x0C, 0x0D, 0x02, 0x03, 0x08, 0x09, 0x0E, 0x0F, 0x04, 0x05, 0x0A, 0x0B,
        0x00, 0x01, 0x06, 0x07, 0x0C, 0x0D, 0x02, 0x03, 0x08, 0x09, 0x0E, 0x0F, 0x04, 0x05, 0x0A, 0x0B
    };
    y5 = _mm256_load_si256((__m256i *)shuffle_yc48);
    y0 = _mm256_shuffle_epi8(y0, y5);                             //5,2,7,4,1,6,3,0
    y1 = _mm256_shuffle_epi8(y1, _mm256_alignr_epi8(y5, y5, 14)); //2,7,4,1,6,3,0,5
    y2 = _mm256_shuffle_epi8(y2, _mm256_alignr_epi8(y5, y5, 12)); //7,4,1,6,3,0,5,2

    y3 = _mm256_blend_epi16(y0, y1, 0x80 + 0x10 + 0x02);
    y3 = _mm256_blend_epi16(y3, y2, 0x20 + 0x04);        //384, 0

    y4 = _mm256_blend_epi16(y2, y1, 0x20 + 0x04);
    y4 = _mm256_blend_epi16(y4, y0, 0x80 + 0x10 + 0x02); //512, 128

    y2 = _mm256_blend_epi16(y2, y0, 0x20 + 0x04);
    y2 = _mm256_blend_epi16(y2, y1, 0x40 + 0x08 + 0x01); //640, 256

    y0 = _mm256_permute2x128_si256(y3, y4, (0x02<<4) + 0x00); // 128, 0
    y1 = _mm256_blend_epi32(y2, y3, 0xf0);                    // 384, 256
    y2 = _mm256_permute2x128_si256(y4, y2, (0x03<<4) + 0x01); // 640, 512
}

static __forceinline void store_y_u_v_to_yc48(char *ptr, __m256i yY, __m256i yU, __m256i yV) {
    gather_y_u_v_to_yc48(yY, yU, yV);

    _mm256_storeu_si256((__m256i *)(ptr +  0), yY); // 128,   0
    _mm256_storeu_si256((__m256i *)(ptr + 32), yU); // 384, 256
    _mm256_storeu_si256((__m256i *)(ptr + 64), yV); // 768, 512
}

static void store_y_u_v_to_yc48_per_pix(char *ptr, __m256i yY, __m256i yU, __m256i yV, int n) {
    PIXEL_YC *dst = (PIXEL_YC *)ptr;
    int16_t __declspec(align(32)) bufY[16];
    int16_t __declspec(align(32)) bufU[16];
    int16_t __declspec(align(32)) bufV[16];
    _mm256_storeu_si256((__m256i *)bufY, yY);
    _mm256_storeu_si256((__m256i *)bufU, yU);
    _mm256_storeu_si256((__m256i *)bufV, yV);
    for (int i = 0; i < n; i++) {
        dst[i].y = bufY[i];
        dst[i].cb = bufU[i];
        dst[i].cr = bufV[i];
    }
}

template<bool aligned>
void __forceinline afs_load_yc48(__m256i& y, __m256i& cb, __m256i& cr, const char *src) {
    alignas(16) static const uint8_t  Array_SUFFLE_YCP_Y[32] = {
        0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
        0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11
    };
    __m256i yShuffleArray = _mm256_load_si256((__m256i *)Array_SUFFLE_YCP_Y);
    __m256i yA0 = (aligned) ? _mm256_load_si256((__m256i *)(src +  0)) : _mm256_loadu_si256((__m256i *)(src +  0));
    __m256i yA1 = (aligned) ? _mm256_load_si256((__m256i *)(src + 32)) : _mm256_loadu_si256((__m256i *)(src + 32));
    __m256i yA2 = (aligned) ? _mm256_load_si256((__m256i *)(src + 64)) : _mm256_loadu_si256((__m256i *)(src + 64));

    __m256i y0, y1, y2, y3, y5, y6, y7;

    // yA0 = 128, 0
    // yA1 = 384, 256
    // yA2 = 640, 512

    //_mm256_permute2x128_si256よりなるべく_mm256_blend_epi32を使う
    y1 = _mm256_blend_epi32(yA0, yA1, 0xf0);                    // 384, 0
    y2 = _mm256_permute2x128_si256(yA0, yA2, (0x02 << 4) + 0x01); // 512, 128
    y3 = _mm256_blend_epi32(yA1, yA2, 0xf0);                    // 640, 256

    y0 = _mm256_blend_epi16(y1, y3, 0x20 + 0x04);
    y6 = _mm256_blend_epi16(y1, y3, 0x40 + 0x08 + 0x01);
    y7 = _mm256_blend_epi16(y1, y3, 0x80 + 0x10 + 0x02);

    y0 = _mm256_blend_epi16(y0, y2, 0x80 + 0x10 + 0x02);
    y6 = _mm256_blend_epi16(y6, y2, 0x20 + 0x04);
    y7 = _mm256_blend_epi16(y7, y2, 0x40 + 0x08 + 0x01);

    y0 = _mm256_shuffle_epi8(y0, yShuffleArray); //Y
    y5 = _mm256_shuffle_epi8(y6, _mm256_alignr_epi8(yShuffleArray, yShuffleArray, 6)); //Cb
    y7 = _mm256_shuffle_epi8(y7, _mm256_alignr_epi8(yShuffleArray, yShuffleArray, 12)); //Cr

    y  = y0;
    cb = y5;
    cr = y7;
}

//1,2,1加算を行う
static __forceinline __m256i smooth_3x3_vertical(const __m256i &y0, const __m256i &y1, const __m256i &y2) {
    __m256i ySum = _mm256_add_epi16(_mm256_add_epi16(y1, y1), _mm256_set1_epi16(2));
    ySum = _mm256_add_epi16(ySum, _mm256_add_epi16(y0, y2));
    return _mm256_srai_epi16(ySum, 2);
}
//1,4,6,4,1加算
static __forceinline __m256i smooth_5x5_vertical(const __m256i& y0, const __m256i &y1, const __m256i &y2, const __m256i &y3, const __m256i &y4) {
    return gaussian_1_4_6_4_1(y0, y1, y2, y3, y4);
}
#pragma warning (push)
#pragma warning (disable:4100) //warning C4100: 引数は関数の本体部で 1 度も参照されません。
//1,6,15,20,15,6,1加算
static __forceinline __m256i smooth_7x7_vertical(const __m256i &y0, const __m256i &y1, const __m256i &y2, const __m256i &y3, const __m256i &y4, const __m256i &y5, const __m256i &y6) {
    // ###################
    //   !!!! 未実装 !!!!!
    // ###################
    return y3;
}
#pragma warning (pop)

//1,2,1加算を行う
static __forceinline __m256i smooth_3x3_horizontal(const __m256i &y0, const __m256i &y1, const __m256i &y2) {
    return smooth_3x3_vertical(
        _mm256_alignr256_epi8(y1, y0, (32 - 2)),
        y1,
        _mm256_alignr256_epi8(y2, y1, 2)
    );
}
//1,4,6,4,1加算
static __forceinline __m256i smooth_5x5_horizontal(const __m256i &y0, const __m256i &y1, const __m256i &y2) {
    return smooth_5x5_vertical(
        _mm256_alignr256_epi8(y1, y0, (32 - 4)),
        _mm256_alignr256_epi8(y1, y0, (32 - 2)),
        y1,
        _mm256_alignr256_epi8(y2, y1, 2),
        _mm256_alignr256_epi8(y2, y1, 4)
    );
}
//1,6,15,20,15,6,1加算
static __forceinline __m256i smooth_7x7_horizontal(const __m256i &y0, const __m256i &y1, const __m256i &y2) {
    __m256i p0 = _mm256_alignr256_epi8(y1, y0, (32 - 6));
    __m256i p1 = _mm256_alignr256_epi8(y1, y0, (32 - 4));
    __m256i p2 = _mm256_alignr256_epi8(y1, y0, (32 - 2));
    __m256i p3 = y1;
    __m256i p4 = _mm256_alignr256_epi8(y2, y1, 2);
    __m256i p5 = _mm256_alignr256_epi8(y2, y1, 4);
    __m256i p6 = _mm256_alignr256_epi8(y2, y1, 6);
    // ###################
    //   !!!! 未実装 !!!!!
    // ###################
    return smooth_7x7_vertical(p0, p1, p2, p3, p4, p5, p6);
}

//rangeに応じてスムージング用の水平加算を行う
template<int range>
static __forceinline __m256i smooth_horizontal(const __m256i &y0, const __m256i &y1, const __m256i &y2) {
    static_assert(0 < range && range <= 2, "range >= 3 not implemeted!");
    switch (range) {
    case 3: return smooth_7x7_horizontal(y0, y1, y2);
    case 2: return smooth_5x5_horizontal(y0, y1, y2);
    case 1:
    default:return smooth_3x3_horizontal(y0, y1, y2);
    }
}

static const uint16_t __declspec(align(32)) SMOOTH_BLEND_MASK[32] = {
    0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
};

//スムージングでは、まず水平方向の加算結果をバッファに格納していく
//この関数は1ラインぶんの水平方向の加算 + バッファへの格納のみを行う
template<int range>
static __forceinline void smooth_fill_buffer_yc48(char *buf_ptr, const char *src_ptr, int x_start, int x_fin, int width, const __m256i& smooth_mask) {
    __m256i yY0, yU0, yV0;
    if (x_start == 0) {
        const PIXEL_YC *firstpix = (const PIXEL_YC *)src_ptr;
        yY0 = _mm256_broadcastw_epi16(_mm_loadu_si16(&firstpix->y));
        yU0 = _mm256_broadcastw_epi16(_mm_loadu_si16(&firstpix->cb));
        yV0 = _mm256_broadcastw_epi16(_mm_loadu_si16(&firstpix->cr));
    } else {
        src_ptr += x_start * sizeof(PIXEL_YC);
        afs_load_yc48<false>(yY0, yU0, yV0, src_ptr - 96);
    }
    //横方向のループ数は、AVX2(256bit)か128bitかによって異なる (logo_pitchとは異なる)
    const int x_fin_align = (((USE_AVX2) ? ((x_fin - x_start) + 15) & ~15 : ((x_fin - x_start) + 7) & ~7)) - ((USE_AVX2) ? 16 : 8);
    __m256i yY1, yU1, yV1;
    afs_load_yc48<false>(yY1, yU1, yV1, src_ptr);
    __m256i yY2, yU2, yV2;
    for (int x = x_fin_align; x; x -= 16, src_ptr += 96, buf_ptr += 96) {
        afs_load_yc48<false>(yY2, yU2, yV2, src_ptr + 96);
        _mm256_storeu_si256((__m256i *)(buf_ptr +  0), smooth_horizontal<range>(yY0, yY1, yY2));
        _mm256_storeu_si256((__m256i *)(buf_ptr + 32), smooth_horizontal<range>(yU0, yU1, yU2));
        _mm256_storeu_si256((__m256i *)(buf_ptr + 64), smooth_horizontal<range>(yV0, yV1, yV2));
        yY0 = yY1; yY1 = yY2;
        yU0 = yU1; yU1 = yU2;
        yV0 = yV1; yV1 = yV2;
    }
    if (x_fin >= width) {
        const PIXEL_YC *lastpix = ((const PIXEL_YC *)src_ptr) + width - x_fin_align - x_start - 1;
        yY2 = _mm256_broadcastw_epi16(_mm_loadu_si16(&lastpix->y));
        yU2 = _mm256_broadcastw_epi16(_mm_loadu_si16(&lastpix->cb));
        yV2 = _mm256_broadcastw_epi16(_mm_loadu_si16(&lastpix->cr));
        yY1 = _mm256_blendv_epi8(yY1, yY2, smooth_mask);
        yU1 = _mm256_blendv_epi8(yU1, yU2, smooth_mask);
        yV1 = _mm256_blendv_epi8(yV1, yV2, smooth_mask);
    } else {
        afs_load_yc48<false>(yY2, yU2, yV2, src_ptr + 96);
    }
    _mm256_storeu_si256((__m256i *)(buf_ptr +  0), smooth_horizontal<range>(yY0, yY1, yY2));
    _mm256_storeu_si256((__m256i *)(buf_ptr + 32), smooth_horizontal<range>(yU0, yU1, yU2));
    _mm256_storeu_si256((__m256i *)(buf_ptr + 64), smooth_horizontal<range>(yV0, yV1, yV2));
}

//バッファのライン数によるオフセットを計算する
#define BUF_LINE_OFFSET(x) ((((x) & (buf_line - 1)) * line_size) * sizeof(int16_t))

#pragma warning (push)
#pragma warning (disable:4127) //warning C4127: 条件式が定数です。
//yNewLineResultの最新のラインの水平加算結果と、バッファに格納済みの水平加算結果を用いて、
//縦方向の加算を行い、スムージング結果を16bit整数に格納して返す。
//yNewLineResultの値は、新たにバッファに格納される
template<unsigned int range, int buf_line, int line_size>
static __forceinline void smooth_vertical(char *buf_ptr, __m256i& yResultY, __m256i &yResultU, __m256i &yResultV, int y) {
    __m256i yResultYOrg = yResultY;
    __m256i yResultUOrg = yResultU;
    __m256i yResultVOrg = yResultV;

    if (range == 1) {
        yResultY = smooth_3x3_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            yResultY
        );
        yResultU = smooth_3x3_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 32)),
            yResultU
        );
        yResultV = smooth_3x3_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 64)),
            yResultV
        );
    } else if (range == 2) {
        yResultY = smooth_5x5_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 0)),
            yResultY
        );
        yResultU = smooth_5x5_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 32)),
            yResultU
        );
        yResultV = smooth_5x5_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 64)),
            yResultV
        );
    } else if (range == 3) {
        yResultY = smooth_7x7_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 0)),
            yResultY
        );
        yResultU = smooth_7x7_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 32)),
            yResultU
        );
        yResultV = smooth_7x7_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 64)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 64)),
            yResultV
        );
    }
    _mm256_storeu_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) +  0), yResultYOrg);
    _mm256_storeu_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) + 32), yResultUOrg);
    _mm256_storeu_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) + 64), yResultVOrg);
}

template<unsigned int range, int buf_line, int line_size>
static __forceinline void smooth_vertical(char *buf_ptr, __m256i &yResultU, __m256i &yResultV, int y) {
    __m256i yResultUOrg = yResultU;
    __m256i yResultVOrg = yResultV;

    if (range == 1) {
        yResultU = smooth_3x3_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) +  0)),
            yResultU
        );
        yResultV = smooth_3x3_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 32)),
            yResultV
        );
    } else if (range == 2) {
        yResultU = smooth_5x5_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) +  0)),
            yResultU
        );
        yResultV = smooth_5x5_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 32)),
            yResultV
        );
    } else if (range == 3) {
        yResultU = smooth_7x7_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) +  0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) +  0)),
            yResultU
        );
        yResultV = smooth_7x7_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 32)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 32)),
            yResultV
        );
    }
    _mm256_storeu_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) +  0), yResultUOrg);
    _mm256_storeu_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) + 32), yResultVOrg);
}

template<unsigned int range, int buf_line, int line_size>
static __forceinline void smooth_vertical(char *buf_ptr, __m256i &yResultY, int y) {
    __m256i yResultYOrg = yResultY;
    if (range == 1) {
        yResultY = smooth_3x3_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            yResultY
        );
    } else if (range == 2) {
        yResultY = smooth_5x5_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 0)),
            yResultY
        );
    } else if (range == 3) {
        yResultY = smooth_7x7_vertical(
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 0) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 1) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 2) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 3) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 4) + 0)),
            _mm256_loadu_si256((const __m256i *)(buf_ptr + BUF_LINE_OFFSET(y + 5) + 0)),
            yResultY
        );
    }
    _mm256_storeu_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y + range * 2) + 0), yResultYOrg);
}
#pragma warning (pop)

//line_sizeはpixel数x3
template<int range, int line_size>
void gaussHV_yc48_avx2_base(char *dst, int dst_pitch, const char *src, int src_pitch, int x_start, int x_fin, int y_start, int y_fin, int width, int height) {
    static_assert(0 < range && range <= 2, "0 < range && range <= 2");
    static_assert(((line_size/3) & ((line_size/3)-1)) == 0 && line_size % 3 == 0, "((line_size/3) & ((line_size/3)-1)) == 0 && line_size % 3 == 0");
    //最低限必要なバッファのライン数の決定、計算上2の乗数を使用する
    //最後の1ラインは一時的に重複して保持させる(必要な物を読んだところに上書きしていく)ため、
    //range4 (9x9)なら8ラインあればよい
    const int buf_line = (range >= 3 ? 8 : (range >= 2 ? 4 : 2));
    //水平方向の加算結果を保持するバッファ
    int16_t __declspec(align(32)) buffer[buf_line * line_size];
    memset(buffer, 0, sizeof(buffer));
    const __m256i smooth_mask = _mm256_loadu_si256((__m256i *)&SMOOTH_BLEND_MASK[(((x_fin - x_start) & 15) == 0) ? 16 : 16 - ((x_fin - x_start) & 15)]);
    const bool store_per_pix_on_edge = dst_pitch < ((x_start + (((x_fin - x_start) + 15) & ~15)) * (int)sizeof(PIXEL_YC));

    src += y_start * src_pitch;
    dst += y_start * dst_pitch;

    //バッファのrange*2-1行目までを埋める (range*2行目はメインループ内でロードする)
    for (int i = (y_start == 0) ? 0 : -range; i < range; i++)
        smooth_fill_buffer_yc48<range>((char *)(buffer + (i + range) * line_size), src + i * src_pitch, x_start, x_fin, width, smooth_mask);

    if (y_start == 0) {
        //range行目と同じものでバッファの1行目～range行目まで埋める
        for (int i = 0; i < range; i++) {
            copy_bufline_avx2<line_size * sizeof(int16_t)>(buffer + i * line_size, buffer + range * line_size);
        }
    }

    //メインループ
    __m256i yDiff2Sum = _mm256_setzero_si256();
    int y = 0; //バッファのライン数のもととなるため、y=0で始めることは重要
    const int y_fin_loop = y_fin - y_start - ((y_fin >= height) ? range : 0); //水平加算用に先読みするため、rangeに配慮してループの終わりを決める
    for (; y < y_fin_loop; y++, dst += dst_pitch, src += src_pitch) {
        const char *src_ptr = src;
        char *dst_ptr = dst;
        char *buf_ptr = (char *)buffer;
        const int range_offset = range * src_pitch; //水平加算用に先読みする位置のオフセット YC48モードではsrc_pitchを使用する
        __m256i yY0, yU0, yV0;
        if (x_start == 0) {
            const PIXEL_YC *firstpix = (const PIXEL_YC *)(src_ptr + range_offset);
            yY0 = _mm256_broadcastw_epi16(_mm_loadu_si16(&firstpix->y));
            yU0 = _mm256_broadcastw_epi16(_mm_loadu_si16(&firstpix->cb));
            yV0 = _mm256_broadcastw_epi16(_mm_loadu_si16(&firstpix->cr));
        } else {
            src_ptr += x_start * sizeof(PIXEL_YC);
            dst_ptr += x_start * sizeof(PIXEL_YC);
            afs_load_yc48<false>(yY0, yU0, yV0, src_ptr + range_offset - 96);
        }
        __m256i yY1, yU1, yV1;
        afs_load_yc48<false>(yY1, yU1, yV1, src_ptr + range_offset);

        //横方向のループ数は、AVX2(256bit)か128bitかによって異なる (logo_pitchとは異なる)
        const int x_fin_align = (((USE_AVX2) ? ((x_fin - x_start) + 15) & ~15 : ((x_fin - x_start) + 7) & ~7)) - ((USE_AVX2) ? 16 : 8);
        for (int x = x_fin_align; x; x -= 16, src_ptr += 96, dst_ptr += 96, buf_ptr += 96) {
            __m256i yY2, yU2, yV2;
            afs_load_yc48<false>(yY2, yU2, yV2, src_ptr + range_offset + 96);
            //連続するデータy0, y1, y2を使って水平方向の加算を行う
            __m256i yResultY = smooth_horizontal<range>(yY0, yY1, yY2);
            __m256i yResultU = smooth_horizontal<range>(yU0, yU1, yU2);
            __m256i yResultV = smooth_horizontal<range>(yV0, yV1, yV2);
            //yResultとバッファに格納されている水平方向の加算結果を合わせて
            //垂直方向の加算を行い、スムージングを完成させる
            //このループで得た水平加算結果はバッファに新たに格納される (不要になったものを上書き)
            smooth_vertical<range, buf_line, line_size>(buf_ptr, yResultY, yResultU, yResultV, y);

            store_y_u_v_to_yc48(dst_ptr, yResultY, yResultU, yResultV);

            yY0 = yY1; yY1 = yY2;
            yU0 = yU1; yU1 = yU2;
            yV0 = yV1; yV1 = yV2;
        }

        __m256i yY2, yU2, yV2;
        if (x_fin >= width) {
            const PIXEL_YC *lastpix = ((const PIXEL_YC *)(src + range_offset)) + width - 1;
            yY2 = _mm256_broadcastw_epi16(_mm_loadu_si16(&lastpix->y));
            yU2 = _mm256_broadcastw_epi16(_mm_loadu_si16(&lastpix->cb));
            yV2 = _mm256_broadcastw_epi16(_mm_loadu_si16(&lastpix->cr));
            yY1 = _mm256_blendv_epi8(yY1, yY2, smooth_mask); // mask ? y2 : y1
            yU1 = _mm256_blendv_epi8(yU1, yU2, smooth_mask);
            yV1 = _mm256_blendv_epi8(yV1, yV2, smooth_mask);
        } else {
            afs_load_yc48<false>(yY2, yU2, yV2, src_ptr + range_offset + 96);
        }

        __m256i yResultY = smooth_horizontal<range>(yY0, yY1, yY2);
        __m256i yResultU = smooth_horizontal<range>(yU0, yU1, yU2);
        __m256i yResultV = smooth_horizontal<range>(yV0, yV1, yV2);
        smooth_vertical<range, buf_line, line_size>(buf_ptr, yResultY, yResultU, yResultV, y);

        if (store_per_pix_on_edge) {
            store_y_u_v_to_yc48_per_pix(dst_ptr, yResultY, yResultU, yResultV, (x_fin - x_start) & 15);
        } else {
            store_y_u_v_to_yc48(dst_ptr, yResultY, yResultU, yResultV);
        }
    }
    if (y_fin >= height) {
        //先読みできる分が終了したら、あとはバッファから読み込んで処理する
        //yとiの値に注意する
        for (int i = 1; i <= range; y++, i++, src += src_pitch, dst += dst_pitch) {
            const char *src_ptr = src + x_start * sizeof(PIXEL_YC);
            char *dst_ptr = dst + x_start * sizeof(PIXEL_YC);
            char *buf_ptr = (char *)buffer;
            const int x_fin_align = (((USE_AVX2) ? ((x_fin - x_start) + 15) & ~15 : ((x_fin - x_start) + 7) & ~7)) - ((USE_AVX2) ? 16 : 8);
            for (int x = x_fin_align; x; x -= 16, src_ptr += 96, dst_ptr += 96, buf_ptr += 96) {
                __m256i yResultY = _mm256_load_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 0));
                __m256i yResultU = _mm256_load_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 32));
                __m256i yResultV = _mm256_load_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 64));
                smooth_vertical<range, buf_line, line_size>(buf_ptr, yResultY, yResultU, yResultV, y);

                store_y_u_v_to_yc48(dst_ptr, yResultY, yResultU, yResultV);
            }
            __m256i yResultY = _mm256_load_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 0));
            __m256i yResultU = _mm256_load_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 32));
            __m256i yResultV = _mm256_load_si256((__m256i *)(buf_ptr + BUF_LINE_OFFSET(y - 1 + range * 2) + 64));
            smooth_vertical<range, buf_line, line_size>(buf_ptr, yResultY, yResultU, yResultV, y);

            if (store_per_pix_on_edge) {
                store_y_u_v_to_yc48_per_pix(dst_ptr, yResultY, yResultU, yResultV, (x_fin - x_start) & 15);
            } else {
                store_y_u_v_to_yc48(dst_ptr, yResultY, yResultU, yResultV);
            }
        }
    }
}

void gaussianHV_avx2(int thread_id, int thread_num, void *param1, void *param2) {
    avx2_dummy();
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
            gaussHV_yc48_avx2_base<2, BLOCK_SIZE_YCP * 3>((char *)ycp_dst, max_w * (int)sizeof(PIXEL_YC), (const char *)ycp_buf, max_w * (int)sizeof(PIXEL_YC), pos_x, pos_x + analyze_block, pos_y, y_fin, w, h);
        }
    } else {
        for (; x_fin - pos_x > max_block_size; pos_x += analyze_block) {
            analyze_block = std::min(x_fin - pos_x, max_block_size);
            gaussHV_yc48_avx2_base<2, BLOCK_SIZE_YCP * 3>((char *)ycp_dst, max_w * (int)sizeof(PIXEL_YC), (const char *)ycp_buf, max_w * (int)sizeof(PIXEL_YC), pos_x, pos_x + analyze_block, pos_y, y_fin, w, h);
        }
        if (pos_x < w) {
            analyze_block = ((w - pos_x) + (min_analyze_cycle - 1)) & ~(min_analyze_cycle - 1);
            pos_x = w - analyze_block;
            gaussHV_yc48_avx2_base<2, BLOCK_SIZE_YCP * 3>((char *)ycp_dst, max_w * (int)sizeof(PIXEL_YC), (const char *)ycp_buf, max_w * (int)sizeof(PIXEL_YC), pos_x, pos_x + analyze_block, pos_y, y_fin, w, h);
        }
    }
}

//---------------------------------------------------------------------
//        修正PDMマルチスレッド関数
//---------------------------------------------------------------------
template <bool use_stream>
static __forceinline void pmd_mt_avx2_line(uint8_t *dst, uint8_t *src, uint8_t *gau, int process_size_in_byte, int max_w, const __m256& yInvThreshold2, const __m256& yStrength2, const __m256& yOnef) {
    uint8_t *src_fin = src + process_size_in_byte;
    for (; src < src_fin; src += 32, dst += 32, gau += 32) {
        __m256i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
        __m256i yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff;
        getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);
        getDiff(gau, max_w, yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff);

        __m256 yGUpperlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(yGauUpperDiff));
        __m256 yGUpperhi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(yGauUpperDiff));
        __m256 yGLowerlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(yGauLowerDiff));
        __m256 yGLowerhi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(yGauLowerDiff));
        __m256 yGLeftlo  = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(yGauLeftDiff));
        __m256 yGLefthi  = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(yGauLeftDiff));
        __m256 yGRightlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(yGauRightDiff));
        __m256 yGRighthi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(yGauRightDiff));

        yGUpperlo = _mm256_mul_ps(yGUpperlo, yGUpperlo);
        yGUpperhi = _mm256_mul_ps(yGUpperhi, yGUpperhi);
        yGLowerlo = _mm256_mul_ps(yGLowerlo, yGLowerlo);
        yGLowerhi = _mm256_mul_ps(yGLowerhi, yGLowerhi);
        yGLeftlo  = _mm256_mul_ps(yGLeftlo,  yGLeftlo);
        yGLefthi  = _mm256_mul_ps(yGLefthi,  yGLefthi);
        yGRightlo = _mm256_mul_ps(yGRightlo, yGRightlo);
        yGRighthi = _mm256_mul_ps(yGRighthi, yGRighthi);

        yGUpperlo = _mm256_fmadd_ps(yGUpperlo, yInvThreshold2, yOnef);
        yGUpperhi = _mm256_fmadd_ps(yGUpperhi, yInvThreshold2, yOnef);
        yGLowerlo = _mm256_fmadd_ps(yGLowerlo, yInvThreshold2, yOnef);
        yGLowerhi = _mm256_fmadd_ps(yGLowerhi, yInvThreshold2, yOnef);
        yGLeftlo  = _mm256_fmadd_ps(yGLeftlo,  yInvThreshold2, yOnef);
        yGLefthi  = _mm256_fmadd_ps(yGLefthi,  yInvThreshold2, yOnef);
        yGRightlo = _mm256_fmadd_ps(yGRightlo, yInvThreshold2, yOnef);
        yGRighthi = _mm256_fmadd_ps(yGRighthi, yInvThreshold2, yOnef);

#if 0
        yGUpperlo = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGUpperlo));
        yGUpperhi = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGUpperhi));
        yGLowerlo = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGLowerlo));
        yGLowerhi = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGLowerhi));
        yGLeftlo  = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGLeftlo));
        yGLefthi  = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGLefthi));
        yGRightlo = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGRightlo));
        yGRighthi = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yGRighthi));
#else
        yGUpperlo = _mm256_div_ps(yStrength2, yGUpperlo);
        yGUpperhi = _mm256_div_ps(yStrength2, yGUpperhi);
        yGLowerlo = _mm256_div_ps(yStrength2, yGLowerlo);
        yGLowerhi = _mm256_div_ps(yStrength2, yGLowerhi);
        yGLeftlo  = _mm256_div_ps(yStrength2, yGLeftlo);
        yGLefthi  = _mm256_div_ps(yStrength2, yGLefthi);
        yGRightlo = _mm256_div_ps(yStrength2, yGRightlo);
        yGRighthi = _mm256_div_ps(yStrength2, yGRighthi);
#endif

        __m256 yAddLo0, yAddHi0, yAddLo1, yAddHi1;
        yGUpperlo = _mm256_mul_ps(yGUpperlo, _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcUpperDiff)));
        yGUpperhi = _mm256_mul_ps(yGUpperhi, _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcUpperDiff)));
        yGLeftlo  = _mm256_mul_ps(yGLeftlo,  _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcLeftDiff)));
        yGLefthi  = _mm256_mul_ps(yGLefthi,  _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcLeftDiff)));

        yAddLo0   = _mm256_fmadd_ps(yGLowerlo, _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcLowerDiff)), yGUpperlo);
        yAddHi0   = _mm256_fmadd_ps(yGLowerhi, _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcLowerDiff)), yGUpperhi);
        yAddLo1   = _mm256_fmadd_ps(yGRightlo, _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcRightDiff)), yGLeftlo);
        yAddHi1   = _mm256_fmadd_ps(yGRighthi, _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcRightDiff)), yGLefthi);

        yAddLo0 = _mm256_add_ps(yAddLo0, yAddLo1);
        yAddHi0 = _mm256_add_ps(yAddHi0, yAddHi1);

        __m256i ySrc = _mm256_loadu_si256((__m256i *)(src));
#define _mm256_stream_switch_si256(x, ymm) ((use_stream) ? _mm256_stream_si256((x), (ymm)) : _mm256_storeu_si256((x), (ymm)))
        _mm256_stream_switch_si256((__m256i *)(dst), _mm256_add_epi16(ySrc, _mm256_packs_epi32(_mm256_cvtps_epi32(yAddLo0), _mm256_cvtps_epi32(yAddHi0))));
#undef _mm256_stream_switch_si256
    }
}

void pmd_mt_avx2_fma3(int thread_id, int thread_num, void *param1, void *param2) {
    avx2_dummy();
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

    __m256 yInvThreshold2 = _mm256_set1_ps(inv_threshold2);
    __m256 yStrength2 = _mm256_set1_ps(strength2 / range);
    __m256 yOnef = _mm256_set1_ps(1.0f);

    //最初の行はそのままコピー
    if (0 == y_start) {
        memcpy_avx2<false, false, false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
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
        const size_t dst_mod32 = (int)((size_t)dst & 0x1f);
        if (dst_mod32) {
            int dw = 32 - dst_mod32;
            pmd_mt_avx2_line<false>(dst, src, gau, dw, max_w, yInvThreshold2, yStrength2, yOnef);
            src += dw; dst += dw; gau += dw; process_size_in_byte -= dw;
        }
        pmd_mt_avx2_line<true>(dst, src, gau, process_size_in_byte & (~0x1f), max_w, yInvThreshold2, yStrength2, yOnef);
        if (process_size_in_byte & 0x1f) {
            src += process_size_in_byte - 32;
            dst += process_size_in_byte - 32;
            gau += process_size_in_byte - 32;
            pmd_mt_avx2_line<false>(dst, src, gau, 32, max_w, yInvThreshold2, yStrength2, yOnef);
        }
        //先端と終端のピクセルをそのままコピー
        *(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
        *(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
    }
    //最後の行はそのままコピー
    if (h-1 == y_fin) {
        memcpy_avx2<false, false, false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
    }
    _mm256_zeroupper();
}

void pmd_mt_exp_avx2(int thread_id, int thread_num, void *param1, void *param2) {
    pmd_mt_exp_avx2_base(thread_id, thread_num, param1, param2);
}

template <bool use_stream>
static __forceinline void anisotropic_mt_avx2_line(uint8_t *dst, uint8_t *src, int process_size_in_byte, int max_w, const __m256& yInvThreshold2, const __m256& yStrength2, const __m256& yOnef) {
    uint8_t *src_fin = src + process_size_in_byte;
    for ( ; src < src_fin; src += 32, dst += 32) {
        __m256i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
        getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);

        __m256 xSUpperlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcUpperDiff));
        __m256 xSUpperhi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcUpperDiff));
        __m256 xSLowerlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcLowerDiff));
        __m256 xSLowerhi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcLowerDiff));
        __m256 xSLeftlo  = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcLeftDiff));
        __m256 xSLefthi  = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcLeftDiff));
        __m256 xSRightlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcRightDiff));
        __m256 xSRighthi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcRightDiff));

        __m256 yTUpperlo = _mm256_mul_ps(xSUpperlo, xSUpperlo);
        __m256 yTUpperhi = _mm256_mul_ps(xSUpperhi, xSUpperhi);
        __m256 yTLowerlo = _mm256_mul_ps(xSLowerlo, xSLowerlo);
        __m256 yTLowerhi = _mm256_mul_ps(xSLowerhi, xSLowerhi);
        __m256 yTLeftlo  = _mm256_mul_ps(xSLeftlo,  xSLeftlo);
        __m256 yTLefthi  = _mm256_mul_ps(xSLefthi,  xSLefthi);
        __m256 yTRightlo = _mm256_mul_ps(xSRightlo, xSRightlo);
        __m256 yTRighthi = _mm256_mul_ps(xSRighthi, xSRighthi);

        yTUpperlo = _mm256_fmadd_ps(yTUpperlo, yInvThreshold2, yOnef);
        yTUpperhi = _mm256_fmadd_ps(yTUpperhi, yInvThreshold2, yOnef);
        yTLowerlo = _mm256_fmadd_ps(yTLowerlo, yInvThreshold2, yOnef);
        yTLowerhi = _mm256_fmadd_ps(yTLowerhi, yInvThreshold2, yOnef);
        yTLeftlo  = _mm256_fmadd_ps(yTLeftlo,  yInvThreshold2, yOnef);
        yTLefthi  = _mm256_fmadd_ps(yTLefthi,  yInvThreshold2, yOnef);
        yTRightlo = _mm256_fmadd_ps(yTRightlo, yInvThreshold2, yOnef);
        yTRighthi = _mm256_fmadd_ps(yTRighthi, yInvThreshold2, yOnef);

#if 0
        yTUpperlo = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTUpperlo));
        yTUpperhi = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTUpperhi));
        yTLowerlo = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTLowerlo));
        yTLowerhi = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTLowerhi));
        yTLeftlo  = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTLeftlo));
        yTLefthi  = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTLefthi));
        yTRightlo = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTRightlo));
        yTRighthi = _mm256_mul_ps(yStrength2, _mm256_rcp_ps_hp(yTRighthi));
#else
        yTUpperlo = _mm256_div_ps(yStrength2, yTUpperlo);
        yTUpperhi = _mm256_div_ps(yStrength2, yTUpperhi);
        yTLowerlo = _mm256_div_ps(yStrength2, yTLowerlo);
        yTLowerhi = _mm256_div_ps(yStrength2, yTLowerhi);
        yTLeftlo  = _mm256_div_ps(yStrength2, yTLeftlo);
        yTLefthi  = _mm256_div_ps(yStrength2, yTLefthi);
        yTRightlo = _mm256_div_ps(yStrength2, yTRightlo);
        yTRighthi = _mm256_div_ps(yStrength2, yTRighthi);
#endif

        __m256 yAddLo0, yAddHi0, yAddLo1, yAddHi1;
        yAddLo0   = _mm256_fmadd_ps(xSLowerlo, yTLowerlo, _mm256_mul_ps(xSUpperlo, yTUpperlo));
        yAddHi0   = _mm256_fmadd_ps(xSLowerhi, yTLowerhi, _mm256_mul_ps(xSUpperhi, yTUpperhi));
        yAddLo1   = _mm256_fmadd_ps(xSRightlo, yTRightlo, _mm256_mul_ps(xSLeftlo, yTLeftlo));
        yAddHi1   = _mm256_fmadd_ps(xSRighthi, yTRighthi, _mm256_mul_ps(xSLefthi, yTLefthi));

        yAddLo0 = _mm256_add_ps(yAddLo0, yAddLo1);
        yAddHi0 = _mm256_add_ps(yAddHi0, yAddHi1);

        __m256i ySrc = _mm256_loadu_si256((__m256i *)(src));
#define _mm256_stream_storeu_switch_si256(x, ymm) ((use_stream) ? _mm256_stream_si256((x), (ymm)) : _mm256_storeu_si256((x), (ymm)))
        _mm256_stream_storeu_switch_si256((__m256i *)(dst), _mm256_add_epi16(ySrc, _mm256_packs_epi32(_mm256_cvtps_epi32(yAddLo0), _mm256_cvtps_epi32(yAddHi0))));
#undef _mm256_stream_storeu_switch_si256
    }
}

void anisotropic_mt_avx2_fma3(int thread_id, int thread_num, void *param1, void *param2) {
    avx2_dummy();
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

    __m256 yInvThreshold2 = _mm256_set1_ps(inv_threshold2);
    __m256 yStrength2 = _mm256_set1_ps(strength2 / range);
    __m256 yOnef = _mm256_set1_ps(1.0f);

    //最初の行はそのままコピー
    if (0 == y_start) {
        memcpy_avx2<false, false, false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
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
        const size_t dst_mod32 = (int)((size_t)dst & 0x1f);
        if (dst_mod32) {
            int dw = 32 - dst_mod32;
            anisotropic_mt_avx2_line<false>(dst, src, dw, max_w, yInvThreshold2, yStrength2, yOnef);
            src += dw; dst += dw; process_size_in_byte -= dw;
        }
        anisotropic_mt_avx2_line<true>(dst, src, process_size_in_byte & (~0x1f), max_w, yInvThreshold2, yStrength2, yOnef);
        if (process_size_in_byte & 0x1f) {
            src += process_size_in_byte - 32;
            dst += process_size_in_byte - 32;
            anisotropic_mt_avx2_line<false>(dst, src, 32, max_w, yInvThreshold2, yStrength2, yOnef);
        }
        //先端と終端をそのままコピー
        *(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
        *(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
    }
    //最後の行はそのままコピー
    if (h-1 == y_fin) {
        memcpy_avx2<false, false, false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
    }
    _mm256_zeroupper();
}

void anisotropic_mt_exp_avx2(int thread_id, int thread_num, void *param1, void *param2) {
    anisotropic_mt_exp_avx2_base(thread_id, thread_num, param1, param2);
}
