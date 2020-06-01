#pragma once
#include <cstdint>
#include <cmath>
#include "pmd_mt.h"
#include "filter.h"
#include "simd_util.h"

void avx2_dummy();

//---------------------------------------------------------------------
//        修正PDMマルチスレッド関数
//---------------------------------------------------------------------

static __forceinline void getDiff(uint8_t *src, int max_w, __m256i& xUpper, __m256i& xLower, __m256i& xLeft, __m256i& xRight) {
    __m128i xSrc0, xSrc1, xSrc2;
    xSrc0 = _mm_loadu_si128((__m128i *)(src - sizeof(PIXEL_YC) +  0));
    xSrc1 = _mm_loadu_si128((__m128i *)(src - sizeof(PIXEL_YC) + 16));
    xSrc2 = _mm_loadu_si128((__m128i *)(src - sizeof(PIXEL_YC) + 32));

    __m256i ySrc0, ySrc1, ySrc;
    ySrc0 = _mm256_set_m128i(xSrc1, xSrc0);
    ySrc1 = _mm256_set_m128i(xSrc2, xSrc1);
    ySrc = _mm256_alignr_epi8(ySrc1, ySrc0, 6);

    xUpper = _mm256_subs_epi16(_mm256_loadu_si256((__m256i *)(src - max_w * sizeof(PIXEL_YC))), ySrc);
    xLower = _mm256_subs_epi16(_mm256_loadu_si256((__m256i *)(src + max_w * sizeof(PIXEL_YC))), ySrc);
    xLeft  = _mm256_subs_epi16(ySrc0, ySrc);
    xRight = _mm256_subs_epi16(_mm256_alignr_epi8(ySrc1, ySrc0, 12), ySrc);
}

static __forceinline void pmd_mt_exp_avx2_base(int thread_id, int thread_num, void *param1, void *param2) {
    avx2_dummy();
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

    __m256 yTempStrength2 = _mm256_set1_ps(strength2 * (1.0f / range));
    __m256 yMinusInvThreshold2 = _mm256_set1_ps(-1.0f * inv_threshold2);
#else
    int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;
#endif

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

#if !USE_VPGATHER && !USE_FMATH
    __declspec(align(32)) int16_t diffBuf[64];
    __declspec(align(32)) int expBuf[64];
#endif
    __m256i yPMDBufLimit = _mm256_set1_epi16(PMD_TABLE_SIZE-1);

    for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC), gau_line += max_w * sizeof(PIXEL_YC)) {
        uint8_t *src = src_line;
        uint8_t *dst = dst_line;
        uint8_t *gau = gau_line;

        //まずは、先端終端ピクセルを気にせず普通に処理してしまう
        //先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
        //最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
        //先端終端ピクセルは後から上書きコピーする
        uint8_t *src_fin = src + w * sizeof(PIXEL_YC);
        for ( ; src < src_fin; src += 32, dst += 32, gau += 32) {
            __m256i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
            __m256i yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff;
            getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);
            getDiff(gau, max_w, yGauUpperDiff, yGauLowerDiff, yGauLeftDiff, yGauRightDiff);
#if USE_FMATH
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

            yGUpperlo = _mm256_mul_ps(yGUpperlo, yMinusInvThreshold2);
            yGUpperhi = _mm256_mul_ps(yGUpperhi, yMinusInvThreshold2);
            yGLowerlo = _mm256_mul_ps(yGLowerlo, yMinusInvThreshold2);
            yGLowerhi = _mm256_mul_ps(yGLowerhi, yMinusInvThreshold2);
            yGLeftlo  = _mm256_mul_ps(yGLeftlo,  yMinusInvThreshold2);
            yGLefthi  = _mm256_mul_ps(yGLefthi,  yMinusInvThreshold2);
            yGRightlo = _mm256_mul_ps(yGRightlo, yMinusInvThreshold2);
            yGRighthi = _mm256_mul_ps(yGRighthi, yMinusInvThreshold2);

            yGUpperlo = fmath::exp_ps256(yGUpperlo);
            yGUpperhi = fmath::exp_ps256(yGUpperhi);
            yGLowerlo = fmath::exp_ps256(yGLowerlo);
            yGLowerhi = fmath::exp_ps256(yGLowerhi);
            yGLeftlo  = fmath::exp_ps256(yGLeftlo);
            yGLefthi  = fmath::exp_ps256(yGLefthi);
            yGRightlo = fmath::exp_ps256(yGRightlo);
            yGRighthi = fmath::exp_ps256(yGRighthi);

            yGUpperlo = _mm256_mul_ps(yGUpperlo, yTempStrength2);
            yGUpperhi = _mm256_mul_ps(yGUpperhi, yTempStrength2);
            yGLowerlo = _mm256_mul_ps(yGLowerlo, yTempStrength2);
            yGLowerhi = _mm256_mul_ps(yGLowerhi, yTempStrength2);
            yGLeftlo  = _mm256_mul_ps(yGLeftlo,  yTempStrength2);
            yGLefthi  = _mm256_mul_ps(yGLefthi,  yTempStrength2);
            yGRightlo = _mm256_mul_ps(yGRightlo, yTempStrength2);
            yGRighthi = _mm256_mul_ps(yGRighthi, yTempStrength2);

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
            _mm256_storeu_si256((__m256i *)(dst), _mm256_add_epi16(ySrc, _mm256_packs_epi32(_mm256_cvtps_epi32(yAddLo0), _mm256_cvtps_epi32(yAddHi0))));
#else
            yGauUpperDiff = _mm256_abs_epi16(yGauUpperDiff);
            yGauLowerDiff = _mm256_abs_epi16(yGauLowerDiff);
            yGauLeftDiff  = _mm256_abs_epi16(yGauLeftDiff);
            yGauRightDiff = _mm256_abs_epi16(yGauRightDiff);

            yGauUpperDiff = _mm256_min_epi16(yGauUpperDiff, yPMDBufLimit);
            yGauLowerDiff = _mm256_min_epi16(yGauLowerDiff, yPMDBufLimit);
            yGauLeftDiff  = _mm256_min_epi16(yGauLeftDiff,  yPMDBufLimit);
            yGauRightDiff = _mm256_min_epi16(yGauRightDiff, yPMDBufLimit);
#if USE_VPGATHER
            __m256i yEUpperlo = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(yGauUpperDiff), 4);
            __m256i yEUpperhi = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(yGauUpperDiff), 4);
            __m256i yELowerlo = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(yGauLowerDiff), 4);
            __m256i yELowerhi = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(yGauLowerDiff), 4);
            __m256i yELeftlo  = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(yGauLeftDiff),  4);
            __m256i yELefthi  = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(yGauLeftDiff),  4);
            __m256i yERightlo = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(yGauRightDiff), 4);
            __m256i yERighthi = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(yGauRightDiff), 4);
#else
            _mm256_store_si256((__m256i *)(diffBuf +  0), yGauUpperDiff);
            _mm256_store_si256((__m256i *)(diffBuf + 16), yGauLowerDiff);
            _mm256_store_si256((__m256i *)(diffBuf + 32), yGauLeftDiff);
            _mm256_store_si256((__m256i *)(diffBuf + 48), yGauRightDiff);

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

            __m256i yEUpperlo = _mm256_load_si256((__m256i *)(expBuf +  0));
            __m256i yEUpperhi = _mm256_load_si256((__m256i *)(expBuf +  8));
            __m256i yELowerlo = _mm256_load_si256((__m256i *)(expBuf + 16));
            __m256i yELowerhi = _mm256_load_si256((__m256i *)(expBuf + 24));
            __m256i yELeftlo  = _mm256_load_si256((__m256i *)(expBuf + 32));
            __m256i yELefthi  = _mm256_load_si256((__m256i *)(expBuf + 40));
            __m256i yERightlo = _mm256_load_si256((__m256i *)(expBuf + 48));
            __m256i yERighthi = _mm256_load_si256((__m256i *)(expBuf + 56));
#endif
#if 1 //こちらの積算の少ないほうが高速
            __m256i yELULo = _mm256_blend_epi16(yELowerlo, _mm256_slli_epi32(yEUpperlo, 16), 0xAA);
            __m256i yELUHi = _mm256_blend_epi16(yELowerhi, _mm256_slli_epi32(yEUpperhi, 16), 0xAA);
            __m256i yERLLo = _mm256_blend_epi16(yERightlo, _mm256_slli_epi32(yELeftlo, 16),  0xAA);
            __m256i yERLHi = _mm256_blend_epi16(yERighthi, _mm256_slli_epi32(yELefthi, 16),  0xAA);

            __m256i yAddLo0 = _mm256_madd_epi16(yELULo, _mm256_unpacklo_epi16(ySrcLowerDiff, ySrcUpperDiff));
            __m256i yAddHi0 = _mm256_madd_epi16(yELUHi, _mm256_unpackhi_epi16(ySrcLowerDiff, ySrcUpperDiff));
            __m256i yAddLo1 = _mm256_madd_epi16(yERLLo, _mm256_unpacklo_epi16(ySrcRightDiff, ySrcLeftDiff));
            __m256i yAddHi1 = _mm256_madd_epi16(yERLHi, _mm256_unpackhi_epi16(ySrcRightDiff, ySrcLeftDiff));
            __m256i yAddLo = _mm256_add_epi32(yAddLo0, yAddLo1);
            __m256i yAddHi = _mm256_add_epi32(yAddHi0, yAddHi1);
#else
            yEUpperlo = _mm256_mullo_epi32(yEUpperlo, cvtlo256_epi16_epi32(ySrcUpperDiff));
            yEUpperhi = _mm256_mullo_epi32(yEUpperhi, cvthi256_epi16_epi32(ySrcUpperDiff));
            yELowerlo = _mm256_mullo_epi32(yELowerlo, cvtlo256_epi16_epi32(ySrcLowerDiff));
            yELowerhi = _mm256_mullo_epi32(yELowerhi, cvthi256_epi16_epi32(ySrcLowerDiff));
            yELeftlo  = _mm256_mullo_epi32(yELeftlo,  cvtlo256_epi16_epi32(ySrcLeftDiff));
            yELefthi  = _mm256_mullo_epi32(yELefthi,  cvthi256_epi16_epi32(ySrcLeftDiff));
            yERightlo = _mm256_mullo_epi32(yERightlo, cvtlo256_epi16_epi32(ySrcRightDiff));
            yERighthi = _mm256_mullo_epi32(yERighthi, cvthi256_epi16_epi32(ySrcRightDiff));

            __m256i yAddLo, yAddHi;
            yAddLo = yEUpperlo;
            yAddHi = yEUpperhi;
            yAddLo = _mm256_add_epi32(yAddLo, yELowerlo);
            yAddHi = _mm256_add_epi32(yAddHi, yELowerhi);
            yAddLo = _mm256_add_epi32(yAddLo, yELeftlo);
            yAddHi = _mm256_add_epi32(yAddHi, yELefthi);
            yAddLo = _mm256_add_epi32(yAddLo, yERightlo);
            yAddHi = _mm256_add_epi32(yAddHi, yERighthi);
#endif

            __m256i ySrc = _mm256_loadu_si256((__m256i *)(src));
            _mm256_storeu_si256((__m256i *)(dst), _mm256_add_epi16(ySrc, _mm256_packs_epi32(_mm256_srai_epi32(yAddLo, 16), _mm256_srai_epi32(yAddHi, 16))));
#endif
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

static __forceinline void anisotropic_mt_exp_avx2_base(int thread_id, int thread_num, void *param1, void *param2) {
    avx2_dummy();
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    const int w = fpip->w;
    const int h = fpip->h;
    const int max_w = fpip->max_w;
    int y_start = h *  thread_id    / thread_num;
    int y_fin   = h * (thread_id+1) / thread_num;

    //最初の行はそのままコピー
    if (0 == y_start) {
        memcpy_avx2<false, false, false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
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

    __m256 yMinusInvThreshold2 = _mm256_set1_ps(-1.0f * inv_threshold2);
    __m256 yStrength2 = _mm256_set1_ps(strength2 / range);
#else
    int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;
#endif

#if !USE_VPGATHER && !USE_FMATH
    __declspec(align(32)) int16_t diffBuf[64];
    __declspec(align(32)) int expBuf[64];
#endif
    __m256i yPMDBufMaxLimit = _mm256_set1_epi16(PMD_TABLE_SIZE-1);
    __m256i yPMDBufMinLimit = _mm256_set1_epi16(-PMD_TABLE_SIZE);

    for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC)) {
        uint8_t *src = src_line;
        uint8_t *dst = dst_line;

        //まずは、先端終端ピクセルを気にせず普通に処理してしまう
        //先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
        //最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
        //先端終端ピクセルは後から上書きコピーする
        uint8_t *src_fin = src + w * sizeof(PIXEL_YC);
        for ( ; src < src_fin; src += 32, dst += 32) {
            __m256i ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff;
            getDiff(src, max_w, ySrcUpperDiff, ySrcLowerDiff, ySrcLeftDiff, ySrcRightDiff);
#if USE_FMATH
            __m256 ySUpperlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcUpperDiff));
            __m256 ySUpperhi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcUpperDiff));
            __m256 ySLowerlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcLowerDiff));
            __m256 ySLowerhi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcLowerDiff));
            __m256 ySLeftlo  = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcLeftDiff));
            __m256 ySLefthi  = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcLeftDiff));
            __m256 ySRightlo = _mm256_cvtepi32_ps(cvtlo256_epi16_epi32(ySrcRightDiff));
            __m256 ySRighthi = _mm256_cvtepi32_ps(cvthi256_epi16_epi32(ySrcRightDiff));

            __m256 yTUpperlo = _mm256_mul_ps(ySUpperlo, ySUpperlo);
            __m256 yTUpperhi = _mm256_mul_ps(ySUpperhi, ySUpperhi);
            __m256 yTLowerlo = _mm256_mul_ps(ySLowerlo, ySLowerlo);
            __m256 yTLowerhi = _mm256_mul_ps(ySLowerhi, ySLowerhi);
            __m256 yTLeftlo  = _mm256_mul_ps(ySLeftlo,  ySLeftlo);
            __m256 yTLefthi  = _mm256_mul_ps(ySLefthi,  ySLefthi);
            __m256 yTRightlo = _mm256_mul_ps(ySRightlo, ySRightlo);
            __m256 yTRighthi = _mm256_mul_ps(ySRighthi, ySRighthi);

            yTUpperlo = _mm256_mul_ps(ySUpperlo, yMinusInvThreshold2);
            yTUpperhi = _mm256_mul_ps(ySUpperhi, yMinusInvThreshold2);
            yTLowerlo = _mm256_mul_ps(ySLowerlo, yMinusInvThreshold2);
            yTLowerhi = _mm256_mul_ps(ySLowerhi, yMinusInvThreshold2);
            yTLeftlo  = _mm256_mul_ps(ySLeftlo,  yMinusInvThreshold2);
            yTLefthi  = _mm256_mul_ps(ySLefthi,  yMinusInvThreshold2);
            yTRightlo = _mm256_mul_ps(ySRightlo, yMinusInvThreshold2);
            yTRighthi = _mm256_mul_ps(ySRighthi, yMinusInvThreshold2);

            yTUpperlo = fmath::exp_ps256(yTUpperlo);
            yTUpperhi = fmath::exp_ps256(yTUpperhi);
            yTLowerlo = fmath::exp_ps256(yTLowerlo);
            yTLowerhi = fmath::exp_ps256(yTLowerhi);
            yTLeftlo  = fmath::exp_ps256(yTLeftlo);
            yTLefthi  = fmath::exp_ps256(yTLefthi);
            yTRightlo = fmath::exp_ps256(yTRightlo);
            yTRighthi = fmath::exp_ps256(yTRighthi);

            yTUpperlo = _mm256_mul_ps(yStrength2, yTUpperlo);
            yTUpperhi = _mm256_mul_ps(yStrength2, yTUpperhi);
            yTLowerlo = _mm256_mul_ps(yStrength2, yTLowerlo);
            yTLowerhi = _mm256_mul_ps(yStrength2, yTLowerhi);
            yTLeftlo  = _mm256_mul_ps(yStrength2, yTLeftlo);
            yTLefthi  = _mm256_mul_ps(yStrength2, yTLefthi);
            yTRightlo = _mm256_mul_ps(yStrength2, yTRightlo);
            yTRighthi = _mm256_mul_ps(yStrength2, yTRighthi);

            __m256 yAddLo0, yAddHi0, yAddLo1, yAddHi1;
            yAddLo0   = _mm256_fmadd_ps(ySLowerlo, yTLowerlo, _mm256_mul_ps(ySUpperlo, yTUpperlo));
            yAddHi0   = _mm256_fmadd_ps(ySLowerhi, yTLowerhi, _mm256_mul_ps(ySUpperhi, yTUpperhi));
            yAddLo1   = _mm256_fmadd_ps(ySRightlo, yTRightlo, _mm256_mul_ps(ySLeftlo, yTLeftlo));
            yAddHi1   = _mm256_fmadd_ps(ySRighthi, yTRighthi, _mm256_mul_ps(ySLefthi, yTLefthi));

            yAddLo0 = _mm256_add_ps(yAddLo0, yAddLo1);
            yAddHi0 = _mm256_add_ps(yAddHi0, yAddHi1);

            __m256i ySrc = _mm256_loadu_si256((__m256i *)(src));
            _mm256_storeu_si256((__m256i *)(dst), _mm256_add_epi16(ySrc, _mm256_packs_epi32(_mm256_cvtps_epi32(yAddLo0), _mm256_cvtps_epi32(yAddHi0))));
#else
            ySrcUpperDiff = _mm256_max_epi16(ySrcUpperDiff, yPMDBufMinLimit);
            ySrcLowerDiff = _mm256_max_epi16(ySrcLowerDiff, yPMDBufMinLimit);
            ySrcLeftDiff  = _mm256_max_epi16(ySrcLeftDiff,  yPMDBufMinLimit);
            ySrcRightDiff = _mm256_max_epi16(ySrcRightDiff, yPMDBufMinLimit);

            ySrcUpperDiff = _mm256_min_epi16(ySrcUpperDiff, yPMDBufMaxLimit);
            ySrcLowerDiff = _mm256_min_epi16(ySrcLowerDiff, yPMDBufMaxLimit);
            ySrcLeftDiff  = _mm256_min_epi16(ySrcLeftDiff,  yPMDBufMaxLimit);
            ySrcRightDiff = _mm256_min_epi16(ySrcRightDiff, yPMDBufMaxLimit);
#if USE_VPGATHER
            __m256i yEUpperlo = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(ySrcUpperDiff), 4);
            __m256i yEUpperhi = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(ySrcUpperDiff), 4);
            __m256i yELowerlo = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(ySrcLowerDiff), 4);
            __m256i yELowerhi = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(ySrcLowerDiff), 4);
            __m256i yELeftlo  = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(ySrcLeftDiff),  4);
            __m256i yELefthi  = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(ySrcLeftDiff),  4);
            __m256i yERightlo = _mm256_i32gather_epi32(pmdp, cvtlo256_epi16_epi32(ySrcRightDiff), 4);
            __m256i yERighthi = _mm256_i32gather_epi32(pmdp, cvthi256_epi16_epi32(ySrcRightDiff), 4);
#else
            _mm256_store_si256((__m256i *)(diffBuf +  0), ySrcUpperDiff);
            _mm256_store_si256((__m256i *)(diffBuf + 16), ySrcLowerDiff);
            _mm256_store_si256((__m256i *)(diffBuf + 32), ySrcLeftDiff);
            _mm256_store_si256((__m256i *)(diffBuf + 48), ySrcRightDiff);

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

            __m256i yEUpperlo = _mm256_load_si256((__m256i *)(expBuf +  0));
            __m256i yEUpperhi = _mm256_load_si256((__m256i *)(expBuf +  8));
            __m256i yELowerlo = _mm256_load_si256((__m256i *)(expBuf + 16));
            __m256i yELowerhi = _mm256_load_si256((__m256i *)(expBuf + 24));
            __m256i yELeftlo  = _mm256_load_si256((__m256i *)(expBuf + 32));
            __m256i yELefthi  = _mm256_load_si256((__m256i *)(expBuf + 40));
            __m256i yERightlo = _mm256_load_si256((__m256i *)(expBuf + 48));
            __m256i yERighthi = _mm256_load_si256((__m256i *)(expBuf + 56));
#endif
            __m256i yAddLo, yAddHi;
            yAddLo = yEUpperlo;
            yAddHi = yEUpperhi;
            yAddLo = _mm256_add_epi32(yAddLo, yELowerlo);
            yAddHi = _mm256_add_epi32(yAddHi, yELowerhi);
            yAddLo = _mm256_add_epi32(yAddLo, yELeftlo);
            yAddHi = _mm256_add_epi32(yAddHi, yELefthi);
            yAddLo = _mm256_add_epi32(yAddLo, yERightlo);
            yAddHi = _mm256_add_epi32(yAddHi, yERighthi);

            __m256i ySrc = _mm256_loadu_si256((__m256i *)(src));
            _mm256_storeu_si256((__m256i *)(dst), _mm256_add_epi16(ySrc, _mm256_packs_epi32(yAddLo, yAddHi)));
#endif
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
