#pragma once

#include <cstdint>
#include <cmath>

#include "pmd_mt.h"
#include "simd_util.h"

// x0*1 + x1*4 + x2*6 + x3*4 + x4*1
static __forceinline __m128i gaussian_1_4_6_4_1(__m128i x0, __m128i x1, __m128i x2, const __m128i& x3, const __m128i& x4) {
	static const __declspec(align(16)) short MULTIPLIZER[8] = { 4, 6, 4, 6, 4, 6, 4, 6 };
	x0 = _mm_adds_epi16(x0, x4);
	x1 = _mm_adds_epi16(x1, x3);
	
	__m128i x0_lower = cvtlo_epi16_epi32(x0);
	__m128i x0_upper = cvthi_epi16_epi32(x0);
	__m128i x1_lower = _mm_madd_epi16(_mm_unpacklo_epi16(x1, x2), _mm_load_si128((__m128i *)MULTIPLIZER));
	__m128i x1_upper = _mm_madd_epi16(_mm_unpackhi_epi16(x1, x2), _mm_load_si128((__m128i *)MULTIPLIZER));

	x0_lower = _mm_add_epi32(x0_lower, x1_lower);
	x0_upper = _mm_add_epi32(x0_upper, x1_upper);
	x0_lower = _mm_srai_epi32(x0_lower, 4);
	x0_upper = _mm_srai_epi32(x0_upper, 4);

	return _mm_packs_epi32(x0_lower, x0_upper);
}

static __forceinline void set_temp_buffer(int i_dst, int i_src, uint8_t *src_top, uint8_t *temp, int analyze_block, int max_w, int tmp_line_size) {
	uint8_t *src_ptr = src_top + i_src * max_w * sizeof(PIXEL_YC);
	uint8_t *tmp_ptr = temp + i_dst * tmp_line_size;
	uint8_t *temp_fin = tmp_ptr + analyze_block * sizeof(PIXEL_YC);
	__m128i x0, x1, x2, x3;
	for (; tmp_ptr < temp_fin; tmp_ptr += 64, src_ptr += 64) {
		x0 = _mm_loadu_si128((const __m128i*)(src_ptr +  0));
		x1 = _mm_loadu_si128((const __m128i*)(src_ptr + 16));
		x2 = _mm_loadu_si128((const __m128i*)(src_ptr + 32));
		x3 = _mm_loadu_si128((const __m128i*)(src_ptr + 48));
		_mm_store_si128((__m128i*)(tmp_ptr +  0), x0);
		_mm_store_si128((__m128i*)(tmp_ptr + 16), x1);
		_mm_store_si128((__m128i*)(tmp_ptr + 32), x2);
		_mm_store_si128((__m128i*)(tmp_ptr + 48), x3);
	}
}

static __forceinline void gaussianH_simd(int thread_id, int thread_num, void *param1, void *param2) {
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
	const int max_w = fpip->max_w;
	const int w     = fpip->w;
	const int h     = fpip->h;
	PIXEL_YC *ycp_src = fpip->ycp_edit;
	PIXEL_YC *ycp_buf = (PIXEL_YC *)param2;
	const int min_analyze_cycle = 8;
	const int max_block_size = 256;
	const int tmp_line_size = max_block_size * sizeof(PIXEL_YC);
	const int x_start = ((int)(w * thread_id / (double)thread_num + 0.5) + (min_analyze_cycle-1)) & ~(min_analyze_cycle-1);
	const int x_fin = ((int)(w * (thread_id+1) / (double)thread_num + 0.5) + (min_analyze_cycle-1)) & ~(min_analyze_cycle-1);
	__declspec(align(16)) uint8_t temp[tmp_line_size * 4];
	int analyze_block = max_block_size;
	uint8_t *src_top = (uint8_t *)ycp_src + x_start * sizeof(PIXEL_YC);
	uint8_t *buf_top = (uint8_t *)ycp_buf + x_start * sizeof(PIXEL_YC);

	for (int pos_x = x_start; pos_x < x_fin; pos_x += analyze_block, src_top += analyze_block * sizeof(PIXEL_YC), buf_top += analyze_block * sizeof(PIXEL_YC)) {
		analyze_block = min(x_fin - pos_x, max_block_size);
		//一時領域にデータを満たす
		set_temp_buffer(0, 0, src_top, temp, analyze_block, max_w, tmp_line_size);
		set_temp_buffer(1, 0, src_top, temp, analyze_block, max_w, tmp_line_size);
		set_temp_buffer(2, 0, src_top, temp, analyze_block, max_w, tmp_line_size);
		set_temp_buffer(3, 1, src_top, temp, analyze_block, max_w, tmp_line_size);

		uint8_t *buf_line = buf_top;
		for (int y = 0; y < h; y++, buf_line += max_w * sizeof(PIXEL_YC)) {
			uint8_t *src = src_top + min(y, h-3) * max_w * sizeof(PIXEL_YC);
			uint8_t *buf = buf_line;
			uint8_t *tmp = temp;
			for (int x = 0; x < analyze_block; x += 8, src += 48, buf += 48, tmp += 48) {
				auto tmp_ptr = [tmp, tmp_line_size](int y) { return tmp + (y&3)*tmp_line_size; };
				auto guassianH_process_internal =[&](int i) {
					__m128i xSrc0 = _mm_loadu_si128((__m128i *)(src + 2 * max_w * sizeof(PIXEL_YC) + (i<<4)));
				
					__m128i x0 = _mm_load_si128((__m128i *)(tmp_ptr(y+0) + (i<<4)));
					__m128i x1 = _mm_load_si128((__m128i *)(tmp_ptr(y+1) + (i<<4)));
					__m128i x2 = _mm_load_si128((__m128i *)(tmp_ptr(y+2) + (i<<4)));
					__m128i x3 = _mm_load_si128((__m128i *)(tmp_ptr(y+3) + (i<<4)));
					__m128i x4 = xSrc0;
					_mm_storeu_si128((__m128i *)(buf + (i<<4)), gaussian_1_4_6_4_1(x0, x1, x2, x3, x4));

					_mm_store_si128((__m128i *)(tmp_ptr(y+4) + (i<<4)), xSrc0);
				};
				_mm_prefetch((char *)src + 3 * max_w * sizeof(PIXEL_YC), _MM_HINT_T0);
				guassianH_process_internal(0);
				guassianH_process_internal(1);
				guassianH_process_internal(2);
			}
		}
	}
}

// xSrc0   v-1, u-1, y-1, v-2, u-2, y-2,   x,   x
// xSrc1    u2,  y2,  v1,  u1,  y1,  v0,  u0,  y0
// xSrc2    y5,  v4,  u4,  y4,  v3,  u3,  y3,  v2
// xSrc3    v7,  u7,  y7,  v6,  u6,  y6,  v5,  u5
// xSrc4     x,   x,  v9,  u9,  y9,  v8,  u8,  y8

//dst <= 横方向にガウシアンのかかった YC48_0 --- YC48_7
static __forceinline void guassianV_process(uint8_t *dst, __m128i xSrc0, __m128i xSrc1, __m128i xSrc2, const __m128i& xSrc3, const __m128i& xSrc4) {
	__m128i xSum0, xSum1, xSum2, xSum3, xSum4;
	xSum0 = _mm_alignr_epi8_simd(xSrc1, xSrc0, (16-12));
	xSum1 = _mm_alignr_epi8_simd(xSrc1, xSrc0, (16- 6));
	xSum2 = xSrc1;
	xSum3 = _mm_alignr_epi8_simd(xSrc2, xSrc1,  6);
	xSum4 = _mm_alignr_epi8_simd(xSrc2, xSrc1, 12);

	_mm_storeu_si128((__m128i *)(dst +  0), gaussian_1_4_6_4_1(xSum0, xSum1, xSum2, xSum3, xSum4));

	xSum0 = _mm_alignr_epi8_simd(xSrc2, xSrc1, (16-12));
	xSum1 = _mm_alignr_epi8_simd(xSrc2, xSrc1, (16- 6));
	xSum2 = xSrc2;
	xSum3 = _mm_alignr_epi8_simd(xSrc3, xSrc2,  6);
	xSum4 = _mm_alignr_epi8_simd(xSrc3, xSrc2, 12);

	_mm_storeu_si128((__m128i *)(dst + 16), gaussian_1_4_6_4_1(xSum0, xSum1, xSum2, xSum3, xSum4));

	xSum0 = _mm_alignr_epi8_simd(xSrc3, xSrc2, (16-12));
	xSum1 = _mm_alignr_epi8_simd(xSrc3, xSrc2, (16- 6));
	xSum2 = xSrc3;
	xSum3 = _mm_alignr_epi8_simd(xSrc4, xSrc3,  6);
	xSum4 = _mm_alignr_epi8_simd(xSrc4, xSrc3, 12);

	_mm_storeu_si128((__m128i *)(dst + 32), gaussian_1_4_6_4_1(xSum0, xSum1, xSum2, xSum3, xSum4));
}

static __forceinline void gaussianV_simd(int thread_id, int thread_num, void *param1, void *param2) {
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
	const int max_w = fpip->max_w;
	const int w     = fpip->w;
	const int h     = fpip->h;
	PIXEL_YC *ycp_buf = fpip->ycp_temp;
	PIXEL_YC *ycp_dst = (PIXEL_YC *)param2;
	const int y_start = (h *  thread_id   ) / thread_num;
	const int y_fin   = (h * (thread_id+1)) / thread_num;
	uint8_t *buf_line = (uint8_t *)ycp_buf + max_w * y_start * sizeof(PIXEL_YC);
	uint8_t *dst_line = (uint8_t *)ycp_dst + max_w * y_start * sizeof(PIXEL_YC);

	for (int y = y_start; y < y_fin; y++, dst_line += max_w * sizeof(PIXEL_YC), buf_line += max_w * sizeof(PIXEL_YC)) {
		uint8_t *buf = buf_line;
		uint8_t *dst = dst_line;
		__m128i xSrc0, xSrc1, xSrc2, xSrc3, xSrc4;
		xSrc1 = _mm_loadu_si128((__m128i *)(buf +  0));
		xSrc0 = _mm_slli_si128(xSrc1, 16-6);
		xSrc0 = _mm_or_si128(xSrc0, _mm_srli_si128(xSrc0, 6));
		int x_cycles = 0;
		const int x_cycle_count = (w - 1) >> 3;
		for (; x_cycles < x_cycle_count; x_cycles++, dst += 48, buf += 48) {
			xSrc2 = _mm_loadu_si128((__m128i *)(buf + 16));
			xSrc3 = _mm_loadu_si128((__m128i *)(buf + 32));
			xSrc4 = _mm_loadu_si128((__m128i *)(buf + 48));

			guassianV_process(dst, xSrc0, xSrc1, xSrc2, xSrc3, xSrc4);

			xSrc0 = xSrc3;
			xSrc1 = xSrc4;
		}
		//終端
		const int last_dispalcement = (x_cycles << 3) + 8 - w;
		if (last_dispalcement) {
			dst -= last_dispalcement * sizeof(PIXEL_YC);
			buf -= last_dispalcement * sizeof(PIXEL_YC);
			xSrc0 = _mm_loadu_si128((__m128i *)(buf - 16));
			xSrc1 = _mm_loadu_si128((__m128i *)(buf +  0));
		}
		xSrc2 = _mm_loadu_si128((__m128i *)(buf + 16));
		xSrc3 = _mm_loadu_si128((__m128i *)(buf + 32));
		xSrc4 = _mm_srli_si128(xSrc3, 16-6);
		xSrc4 = _mm_or_si128(xSrc4, _mm_slli_si128(xSrc4, 6));
		guassianV_process(dst, xSrc0, xSrc1, xSrc2, xSrc3, xSrc4);
	}
}
//---------------------------------------------------------------------
//		修正PDMマルチスレッド関数
//---------------------------------------------------------------------

static __forceinline void getDiff(uint8_t *src, int max_w, __m128i& xUpper, __m128i& xLower, __m128i& xLeft, __m128i& xRight) {
	__m128i xSrc0, xSrc1, xSrc;
	xSrc0 = _mm_loadu_si128((__m128i *)(src - sizeof(PIXEL_YC) +  0));
	xSrc1 = _mm_loadu_si128((__m128i *)(src - sizeof(PIXEL_YC) + 16));

	xSrc = _mm_alignr_epi8_simd(xSrc1, xSrc0, 6);

	xUpper = _mm_subs_epi16(_mm_loadu_si128((__m128i *)(src - max_w * sizeof(PIXEL_YC))), xSrc);
	xLower = _mm_subs_epi16(_mm_loadu_si128((__m128i *)(src + max_w * sizeof(PIXEL_YC))), xSrc);
	xLeft  = _mm_subs_epi16(xSrc0, xSrc);
	xRight = _mm_subs_epi16(_mm_alignr_epi8_simd(xSrc1, xSrc0, 12), xSrc);
}

static __forceinline void pmd_mt_simd(int thread_id, int thread_num, void *param1, void *param2) {
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
	PIXEL_YC *gauss	= ((PMD_MT_PRM *)param2)->gauss;
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
	
#if USE_AVX
	__m256 yStrength2 = _mm256_set1_ps(strength2 / range);
	__m256 yInvThreshold2 = _mm256_set1_ps(inv_threshold2);
	__m256 yOnef = _mm256_set1_ps(1.0f);
#else
	__m128 xStrength2 = _mm_set1_ps(strength2 / range);
	__m128 xInvThreshold2 = _mm_set1_ps(inv_threshold2);
	__m128 xOnef = _mm_set1_ps(1.0f);
#endif

	//最初の行はそのままコピー
	if (0 == y_start) {
		memcpy_sse<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
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
		uint8_t *src_fin = src + w * sizeof(PIXEL_YC);
		for ( ; src < src_fin; src += 16, dst += 16, gau += 16) {
			__m128i xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff;
			__m128i xGauUpperDiff, xGauLowerDiff, xGauLeftDiff, xGauRightDiff;
			getDiff(src, max_w, xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff);
			getDiff(gau, max_w, xGauUpperDiff, xGauLowerDiff, xGauLeftDiff, xGauRightDiff);

			__m128 xGUpperlo = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xGauUpperDiff));
			__m128 xGUpperhi = _mm_cvtepi32_ps(cvthi_epi16_epi32(xGauUpperDiff));
			__m128 xGLowerlo = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xGauLowerDiff));
			__m128 xGLowerhi = _mm_cvtepi32_ps(cvthi_epi16_epi32(xGauLowerDiff));
			__m128 xGLeftlo  = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xGauLeftDiff));
			__m128 xGLefthi  = _mm_cvtepi32_ps(cvthi_epi16_epi32(xGauLeftDiff));
			__m128 xGRightlo = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xGauRightDiff));
			__m128 xGRighthi = _mm_cvtepi32_ps(cvthi_epi16_epi32(xGauRightDiff));
			
#if USE_AVX
			__m256 yGUpper = _mm256_set_m128(xGUpperhi, xGUpperlo);
			__m256 yGLower = _mm256_set_m128(xGLowerhi, xGLowerlo);
			__m256 yGLeft  = _mm256_set_m128(xGLefthi,  xGLeftlo);
			__m256 yGRight = _mm256_set_m128(xGRighthi, xGRightlo);

			yGUpper = _mm256_mul_ps(yGUpper, yGUpper);
			yGLower = _mm256_mul_ps(yGLower, yGLower);
			yGLeft  = _mm256_mul_ps(yGLeft,  yGLeft);
			yGRight = _mm256_mul_ps(yGRight, yGRight);

			yGUpper = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yGUpper, yInvThreshold2)));
			yGLower = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yGLower, yInvThreshold2)));
			yGLeft  = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yGLeft,  yInvThreshold2)));
			yGRight = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yGRight, yInvThreshold2)));

			yGUpper = _mm256_mul_ps(yStrength2, yGUpper);
			yGLower = _mm256_mul_ps(yStrength2, yGLower);
			yGLeft  = _mm256_mul_ps(yStrength2, yGLeft);
			yGRight = _mm256_mul_ps(yStrength2, yGRight);

			__m256 ySUpper = _mm256_set_m128(_mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcUpperDiff)), _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcUpperDiff)));
			__m256 ySLower = _mm256_set_m128(_mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcLowerDiff)), _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcLowerDiff)));
			__m256 ySLeft  = _mm256_set_m128(_mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcLeftDiff)),  _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcLeftDiff)));
			__m256 ySRight = _mm256_set_m128(_mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcRightDiff)), _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcRightDiff)));

			yGUpper = _mm256_mul_ps(yGUpper, ySUpper);
			yGLower = _mm256_mul_ps(yGLower, ySLower);
			yGLeft  = _mm256_mul_ps(yGLeft,  ySLeft);
			yGRight = _mm256_mul_ps(yGRight, ySRight);

			yGUpper = _mm256_add_ps(yGUpper, yGLower);
			yGLeft  = _mm256_add_ps(yGLeft, yGRight);
			yGUpper = _mm256_add_ps(yGUpper, yGLeft);
			
			__m128 xAddHi = _mm256_extractf128_ps(yGUpper, 1);
			__m128 xAddLo = _mm256_castps256_ps128(yGUpper);
#else
			xGUpperlo = _mm_mul_ps(xGUpperlo, xGUpperlo);
			xGUpperhi = _mm_mul_ps(xGUpperhi, xGUpperhi);
			xGLowerlo = _mm_mul_ps(xGLowerlo, xGLowerlo);
			xGLowerhi = _mm_mul_ps(xGLowerhi, xGLowerhi);
			xGLeftlo  = _mm_mul_ps(xGLeftlo,  xGLeftlo);
			xGLefthi  = _mm_mul_ps(xGLefthi,  xGLefthi);
			xGRightlo = _mm_mul_ps(xGRightlo, xGRightlo);
			xGRighthi = _mm_mul_ps(xGRighthi, xGRighthi);

			xGUpperlo = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGUpperlo, xInvThreshold2)));
			xGUpperhi = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGUpperhi, xInvThreshold2)));
			xGLowerlo = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGLowerlo, xInvThreshold2)));
			xGLowerhi = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGLowerhi, xInvThreshold2)));
			xGLeftlo  = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGLeftlo,  xInvThreshold2)));
			xGLefthi  = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGLefthi,  xInvThreshold2)));
			xGRightlo = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGRightlo, xInvThreshold2)));
			xGRighthi = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xGRighthi, xInvThreshold2)));

			xGUpperlo = _mm_mul_ps(xStrength2, xGUpperlo);
			xGUpperhi = _mm_mul_ps(xStrength2, xGUpperhi);
			xGLowerlo = _mm_mul_ps(xStrength2, xGLowerlo);
			xGLowerhi = _mm_mul_ps(xStrength2, xGLowerhi);
			xGLeftlo  = _mm_mul_ps(xStrength2, xGLeftlo);
			xGLefthi  = _mm_mul_ps(xStrength2, xGLefthi);
			xGRightlo = _mm_mul_ps(xStrength2, xGRightlo);
			xGRighthi = _mm_mul_ps(xStrength2, xGRighthi);

			xGUpperlo = _mm_mul_ps(xGUpperlo, _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcUpperDiff)));
			xGUpperhi = _mm_mul_ps(xGUpperhi, _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcUpperDiff)));
			xGLowerlo = _mm_mul_ps(xGLowerlo, _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcLowerDiff)));
			xGLowerhi = _mm_mul_ps(xGLowerhi, _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcLowerDiff)));
			xGLeftlo  = _mm_mul_ps(xGLeftlo,  _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcLeftDiff)));
			xGLefthi  = _mm_mul_ps(xGLefthi,  _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcLeftDiff)));
			xGRightlo = _mm_mul_ps(xGRightlo, _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcRightDiff)));
			xGRighthi = _mm_mul_ps(xGRighthi, _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcRightDiff)));

			__m128 xAddLo, xAddHi;
			xAddLo = xGUpperlo;
			xAddHi = xGUpperhi;
			xAddLo = _mm_add_ps(xAddLo, xGLowerlo);
			xAddHi = _mm_add_ps(xAddHi, xGLowerhi);
			xGLeftlo = _mm_add_ps(xGLeftlo, xGRightlo);
			xGLefthi = _mm_add_ps(xGLefthi, xGRighthi);
			xAddLo = _mm_add_ps(xAddLo, xGLeftlo);
			xAddHi = _mm_add_ps(xAddHi, xGLefthi);
#endif
			__m128i xSrc = _mm_loadu_si128((__m128i *)(src));
			_mm_storeu_si128((__m128i *)(dst), _mm_add_epi16(xSrc, _mm_packs_epi32(_mm_cvtps_epi32(xAddLo), _mm_cvtps_epi32(xAddHi))));
		}
		//先端と終端のピクセルをそのままコピー
		*(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
		*(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
	}
#if USE_AVX
	_mm256_zeroupper();
#endif
	//最後の行はそのままコピー
	if (h-1 == y_fin) {
		memcpy_sse<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
	}
}

static __forceinline void pmd_mt_exp_simd(int thread_id, int thread_num, void *param1, void *param2) {
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
	PIXEL_YC *gauss	= ((PMD_MT_PRM *)param2)->gauss;
	int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;
	const int w = fpip->w;
	const int h = fpip->h;
	const int max_w = fpip->max_w;
	int y_start = h *  thread_id    / thread_num;
	int y_fin   = h * (thread_id+1) / thread_num;
	
	//最初の行はそのままコピー
	if (0 == y_start) {
		memcpy_sse<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
		y_start++;
	}
	//最後の行はそのままコピー
	y_fin -= (h == y_fin);

	uint8_t *src_line = (uint8_t *)(fpip->ycp_edit + y_start * max_w);
	uint8_t *dst_line = (uint8_t *)(fpip->ycp_temp + y_start * max_w);
	uint8_t *gau_line = (uint8_t *)(gauss          + y_start * max_w);

	__declspec(align(16)) int16_t diffBuf[32];
	__declspec(align(16)) int expBuf[32];
	__m128i xPMDBufLimit = _mm_set1_epi16(PMD_TABLE_SIZE-1);

	for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC), gau_line += max_w * sizeof(PIXEL_YC)) {
		uint8_t *src = src_line;
		uint8_t *dst = dst_line;
		uint8_t *gau = gau_line;
		
		//まずは、先端終端ピクセルを気にせず普通に処理してしまう
		//先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
		//最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
		//先端終端ピクセルは後から上書きコピーする
		uint8_t *src_fin = src + w * sizeof(PIXEL_YC);
		for ( ; src < src_fin; src += 16, dst += 16, gau += 16) {
			__m128i xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff;
			__m128i xGauUpperDiff, xGauLowerDiff, xGauLeftDiff, xGauRightDiff;
			getDiff(src, max_w, xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff);
			getDiff(gau, max_w, xGauUpperDiff, xGauLowerDiff, xGauLeftDiff, xGauRightDiff);

			xGauUpperDiff = _mm_abs_epi16_simd(xGauUpperDiff);
			xGauLowerDiff = _mm_abs_epi16_simd(xGauLowerDiff);
			xGauLeftDiff  = _mm_abs_epi16_simd(xGauLeftDiff);
			xGauRightDiff = _mm_abs_epi16_simd(xGauRightDiff);

			xGauUpperDiff = _mm_min_epi16(xGauUpperDiff, xPMDBufLimit);
			xGauLowerDiff = _mm_min_epi16(xGauLowerDiff, xPMDBufLimit);
			xGauLeftDiff  = _mm_min_epi16(xGauLeftDiff,  xPMDBufLimit);
			xGauRightDiff = _mm_min_epi16(xGauRightDiff, xPMDBufLimit);

			_mm_store_si128((__m128i *)(diffBuf +  0), xGauUpperDiff);
			_mm_store_si128((__m128i *)(diffBuf +  8), xGauLowerDiff);
			_mm_store_si128((__m128i *)(diffBuf + 16), xGauLeftDiff);
			_mm_store_si128((__m128i *)(diffBuf + 24), xGauRightDiff);

			for (int i = 0; i < 32; i += 4) {
				expBuf[i+0] = pmdp[diffBuf[i+0]];
				expBuf[i+1] = pmdp[diffBuf[i+1]];
				expBuf[i+2] = pmdp[diffBuf[i+2]];
				expBuf[i+3] = pmdp[diffBuf[i+3]];
			}

			__m128i xEUpperlo = _mm_load_si128((__m128i *)(expBuf +  0));
			__m128i xEUpperhi = _mm_load_si128((__m128i *)(expBuf +  4));
			__m128i xELowerlo = _mm_load_si128((__m128i *)(expBuf +  8));
			__m128i xELowerhi = _mm_load_si128((__m128i *)(expBuf + 12));
			__m128i xELeftlo  = _mm_load_si128((__m128i *)(expBuf + 16));
			__m128i xELefthi  = _mm_load_si128((__m128i *)(expBuf + 20));
			__m128i xERightlo = _mm_load_si128((__m128i *)(expBuf + 24));
			__m128i xERighthi = _mm_load_si128((__m128i *)(expBuf + 28));

			xEUpperlo = _mm_mullo_epi32_simd(xEUpperlo, cvtlo_epi16_epi32(xSrcUpperDiff));
			xEUpperhi = _mm_mullo_epi32_simd(xEUpperhi, cvthi_epi16_epi32(xSrcUpperDiff));
			xELowerlo = _mm_mullo_epi32_simd(xELowerlo, cvtlo_epi16_epi32(xSrcLowerDiff));
			xELowerhi = _mm_mullo_epi32_simd(xELowerhi, cvthi_epi16_epi32(xSrcLowerDiff));
			xELeftlo  = _mm_mullo_epi32_simd(xELeftlo,  cvtlo_epi16_epi32(xSrcLeftDiff));
			xELefthi  = _mm_mullo_epi32_simd(xELefthi,  cvthi_epi16_epi32(xSrcLeftDiff));
			xERightlo = _mm_mullo_epi32_simd(xERightlo, cvtlo_epi16_epi32(xSrcRightDiff));
			xERighthi = _mm_mullo_epi32_simd(xERighthi, cvthi_epi16_epi32(xSrcRightDiff));

			__m128i xAddLo, xAddHi;
			xAddLo = xEUpperlo;
			xAddHi = xEUpperhi;
			xAddLo = _mm_add_epi32(xAddLo, xELowerlo);
			xAddHi = _mm_add_epi32(xAddHi, xELowerhi);
			xAddLo = _mm_add_epi32(xAddLo, xELeftlo);
			xAddHi = _mm_add_epi32(xAddHi, xELefthi);
			xAddLo = _mm_add_epi32(xAddLo, xERightlo);
			xAddHi = _mm_add_epi32(xAddHi, xERighthi);

			__m128i xSrc = _mm_loadu_si128((__m128i *)(src));
			_mm_storeu_si128((__m128i *)(dst), _mm_add_epi16(xSrc, _mm_packs_epi32(_mm_srai_epi32(xAddLo, 16), _mm_srai_epi32(xAddHi, 16))));
		}
		//先端と終端をそのままコピー
		*(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
		*(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
	}
	//最後の行はそのままコピー
	if (h-1 == y_fin) {
		memcpy_sse<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
	}
}

static __forceinline void anisotropic_mt_simd(int thread_id, int thread_num, void *param1, void *param2) {
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
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
	
#if USE_AVX
	__m256 yStrength2 = _mm256_set1_ps(strength2 / range);
	__m256 yInvThreshold2 = _mm256_set1_ps(inv_threshold2);
	__m256 yOnef = _mm256_set1_ps(1.0f);
#else
	__m128 xStrength2 = _mm_set1_ps(strength2 / range);
	__m128 xInvThreshold2 = _mm_set1_ps(inv_threshold2);
	__m128 xOnef = _mm_set1_ps(1.0f);
#endif
	
	//最初の行はそのままコピー
	if (0 == y_start) {
		memcpy_sse<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
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
		uint8_t *src_fin = src + w * sizeof(PIXEL_YC);
		for ( ; src < src_fin; src += 16, dst += 16) {
			__m128i xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff;
			getDiff(src, max_w, xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff);

			__m128 xSUpperlo = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcUpperDiff));
			__m128 xSUpperhi = _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcUpperDiff));
			__m128 xSLowerlo = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcLowerDiff));
			__m128 xSLowerhi = _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcLowerDiff));
			__m128 xSLeftlo  = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcLeftDiff));
			__m128 xSLefthi  = _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcLeftDiff));
			__m128 xSRightlo = _mm_cvtepi32_ps(cvtlo_epi16_epi32(xSrcRightDiff));
			__m128 xSRighthi = _mm_cvtepi32_ps(cvthi_epi16_epi32(xSrcRightDiff));
#if USE_AVX
			__m256 ySUpper = _mm256_set_m128(xSUpperhi, xSUpperlo);
			__m256 ySLower = _mm256_set_m128(xSLowerhi, xSLowerlo);
			__m256 ySLeft  = _mm256_set_m128(xSLefthi,  xSLeftlo);
			__m256 ySRight = _mm256_set_m128(xSRighthi, xSRightlo);

			__m256 yTUpper = _mm256_mul_ps(ySUpper, ySUpper);
			__m256 yTLower = _mm256_mul_ps(ySLower, ySLower);
			__m256 yTLeft  = _mm256_mul_ps(ySLeft,  ySLeft);
			__m256 yTRight = _mm256_mul_ps(ySRight, ySRight);

			yTUpper = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yTUpper, yInvThreshold2)));
			yTLower = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yTLower, yInvThreshold2)));
			yTLeft  = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yTLeft,  yInvThreshold2)));
			yTRight = _mm256_rcp_ps_hp(_mm256_add_ps(yOnef, _mm256_mul_ps(yTRight, yInvThreshold2)));

			yTUpper = _mm256_mul_ps(yStrength2, yTUpper);
			yTLower = _mm256_mul_ps(yStrength2, yTLower);
			yTLeft  = _mm256_mul_ps(yStrength2, yTLeft);
			yTRight = _mm256_mul_ps(yStrength2, yTRight);

			yTUpper = _mm256_mul_ps(ySUpper, yTUpper);
			yTLower = _mm256_mul_ps(ySLower, yTLower);
			yTLeft  = _mm256_mul_ps(ySLeft,  yTLeft);
			yTRight = _mm256_mul_ps(ySRight, yTRight);

			yTUpper = _mm256_add_ps(yTUpper, yTLower);
			yTLeft  = _mm256_add_ps(yTLeft,  yTRight);
			yTUpper = _mm256_add_ps(yTUpper, yTLeft);
			
			__m128 xAddHi = _mm256_extractf128_ps(yTUpper, 1);
			__m128 xAddLo = _mm256_castps256_ps128(yTUpper);
#else
			__m128 xTUpperlo = _mm_mul_ps(xSUpperlo, xSUpperlo);
			__m128 xTUpperhi = _mm_mul_ps(xSUpperhi, xSUpperhi);
			__m128 xTLowerlo = _mm_mul_ps(xSLowerlo, xSLowerlo);
			__m128 xTLowerhi = _mm_mul_ps(xSLowerhi, xSLowerhi);
			__m128 xTLeftlo  = _mm_mul_ps(xSLeftlo,  xSLeftlo);
			__m128 xTLefthi  = _mm_mul_ps(xSLefthi,  xSLefthi);
			__m128 xTRightlo = _mm_mul_ps(xSRightlo, xSRightlo);
			__m128 xTRighthi = _mm_mul_ps(xSRighthi, xSRighthi);

			xTUpperlo = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTUpperlo, xInvThreshold2)));
			xTUpperhi = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTUpperhi, xInvThreshold2)));
			xTLowerlo = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTLowerlo, xInvThreshold2)));
			xTLowerhi = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTLowerhi, xInvThreshold2)));
			xTLeftlo  = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTLeftlo,  xInvThreshold2)));
			xTLefthi  = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTLefthi,  xInvThreshold2)));
			xTRightlo = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTRightlo, xInvThreshold2)));
			xTRighthi = _mm_rcp_ps_hp(_mm_add_ps(xOnef, _mm_mul_ps(xTRighthi, xInvThreshold2)));

			xTUpperlo = _mm_mul_ps(xStrength2, xTUpperlo);
			xTUpperhi = _mm_mul_ps(xStrength2, xTUpperhi);
			xTLowerlo = _mm_mul_ps(xStrength2, xTLowerlo);
			xTLowerhi = _mm_mul_ps(xStrength2, xTLowerhi);
			xTLeftlo  = _mm_mul_ps(xStrength2, xTLeftlo);
			xTLefthi  = _mm_mul_ps(xStrength2, xTLefthi);
			xTRightlo = _mm_mul_ps(xStrength2, xTRightlo);
			xTRighthi = _mm_mul_ps(xStrength2, xTRighthi);

			xTUpperlo = _mm_mul_ps(xSUpperlo, xTUpperlo);
			xTUpperhi = _mm_mul_ps(xSUpperhi, xTUpperhi);
			xTLowerlo = _mm_mul_ps(xSLowerlo, xTLowerlo);
			xTLowerhi = _mm_mul_ps(xSLowerhi, xTLowerhi);
			xTLeftlo  = _mm_mul_ps(xSLeftlo,  xTLeftlo);
			xTLefthi  = _mm_mul_ps(xSLefthi,  xTLefthi);
			xTRightlo = _mm_mul_ps(xSRightlo, xTRightlo);
			xTRighthi = _mm_mul_ps(xSRighthi, xTRighthi);

			__m128 xAddLo, xAddHi;
			xAddLo = xTUpperlo;
			xAddHi = xTUpperhi;
			xAddLo = _mm_add_ps(xAddLo, xTLowerlo);
			xAddHi = _mm_add_ps(xAddHi, xTLowerhi);
			xTLeftlo = _mm_add_ps(xTLeftlo, xTRightlo);
			xTLefthi = _mm_add_ps(xTLefthi, xTRighthi);
			xAddLo = _mm_add_ps(xAddLo, xTLeftlo);
			xAddHi = _mm_add_ps(xAddHi, xTLefthi);
#endif
			__m128i xSrc = _mm_loadu_si128((__m128i *)(src));
			_mm_storeu_si128((__m128i *)(dst), _mm_add_epi16(xSrc, _mm_packs_epi32(_mm_cvtps_epi32(xAddLo), _mm_cvtps_epi32(xAddHi))));
		}
		//先端と終端をそのままコピー
		*(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
		*(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
	}
	//最後の行はそのままコピー
	if (h-1 == y_fin) {
		memcpy_sse<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
	}
}

static __forceinline void anisotropic_mt_exp_simd(int thread_id, int thread_num, void *param1, void *param2) {
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
	int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;
	const int w = fpip->w;
	const int h = fpip->h;
	const int max_w = fpip->max_w;
	int y_start = h *  thread_id    / thread_num;
	int y_fin   = h * (thread_id+1) / thread_num;
	
	//最初の行はそのままコピー
	if (0 == y_start) {
		memcpy_sse<false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
		y_start++;
	}
	//最後の行はそのままコピー
	y_fin -= (h == y_fin);

	uint8_t *src_line = (uint8_t *)(fpip->ycp_edit + y_start * max_w);
	uint8_t *dst_line = (uint8_t *)(fpip->ycp_temp + y_start * max_w);

	__declspec(align(16)) int16_t diffBuf[32];
	__declspec(align(16)) int expBuf[32];
	__m128i xPMDBufMaxLimit = _mm_set1_epi16(PMD_TABLE_SIZE-1);
	__m128i xPMDBufMinLimit = _mm_set1_epi16(-PMD_TABLE_SIZE);

	for (int y = y_start; y < y_fin; y++, src_line += max_w * sizeof(PIXEL_YC), dst_line += max_w * sizeof(PIXEL_YC)) {
		uint8_t *src = src_line;
		uint8_t *dst = dst_line;
		
		//まずは、先端終端ピクセルを気にせず普通に処理してしまう
		//先端終端を処理する際に、getDiffがはみ出して読み込んでしまうが
		//最初と最後の行は別に処理するため、フレーム範囲外を読み込む心配はない
		//先端終端ピクセルは後から上書きコピーする
		uint8_t *src_fin = src + w * sizeof(PIXEL_YC);
		for ( ; src < src_fin; src += 16, dst += 16) {
			__m128i xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff;
			getDiff(src, max_w, xSrcUpperDiff, xSrcLowerDiff, xSrcLeftDiff, xSrcRightDiff);

			xSrcUpperDiff = _mm_max_epi16(xSrcUpperDiff, xPMDBufMinLimit);
			xSrcLowerDiff = _mm_max_epi16(xSrcLowerDiff, xPMDBufMinLimit);
			xSrcLeftDiff  = _mm_max_epi16(xSrcLeftDiff,  xPMDBufMinLimit);
			xSrcRightDiff = _mm_max_epi16(xSrcRightDiff, xPMDBufMinLimit);

			xSrcUpperDiff = _mm_min_epi16(xSrcUpperDiff, xPMDBufMaxLimit);
			xSrcLowerDiff = _mm_min_epi16(xSrcLowerDiff, xPMDBufMaxLimit);
			xSrcLeftDiff  = _mm_min_epi16(xSrcLeftDiff,  xPMDBufMaxLimit);
			xSrcRightDiff = _mm_min_epi16(xSrcRightDiff, xPMDBufMaxLimit);

			_mm_store_si128((__m128i *)(diffBuf +  0), xSrcUpperDiff);
			_mm_store_si128((__m128i *)(diffBuf +  8), xSrcLowerDiff);
			_mm_store_si128((__m128i *)(diffBuf + 16), xSrcLeftDiff);
			_mm_store_si128((__m128i *)(diffBuf + 24), xSrcRightDiff);

			for (int i = 0; i < 32; i += 4) {
				expBuf[i+0] = pmdp[diffBuf[i+0]];
				expBuf[i+1] = pmdp[diffBuf[i+1]];
				expBuf[i+2] = pmdp[diffBuf[i+2]];
				expBuf[i+3] = pmdp[diffBuf[i+3]];
			}

			__m128i xEUpperlo = _mm_load_si128((__m128i *)(expBuf +  0));
			__m128i xEUpperhi = _mm_load_si128((__m128i *)(expBuf +  4));
			__m128i xELowerlo = _mm_load_si128((__m128i *)(expBuf +  8));
			__m128i xELowerhi = _mm_load_si128((__m128i *)(expBuf + 12));
			__m128i xELeftlo  = _mm_load_si128((__m128i *)(expBuf + 16));
			__m128i xELefthi  = _mm_load_si128((__m128i *)(expBuf + 20));
			__m128i xERightlo = _mm_load_si128((__m128i *)(expBuf + 24));
			__m128i xERighthi = _mm_load_si128((__m128i *)(expBuf + 28));

			__m128i xAddLo, xAddHi;
			xAddLo = xEUpperlo;
			xAddHi = xEUpperhi;
			xAddLo = _mm_add_epi32(xAddLo, xELowerlo);
			xAddHi = _mm_add_epi32(xAddHi, xELowerhi);
			xAddLo = _mm_add_epi32(xAddLo, xELeftlo);
			xAddHi = _mm_add_epi32(xAddHi, xELefthi);
			xAddLo = _mm_add_epi32(xAddLo, xERightlo);
			xAddHi = _mm_add_epi32(xAddHi, xERighthi);

			__m128i xSrc = _mm_loadu_si128((__m128i *)(src));
			_mm_storeu_si128((__m128i *)(dst), _mm_add_epi16(xSrc, _mm_packs_epi32(xAddLo, xAddHi)));
		}
		//先端と終端をそのままコピー
		*(PIXEL_YC *)dst_line = *(PIXEL_YC *)src_line;
		*(PIXEL_YC *)(dst_line + (w-1) * sizeof(PIXEL_YC)) = *(PIXEL_YC *)(src_line + (w-1) * sizeof(PIXEL_YC));
	}
	//最後の行はそのままコピー
	if (h-1 == y_fin) {
		memcpy_sse<false>((uint8_t *)dst_line, (uint8_t *)src_line, w * sizeof(PIXEL_YC));
	}
}
