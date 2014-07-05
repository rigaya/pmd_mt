#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   1
#define USE_AVX2  1
#define USE_FMA3  1
#define USE_VPGATHER 0

#include <cstdint>
#include <cmath>
#include "pmd_mt.h"
#include "filter.h"
#include "simd_util.h"

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
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
	const int max_w = fpip->max_w;
	const int w     = fpip->w;
	const int h     = fpip->h;
	PIXEL_YC *ycp_src = fpip->ycp_edit;
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
//---------------------------------------------------------------------
//		修正PDMマルチスレッド関数
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

		yGUpperlo = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGUpperlo, yInvThreshold2, yOnef));
		yGUpperhi = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGUpperhi, yInvThreshold2, yOnef));
		yGLowerlo = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGLowerlo, yInvThreshold2, yOnef));
		yGLowerhi = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGLowerhi, yInvThreshold2, yOnef));
		yGLeftlo  = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGLeftlo,  yInvThreshold2, yOnef));
		yGLefthi  = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGLefthi,  yInvThreshold2, yOnef));
		yGRightlo = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGRightlo, yInvThreshold2, yOnef));
		yGRighthi = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yGRighthi, yInvThreshold2, yOnef));

		yGUpperlo = _mm256_mul_ps(yStrength2, yGUpperlo);
		yGUpperhi = _mm256_mul_ps(yStrength2, yGUpperhi);
		yGLowerlo = _mm256_mul_ps(yStrength2, yGLowerlo);
		yGLowerhi = _mm256_mul_ps(yStrength2, yGLowerhi);
		yGLeftlo  = _mm256_mul_ps(yStrength2, yGLeftlo);
		yGLefthi  = _mm256_mul_ps(yStrength2, yGLefthi);
		yGRightlo = _mm256_mul_ps(yStrength2, yGRightlo);
		yGRighthi = _mm256_mul_ps(yStrength2, yGRighthi);

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
		memcpy_avx2<false, false, false>((uint8_t *)fpip->ycp_temp, (uint8_t *)fpip->ycp_edit, w * sizeof(PIXEL_YC));
		y_start++;
	}
	//最後の行はそのままコピー
	y_fin -= (h == y_fin);

	uint8_t *src_line = (uint8_t *)(fpip->ycp_edit + y_start * max_w);
	uint8_t *dst_line = (uint8_t *)(fpip->ycp_temp + y_start * max_w);
	uint8_t *gau_line = (uint8_t *)(gauss          + y_start * max_w);
	
#if !USE_VPGATHER
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

			__m256i ySrc = _mm256_loadu_si256((__m256i *)(src));
			_mm256_storeu_si256((__m256i *)(dst), _mm256_add_epi16(ySrc, _mm256_packs_epi32(_mm256_srai_epi32(yAddLo, 16), _mm256_srai_epi32(yAddHi, 16))));
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

		yTUpperlo = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTUpperlo, yInvThreshold2, yOnef));
		yTUpperhi = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTUpperhi, yInvThreshold2, yOnef));
		yTLowerlo = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTLowerlo, yInvThreshold2, yOnef));
		yTLowerhi = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTLowerhi, yInvThreshold2, yOnef));
		yTLeftlo  = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTLeftlo,  yInvThreshold2, yOnef));
		yTLefthi  = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTLefthi,  yInvThreshold2, yOnef));
		yTRightlo = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTRightlo, yInvThreshold2, yOnef));
		yTRighthi = _mm256_rcp_ps_hp(_mm256_fmadd_ps(yTRighthi, yInvThreshold2, yOnef));

		yTUpperlo = _mm256_mul_ps(yStrength2, yTUpperlo);
		yTUpperhi = _mm256_mul_ps(yStrength2, yTUpperhi);
		yTLowerlo = _mm256_mul_ps(yStrength2, yTLowerlo);
		yTLowerhi = _mm256_mul_ps(yStrength2, yTLowerhi);
		yTLeftlo  = _mm256_mul_ps(yStrength2, yTLeftlo);
		yTLefthi  = _mm256_mul_ps(yStrength2, yTLefthi);
		yTRightlo = _mm256_mul_ps(yStrength2, yTRightlo);
		yTRighthi = _mm256_mul_ps(yStrength2, yTRighthi);

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
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param1;
	int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;
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
	
#if !USE_VPGATHER
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
