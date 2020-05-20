﻿//----------------------------------------------------------------------------------
//        PMD_MT
//----------------------------------------------------------------------------------
/*
エモリ氏が作成されたPMD for AviUtlをマルチスレッド化対応にしてみました。
マルチスレッド化を含めて、主な変更点は以下です。

・マルチスレッド化
・浮動小数点演算から整数演算へ変更
・計算式の単純化
・簡易NRだった色差の処理も修正PMD法に変更
・3*3の範囲を追加
・効果が見られないので"sharp"スライダを削除

上の三つで高速化したので気を良くして、色差も修正PMD法でNRするようにして、オリジナルの十字4ピクセル範囲では細かい部分に点状のノイズが出ることから3*3範囲に変更しました。なので、結果的に、ほとんど高速化とならなかったのでした。とりあえずは性能が上がったということで、良しとしてください。
それにしましても。本当に酷いソースです。色々と試してみたのですが、現時点の私の技術ではSIMD化など高度な高速化ができないとあきらめまして、計算式を極端に単純化してコンパイラの最適化に高速化を任せることにしました。確かに高速化したものの……こんなソース、これ以上の改良とかする気が起きないです。バグがあっても修正に手間取ること請け合いです。もしSIMD化をしようとするならば、一から作り直さなければならないかもしれません。
なお、修正PMD法はインパルスノイズに強いそうで、このフィルタも輪郭を残したまま自然にモスキートノイズを除去してくれます。ですが、色ムラなどのノイズにはほとんど効果がありません。なんとかならないものかと、ガウスぼかしの範囲を広げたり、ガウスぼかしではなく強くぼかしたり、NR範囲を3*3以上に広げたりしたものの、ノイズ除去の効果以上にぼけていくだけでした。実装の仕方が不味かっただけかもしれませんが……

追記(ver0.14)
以下のサイトに、TAnisotropicというavusynthの2DNRフィルタがありまして。

tritical's Avisynth Filters
http://web.missouri.edu/~kes25c/

ソースを見てみたら、なんか同じようなことをやっているなぁ、なんて思ったら。修正ではないPMD法でした。アルゴリズムはこちら。

http://www.cs.berkeley.edu/~malik/papers/MP-aniso.pdf

専門用語ばかりの英文、なおかつ数式がやたら多いので、正直なところ私にはチンプンカンプンなのですが。TAnisotropicのソースを見ると基本のアルゴリズムそのものは面倒なものではないのかな、と。
ということで。修正ではないPMD法のノイズ除去も追加しました。そしてようやく、修正PMD法も理解できました。
PMD法とは、弱いNRを何度か行うことで輪郭を維持しながら平坦化を行う、、、かな?
修正PMD法とは、NRの閾値判定にぼかした画像を使用することで、輪郭を維持しながら小さな点(インパルスノイズ)や細い線(リンギング)を効果的に除去できる、、、かな?
ぼかした画像を判定に使う発想は、このフィルタに限らず、色々と応用が効きそうな気がしないでもないです。また、閾値判定に使用する画像を色々と変えてみる、例えばぼかした画像ではなくメディアン処理したものではどうなるのかな……なんて思いをめぐらしつつ。想像するだけならば簡単なので。
なお、閾値判定にTAnisotropicにもあった指数関数を使ったものを追加しました。輪郭が強調されますが、ちょっと不自然に強調されるような気も。
そして。画像の四辺の処理ですが。以前のソースは速度重視で判定をしていなかったため、それはもう酷くて見れたものではありませんでした。こんなことならば、いっそのこと四辺は処理しなくともよいだろうと思っていたのですが、、、TAnisotropicも四辺を処理していませんでした。ということで。四辺はNRをしていません。これでソースもすっきり!

念の為。
PMD(Perona-Malik Diffusions)
PIETRO PERONAさんとJITENDRA MALIKさんのアルゴリズム
異方性拡散(Anisotropic Diffusion)という考え方らしいけれど。数学(物理?)は分からない、、、


2008/12/21 ver0.14
・四辺を処理しないことに
・オリジナルのPMD法のNRを追加
・3*3を削除
2008/12/1 ver0.07
・公開
*/

//----------------------------------------------------------------------------------
//        PMD_MT 高速化版 (rigaya)
//----------------------------------------------------------------------------------
//もともとのCによるコードに対し
//SSE2 / SSSE3 / SSE4.1 / 128bit-AVX / 256bit-AVX / 256bit-AVX2 / 256bit-FMA3
//などにより高速化しました。(イントリンシックを使用)
//
//自動的に使用可能な最速の関数が使用されるようになっています。
//
//AVX/AVX2/FMA3命令を使用するには、CPUの対応のほか、
//OSがWin7 SP1以降であることが必要です。
//
//主な変更点
// useExpの場合には従来通りテーブル参照を使って計算
// useExp出ない時には、テーブル参照せずに重みを浮動小数点演算するように変更
// ガウスぼかしをする際に毎回メモリ確保するのをやめて高速化

#include <windows.h>
#include <stdlib.h>    //mallocを使用するので
#include <math.h>    //powを使用するので
#include "filter.h"
#include "pmd_mt.h"
#if CHECK_PERFORMANCE
#include <stdio.h>
#endif

//---------------------------------------------------------------------
//        プロトタイプ宣言
//---------------------------------------------------------------------
// Perona-Malik エッジ停止関数
static int *PMD = NULL;
static PIXEL_YC *gauss = NULL;
static const PMD_MT_FUNC *func = NULL;
static PERFORMANCE_CHECKER pmd_qpc = { 0 };

#define SWAP(type,x,y) { type temp = x; x = y; y = temp; }

//---------------------------------------------------------------------
//        フィルタ構造体定義
//---------------------------------------------------------------------
#define    TRACK_N 3                            //    トラックバーの数
TCHAR    *track_name[] =    {"強さ", "閾値", "回数"};    //    トラックバーの名前
int    track_default[] =    {100, 100,  1};            //    トラックバーの初期値
int    track_s[] =            { 0,    1,  1};            //    トラックバーの下限値
int    track_e[] =            {100, 240, 10};            //    トラックバーの上限値

#define    CHECK_N 2                            //    チェックボックスの数
TCHAR    *check_name[] = {"useExp", "修正PMD"};            //    チェックボックスの名前
int    check_default[] =    {0, 1};                //    チェックボックスの初期値 (値は0か1)

FILTER_DLL filter = {
    FILTER_FLAG_EX_INFORMATION,    //    フィルタのフラグ
    0,0,                        //    設定ウインドウのサイズ (FILTER_FLAG_WINDOW_SIZEが立っている時に有効)
    "PMD_MT",                    //    フィルタの名前
    TRACK_N,                    //    トラックバーの数 (0なら名前初期値等もNULLでよい)
    track_name,                    //    トラックバーの名前郡へのポインタ
    track_default,                //    トラックバーの初期値郡へのポインタ
    track_s,track_e,            //    トラックバーの数値の下限上限 (NULLなら全て0～256)
    CHECK_N,                    //    チェックボックスの数 (0なら名前初期値等もNULLでよい)
    check_name,                    //    チェックボックスの名前郡へのポインタ
    check_default,                //    チェックボックスの初期値郡へのポインタ
    func_proc,                    //    フィルタ処理関数へのポインタ (NULLなら呼ばれません)
    func_init,                    //    開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_update,                //    設定が変更されたときに呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_WndProc,                //    設定ウィンドウにウィンドウメッセージが来た時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,NULL,                    //    システムで使いますので使用しないでください
    NULL,                        //    拡張データ領域へのポインタ (FILTER_FLAG_EX_DATAが立っている時に有効)
    NULL,                        //    拡張データサイズ (FILTER_FLAG_EX_DATAが立っている時に有効)
    "PMD_MT",                    //    フィルタ情報へのポインタ (FILTER_FLAG_EX_INFORMATIONが立っている時に有効)
    NULL,                        //    セーブが開始される直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                        //    セーブが終了した直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
};
//---------------------------------------------------------------------
//        フィルタ構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable(void) {
    return &filter;
}
//---------------------------------------------------------------------
//        横軸ガウシアンぼかし関数
//---------------------------------------------------------------------
void gaussianV(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    int y_start = (fpip->h * thread_id     ) / thread_num;
    int y_end   = (fpip->h * (thread_id+1)) / thread_num;

    PIXEL_YC *ycp, *ycp2;
    PIXEL_YC *ycp_gauss = (PIXEL_YC *)param2;
    const int w = fpip->w-2;
    const int max_w = fpip->max_w;

    //" 1 : 4 : 6 : 4 : 1 "の比率で横軸をぼかしていきます
    //端は条件判定をした方がソースがすっきりしますが、処理が遅くなるので強引に計算しています
    for (int y = y_start; y < y_end; y++) {
        ycp  = fpip->ycp_temp + y*max_w;
        ycp2 = ycp_gauss + y*max_w;
        //左端
        ycp2->y  = (ycp[ 0].y  + (ycp[ 0].y <<2) + (ycp[0].y *6) + (ycp[1].y <<2) + ycp[2].y ) >> 4;
        ycp2->cb = (ycp[ 0].cb + (ycp[ 0].cb<<2) + (ycp[0].cb*6) + (ycp[1].cb<<2) + ycp[2].cb) >> 4;
        ycp2->cr = (ycp[ 0].cr + (ycp[ 0].cr<<2) + (ycp[0].cr*6) + (ycp[1].cr<<2) + ycp[2].cr) >> 4;
        ycp++;    ycp2++;
        //左端+1
        ycp2->y  = (ycp[-1].y  + (ycp[-1].y <<2) + (ycp[0].y *6) + (ycp[1].y <<2) + ycp[2].y ) >> 4;
        ycp2->cb = (ycp[-1].cb + (ycp[-1].cb<<2) + (ycp[0].cb*6) + (ycp[1].cb<<2) + ycp[2].cb) >> 4;
        ycp2->cr = (ycp[-1].cr + (ycp[-1].cr<<2) + (ycp[0].cr*6) + (ycp[1].cr<<2) + ycp[2].cr) >> 4;
        ycp++;    ycp2++;
        //中央
        for (int x = 2; x < w; x++) {
            ycp2->y  = (ycp[-2].y  + (ycp[-1].y <<2) + (ycp[0].y *6) + (ycp[1].y <<2) + ycp[2].y ) >> 4;
            ycp2->cb = (ycp[-2].cb + (ycp[-1].cb<<2) + (ycp[0].cb*6) + (ycp[1].cb<<2) + ycp[2].cb) >> 4;
            ycp2->cr = (ycp[-2].cr + (ycp[-1].cr<<2) + (ycp[0].cr*6) + (ycp[1].cr<<2) + ycp[2].cr) >> 4;
            ycp++;    ycp2++;
        }
        //右端-1
        ycp2->y  = (ycp[-2].y  + (ycp[-1].y <<2) + (ycp[0].y *6) + (ycp[1].y <<2) + ycp[1].y ) >> 4;
        ycp2->cb = (ycp[-2].cb + (ycp[-1].cb<<2) + (ycp[0].cb*6) + (ycp[1].cb<<2) + ycp[1].cb) >> 4;
        ycp2->cr = (ycp[-2].cr + (ycp[-1].cr<<2) + (ycp[0].cr*6) + (ycp[1].cr<<2) + ycp[1].cr) >> 4;
        ycp++;    ycp2++;
        //右端
        ycp2->y  = (ycp[-2].y  + (ycp[-1].y <<2) + (ycp[0].y *6) + (ycp[0].y <<2) + ycp[0].y ) >> 4;
        ycp2->cb = (ycp[-2].cb + (ycp[-1].cb<<2) + (ycp[0].cb*6) + (ycp[0].cb<<2) + ycp[0].cb) >> 4;
        ycp2->cr = (ycp[-2].cr + (ycp[-1].cr<<2) + (ycp[0].cr*6) + (ycp[0].cr<<2) + ycp[0].cr) >> 4;
    }
}
//---------------------------------------------------------------------
//        縦軸ガウシアンぼかし関数
//---------------------------------------------------------------------
//処理内容は横軸とまったく同じです
void gaussianH(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    PIXEL_YC *ycp_buf = (PIXEL_YC *)param2;
    int y_start = (fpip->h * thread_id    ) / thread_num;
    int y_end   = (fpip->h * (thread_id+1)) / thread_num;

    PIXEL_YC *ycp, *gref;
    const int w = fpip->w;
    const int h = (y_end==fpip->h) ? y_end-2 : y_end;
    const int max_w = fpip->max_w;

    for (int x = 0; x < w; x++) {
        ycp  = fpip->ycp_edit + x + y_start*max_w;
        gref = ycp_buf + x + y_start*max_w;

        int y = y_start;

        if (0 == y_start) {
            //1行目
            gref->y  = (ycp[         0].y  + (ycp[     0].y <<2) + (ycp[0].y *6) + (ycp[max_w].y <<2) + ycp[max_w<<1].y ) >> 4;
            gref->cb = (ycp[         0].cb + (ycp[     0].cb<<2) + (ycp[0].cb*6) + (ycp[max_w].cb<<2) + ycp[max_w<<1].cb) >> 4;
            gref->cr = (ycp[         0].cr + (ycp[     0].cr<<2) + (ycp[0].cr*6) + (ycp[max_w].cr<<2) + ycp[max_w<<1].cr) >> 4;
            ycp += max_w;    gref += max_w;
            //2行目
            gref->y  = (ycp[ -max_w   ].y  + (ycp[-max_w].y <<2) + (ycp[0].y *6) + (ycp[max_w].y <<2) + ycp[max_w<<1].y ) >> 4;
            gref->cb = (ycp[ -max_w   ].cb + (ycp[-max_w].cb<<2) + (ycp[0].cb*6) + (ycp[max_w].cb<<2) + ycp[max_w<<1].cb) >> 4;
            gref->cr = (ycp[ -max_w   ].cr + (ycp[-max_w].cr<<2) + (ycp[0].cr*6) + (ycp[max_w].cr<<2) + ycp[max_w<<1].cr) >> 4;
            ycp += max_w;    gref += max_w;

            y += 2;
        }
        //中央
        for ( ; y < h; y++) {
            gref->y  = (ycp[-(max_w<<1)].y  + (ycp[-max_w].y <<2) + (ycp[0].y *6) + (ycp[max_w].y <<2) + ycp[max_w<<1].y ) >> 4;
            gref->cb = (ycp[-(max_w<<1)].cb + (ycp[-max_w].cb<<2) + (ycp[0].cb*6) + (ycp[max_w].cb<<2) + ycp[max_w<<1].cb) >> 4;
            gref->cr = (ycp[-(max_w<<1)].cr + (ycp[-max_w].cr<<2) + (ycp[0].cr*6) + (ycp[max_w].cr<<2) + ycp[max_w<<1].cr) >> 4;
            ycp += max_w;    gref += max_w;
        }

        if (y_end==fpip->h) {
            //最終行-1
            gref->y  = (ycp[-(max_w<<1)].y  + (ycp[-max_w].y <<2) + (ycp[0].y *6) + (ycp[max_w].y <<2) + ycp[max_w].y ) >> 4;
            gref->cb = (ycp[-(max_w<<1)].cb + (ycp[-max_w].cb<<2) + (ycp[0].cb*6) + (ycp[max_w].cb<<2) + ycp[max_w].cb) >> 4;
            gref->cr = (ycp[-(max_w<<1)].cr + (ycp[-max_w].cr<<2) + (ycp[0].cr*6) + (ycp[max_w].cr<<2) + ycp[max_w].cr) >> 4;
            ycp += max_w;    gref += max_w;
            //最終行
            gref->y  = (ycp[-(max_w<<1)].y  + (ycp[-max_w].y <<2) + (ycp[0].y *6) + (ycp[     0].y <<2) + ycp[          0].y ) >> 4;
            gref->cb = (ycp[-(max_w<<1)].cb + (ycp[-max_w].cb<<2) + (ycp[0].cb*6) + (ycp[     0].cb<<2) + ycp[          0].cb) >> 4;
            gref->cr = (ycp[-(max_w<<1)].cr + (ycp[-max_w].cr<<2) + (ycp[0].cr*6) + (ycp[     0].cr<<2) + ycp[          0].cr) >> 4;
        }
    }
}

//---------------------------------------------------------------------
//        修正PDMマルチスレッド関数
//---------------------------------------------------------------------
void pmd_mt(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    PIXEL_YC *ycp_gauss       = ((PMD_MT_PRM *)param2)->gauss;
    const int y_start = (fpip->h *  thread_id   ) / thread_num;
    const int y_end   = (fpip->h * (thread_id+1)) / thread_num;

    //以下、修正PMD法によるノイズ除去
    const int w = fpip->w;
    const int max_w = fpip->max_w;
    const int y_fin = (y_end==fpip->h) ? y_end-1 : y_end;

    PIXEL_YC *dst, *src, *gref;
    int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;

    if (0 == y_start)
        memcpy(fpip->ycp_temp, fpip->ycp_edit, w * sizeof(PIXEL_YC));

    for (int y = max(1, y_start); y < y_fin; y++) {
        src  = fpip->ycp_edit + y*max_w +1;
        dst  = fpip->ycp_temp + y*max_w +1;
        gref = ycp_gauss      + y*max_w +1;

        *(dst-1) = *(src-1);
        //for (int x = 1; x < w-1; x++) {    //判定するたびに"-1"の計算をしてしまうので下に変更
        for (int x = 2; x < w; x++) {
            dst->y  = src->y  + (short)((
                    ((src-max_w)->y  - src->y )*pmdp[(gref-max_w)->y  - gref->y ] +    //上
                    ((src-1    )->y  - src->y )*pmdp[(gref-1)->y      - gref->y ] +    //左
                    ((src+1    )->y  - src->y )*pmdp[(gref+1)->y      - gref->y ] +    //右
                    ((src+max_w)->y  - src->y )*pmdp[(gref+max_w)->y  - gref->y ]    //下
                    ) >>16 );
            dst->cb = src->cb + (short)((
                    ((src-max_w)->cb - src->cb)*pmdp[(gref-max_w)->cb - gref->cb] +    //上
                    ((src-1    )->cb - src->cb)*pmdp[(gref-1)->cb     - gref->cb] +    //左
                    ((src+1    )->cb - src->cb)*pmdp[(gref+1)->cb     - gref->cb] +    //右
                    ((src+max_w)->cb - src->cb)*pmdp[(gref+max_w)->cb - gref->cb]    //下
                    ) >>16 );
            dst->cr = src->cr + (short)((
                    ((src-max_w)->cr - src->cr)*pmdp[(gref-max_w)->cr - gref->cr] +    //上
                    ((src-1    )->cr - src->cr)*pmdp[(gref-1)->cr     - gref->cr] +    //左
                    ((src+1    )->cr - src->cr)*pmdp[(gref+1)->cr     - gref->cr] +    //右
                    ((src+max_w)->cr - src->cr)*pmdp[(gref+max_w)->cr - gref->cr]    //下
                    ) >>16 );
            dst++, src++, gref++;
        }
        *dst = *src;
    }
    if (y_end==fpip->h)
        memcpy(fpip->ycp_temp + (fpip->h-1) * max_w, fpip->ycp_edit + (fpip->h-1) * max_w, w * sizeof(PIXEL_YC));
}
//---------------------------------------------------------------------
//        Anisotropicマルチスレッド関数
//---------------------------------------------------------------------
void anisotropic_mt(int thread_id, int thread_num, void *param1, void *param2) {
    FILTER_PROC_INFO *fpip    = (FILTER_PROC_INFO *)param1;
    int y_start = (fpip->h * thread_id    ) / thread_num;
    int y_end   = (fpip->h * (thread_id+1)) / thread_num;

    //以下、PMD法によるノイズ除去
    const int w = fpip->w;
    const int max_w = fpip->max_w;
    const int y_fin = (y_end==fpip->h) ? y_end-1 : y_end;

    PIXEL_YC *dst, *src;
    int* pmdp = ((PMD_MT_PRM *)param2)->pmd + PMD_TABLE_SIZE;

    if (0 == y_start)
        memcpy(fpip->ycp_temp, fpip->ycp_edit, w * sizeof(PIXEL_YC));

    for (int y = max(1, y_start); y < y_fin; y++) {
        src = fpip->ycp_edit + y*max_w +1;
        dst = fpip->ycp_temp + y*max_w +1;

        *(dst-1) = *(src-1);
        for (int x = 2; x < w; x++) {
            dst->y  = src->y  + (short)(
                    pmdp[(src-max_w)->y  - src->y ] +    //上
                    pmdp[(src-1    )->y  - src->y ] +    //左
                    pmdp[(src+1    )->y  - src->y ] +    //右
                    pmdp[(src+max_w)->y  - src->y ]        //下
                    );
            dst->cb = src->cb + (short)(
                    pmdp[(src-max_w)->cb - src->cb] +    //上
                    pmdp[(src-1    )->cb - src->cb] +    //左
                    pmdp[(src+1    )->cb - src->cb] +    //右
                    pmdp[(src+max_w)->cb - src->cb]        //下
                    );
            dst->cr = src->cr + (short)(
                    pmdp[(src-max_w)->cr - src->cr] +    //上
                    pmdp[(src-1    )->cr - src->cr] +    //左
                    pmdp[(src+1    )->cr - src->cr] +    //右
                    pmdp[(src+max_w)->cr - src->cr]        //下
                    );
            dst++, src++;
        }
        *dst = *src;
    }
    if (y_end==fpip->h)
        memcpy(fpip->ycp_temp + (fpip->h-1) * max_w, fpip->ycp_edit + (fpip->h-1) * max_w, w * sizeof(PIXEL_YC));
}
//---------------------------------------------------------------------
//        事前計算関数
//---------------------------------------------------------------------
void make_table(int strength, int threshold, int useExp, int cPMD) {
    const double range = 4.0;
    const double strength2 = strength/100.0;
    //閾値の設定を変えた方が使いやすいです
    const double threshold2 = (cPMD) ? pow(2.0, threshold/10.0) : threshold*16/10.0*threshold*16/10.0;
    const double inv_threshold2 = 1.0 / threshold2;
    if (cPMD) {
        const double temp = (1<<16)/range * strength2;
        if (useExp) {
            for (int x = -PMD_TABLE_SIZE; x <= PMD_TABLE_SIZE; x++) {
                PMD[x+PMD_TABLE_SIZE] = int(temp * exp(-x*x * inv_threshold2));
            }
        } else {
            for (int x = -PMD_TABLE_SIZE; x <= PMD_TABLE_SIZE; x++) {
                PMD[x+PMD_TABLE_SIZE] = int(temp * (1.0/( 1.0 + ( x*x * inv_threshold2 ) )));
            }
        }
    } else {
        if (useExp) {
            for (int x = -PMD_TABLE_SIZE; x <= PMD_TABLE_SIZE; x++) {
                double temp = (exp( -x*x * inv_threshold2 ) )* strength2 * x / range;
                PMD[x+PMD_TABLE_SIZE] = int(temp + 0.5 - (double)(temp<0));
            }
        } else {
            for (int x = -PMD_TABLE_SIZE; x <= PMD_TABLE_SIZE; x++) {
                double temp = (1.0/( 1.0 + x*x * inv_threshold2 ))*strength2 * x /range;
                PMD[x+PMD_TABLE_SIZE] = int(temp + 0.5 - (double)(temp<0));
            }
        }
    }
}
//---------------------------------------------------------------------
//        設定ウィンドウにウィンドウメッセージが来た時に呼ばれる関数
//---------------------------------------------------------------------
#pragma warning (push)
#pragma warning (disable: 4100)
BOOL func_WndProc( HWND hwnd,UINT message,WPARAM wparam,LPARAM lparam,void *editp,FILTER *fp ) {
    switch(message) {
        //フィルタがアクティブでなければメモリを開放
        case WM_FILTER_CHANGE_ACTIVE:
            if (!(fp->exfunc->is_filter_active(fp))) {
                if (PMD) {
                    free(PMD);
                    PMD = NULL;
                }
                if (gauss) {
                    _aligned_free(gauss);
                    gauss = NULL;
                }
            }
            break;
        //終了時にメモリを開放
        case WM_FILTER_EXIT:
            if (PMD) {
                free(PMD);
                PMD = NULL;
            }
            if (gauss) {
                _aligned_free(gauss);
                gauss = NULL;
            }
            break;
    }
    return FALSE;
}
//---------------------------------------------------------------------
//        設定が変更されたときに呼ばれる関数
//---------------------------------------------------------------------
BOOL func_update(FILTER *fp, int status) {
    //スライダを動かしたら計算を
    //メモリが確保されているか判定
    if (PMD)
        make_table(fp->track[0], fp->track[1], fp->check[0], fp->check[1]);
    return TRUE;
}
//---------------------------------------------------------------------
//        開始時に呼ばれる関数
//---------------------------------------------------------------------
BOOL func_init(FILTER *fp) {
    func = get_pmd_func_list();
#if CHECK_PERFORMANCE
    QueryPerformanceFrequency((LARGE_INTEGER *)&pmd_qpc.freq);
#endif
    return TRUE;
}
#pragma warning (pop)
//---------------------------------------------------------------------
//        フィルタ処理関数
//---------------------------------------------------------------------
BOOL func_proc(FILTER *fp, FILTER_PROC_INFO *fpip) {
    get_qp_counter(&pmd_qpc.tmp[0]);

    const int useCPMD   = !!fp->check[1];
    const int useExp    = !!fp->check[0];
    const int strength  = fp->track[0];
    const int threshold = fp->track[1];

    //事前に計算した値を入れる領域を確保
    //領域を確保できなければここで処理は止めてしまいます
    if (NULL == PMD) {
        if (NULL == (PMD = (int *)malloc((PMD_TABLE_SIZE*2+1) * sizeof(int))))
            return TRUE;
        make_table(strength, threshold, useExp, useCPMD);
    }

    //修正PMD法ならばガウスぼかしをおこなう
    if (useCPMD) {
        if (NULL == gauss) {
            if (NULL == (gauss = (PIXEL_YC *)_aligned_malloc(fpip->max_w * fpip->max_h * sizeof(PIXEL_YC) + 32, 32)))
                return TRUE;
        }

        //横軸のガウスぼかし、続けて縦軸のガウスぼかしをマルチスレッドで呼び出す
        //横軸の処理を完全に終えてから縦軸の処理をしなければならないので、メインの関数をマルチスレッドで呼び出してそこから縦軸横軸と分岐するのではなく、一つ一つマルチスレッドで呼び出します
        get_qp_counter(&pmd_qpc.tmp[2]);
        fp->exfunc->exec_multi_thread_func(func->gaussianH, (void *)fpip, fpip->ycp_temp);
        get_qp_counter(&pmd_qpc.tmp[3]);
        fp->exfunc->exec_multi_thread_func(func->gaussianV, (void *)fpip, (void *)gauss);
        get_qp_counter(&pmd_qpc.tmp[4]);
        add_qpctime(&pmd_qpc.value[1], pmd_qpc.tmp[3] - pmd_qpc.tmp[2]);
        add_qpctime(&pmd_qpc.value[2], pmd_qpc.tmp[4] - pmd_qpc.tmp[3]);
#if SIMD_DEBUG
        PIXEL_YC *debug = (PIXEL_YC *)malloc(fpip->max_w * fpip->max_h * sizeof(PIXEL_YC));
        fp->exfunc->exec_multi_thread_func(gaussianH, (void *)fpip, fpip->ycp_temp);
        fp->exfunc->exec_multi_thread_func(gaussianV, (void *)fpip, (void *)debug);

        int error = 0;
        for (int y = 0; y < fpip->h; y++) {
            PIXEL_YC *d_ptr = debug + y * fpip->max_w;
            PIXEL_YC *g_ptr = gauss + y * fpip->max_w;
            for (int x = 0; x < fpip->w; x++, d_ptr++, g_ptr++) {
                if (memcmp(d_ptr, g_ptr, sizeof(PIXEL_YC)))
                    error++;
            }
        }
        if (error)
            MessageBox(NULL, "Error at gaussian SIMD calc.", "PMD_MT", NULL);

        free(debug);
#endif
    }
    for (int i = 0; i < fp->track[2]; i++) {
        PMD_MT_PRM prm;
        prm.pmd = PMD;
        prm.gauss = gauss;
        prm.strength = strength;
        prm.threshold = threshold;
        //メインの処理関数をマルチスレッドで呼び出します
        get_qp_counter(&pmd_qpc.tmp[5]);
        fp->exfunc->exec_multi_thread_func(func->main_func[useCPMD][useExp], (void *)fpip, (void *)&prm);
        get_qp_counter(&pmd_qpc.tmp[6]);
        add_qpctime(&pmd_qpc.value[3], pmd_qpc.tmp[6] - pmd_qpc.tmp[5]);
#if SIMD_DEBUG
        PIXEL_YC *debug = (PIXEL_YC *)malloc(fpip->max_w * fpip->max_h * sizeof(PIXEL_YC));
        SWAP(PIXEL_YC *, fpip->ycp_temp, debug);
        fp->exfunc->exec_multi_thread_func((useCPMD) ? pmd_mt : anisotropic_mt, (void *)fpip, (void *)&prm);
        SWAP(PIXEL_YC *, fpip->ycp_temp, debug);

        int error = 0;
        int error_threshold = (useExp) ? 0 : 1<<3;
        for (int y = 0; y < fpip->h; y++) {
            PIXEL_YC *d_ptr = debug + y * fpip->max_w;
            PIXEL_YC *g_ptr = fpip->ycp_temp + y * fpip->max_w;
            for (int x = 0; x < fpip->w; x++, d_ptr++, g_ptr++) {
                if (error_threshold < abs(d_ptr->y - g_ptr->y) + abs(d_ptr->cb - g_ptr->cb) + abs(d_ptr->cr - g_ptr->cr))
                    error++;
            }
        }
        if (error) {
            MessageBox(NULL, "Error at main calc", "PMD_MT", NULL);
        }

        free(debug);
#endif
        //テンポラリ領域と入れ替えます
        SWAP(PIXEL_YC *, fpip->ycp_edit, fpip->ycp_temp);
    }

    get_qp_counter(&pmd_qpc.tmp[1]);
    add_qpctime(&pmd_qpc.value[0], pmd_qpc.tmp[1] - pmd_qpc.tmp[0]);
#if CHECK_PERFORMANCE
    pmd_qpc.frame_count++;
    if ((pmd_qpc.frame_count & 1023) == 0) {
        FILE *fp = NULL;
        if (0 == fopen_s(&fp, "pmd_mt_performance.csv", "a")) {
            fprintf(fp, "%d,%lf,%lf,%lf,%lf\n",
                pmd_qpc.frame_count,
                pmd_qpc.value[0] * 1000.0 / (double)pmd_qpc.freq / pmd_qpc.frame_count,
                pmd_qpc.value[1] * 1000.0 / (double)pmd_qpc.freq / pmd_qpc.frame_count,
                pmd_qpc.value[2] * 1000.0 / (double)pmd_qpc.freq / pmd_qpc.frame_count,
                pmd_qpc.value[3] * 1000.0 / (double)pmd_qpc.freq / pmd_qpc.frame_count);

            fclose(fp);
        }
    }
#endif

    return TRUE;
}
