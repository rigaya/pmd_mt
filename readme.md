
# PMD_MT 高速化版
高速化 by rigaya

その名の通りPMD_MTの高速化版です。

もともとのCによるコードに対し、SSE2 / SSSE3 / SSE4.1 / AVX / FMA3 / FMA4 / AVX2 / AVX2VNNI / AVX512BW / AVX512VBMI / AVX512VNNI などにより高速化しました。自動的に使用可能な最速の関数が使用されるようになっています。

マルチスレッドにも対応しています。(オリジナル版からマルチスレッドには対応しています)  

オリジナルを作成されたエモリ氏、およびその改良版を作成された方に深く感謝いたします。

## ダウンロード & 更新履歴
[rigayaの日記兼メモ帳＞＞](http://rigaya34589.blog135.fc2.com/blog-category-18.html)

## 基本動作環境
Windows 10 (x86/x64)  
Aviutl 1.00 以降 推奨

## PMD_MT 高速化版 使用にあたっての注意事項
無保証です。自己責任で使用してください。  
PMD_MT 高速化版を使用したことによる、いかなる損害・トラブルについても責任を負いません。


## オプション
- 強さ (デフォルト:100)  
　ノイズ除去の強さです。最大の100で100％です。  

- 閾値 (デフォルト:100)  
　ノイズを認識するための閾値を設定します。  

- 回数 (デフォルト:1)  
　ノイズ除去を繰り返す回数です。  

- useExp (デフォルト:OFF)  
　閾値の計算方法を変更します。ONにすると輪郭が強調されます。  

- 修正PMD (デフォルト:ON)  
  ONで修正PMD法でノイズ除去を行います。OFFでPMD法で処理します。  
  
## pmd_mt.auf.iniによる設定

pmd_mt.aufと同じフォルダに、下記のように記述したpmd_mt.auf.iniを置くと、使用するSIMD関数を選択できるように動作を変更できます。変更の際には、Aviutlの再起動が必要です。

```
[PMD_MT]
simd=auto
```

基本的にはテスト用です。

使用されているSIMD関数群は設定画面の上部に表示されます。


|simd="?" |使用されるもの|対応環境の例|
|:---|:---|:---|
| auto           | 環境に合わせ自動選択        |                       |
| avx512vbmi     | avx512vbmi+vnni             | Icelake               |
| avx512bw       | avx512bw                    | Skylake-X             |
| avx2vnni       | avx2 (gather使用)+vnni      | Alderlake             |
| avx2           | avx2 (gather使用)           | Broadwell             |
| avx2nogather   | avx2 (gather不使用)         | Haswell/Ryzen         | 
| avx            | 128bit-AVX                  | SandyBridge/Bulldozer |
| sse4.1         | SSE4.1                      | Penryn                |
| ssse3          | SSSE3                       | Merom                 |
| sse2           | SSE2                        |                       |


## ソースコードについて
- 無保証です。自己責任で使用してください。

### ソースの構成
Windows ... VCビルド  

文字コード: UTF-8-BOM  
改行: CRLF  
インデント: 空白x4  
