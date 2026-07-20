
# 青木さんの1D 低ベータトラッキングツール (python w/ 空間電荷)

- 低β線形加速器を対象とした、簡易1次元縦方向トラッキングツール。
- 青木さんコード (mathematika -> python 化)、及び、空間電荷効果を入れたコードを作成。
- RF加速、TTF、局所セパラトリクス、捕獲効率および縦方向空間電荷効果を取り扱う。


### ディレクトリ構成

```terminal
.
├── inp
│   ├── eh_DTL.#01 ( 電場マップ track DTL要素用 )
│   ├── parameters.json ( パラメータファイル )
│   └── ttf_table.csv ( ttf テーブル )
├── out ( 出力ディレクトリ )
├── png ( 画像出力 )
├── pyt ( python codes )
│   ├── analyze__TTFbyEnergyAperture.py ( TTF テーブル作成用 )
│   └── track__longitudinal1D.py ( tracking 用 )
│   └── plot__longitudinal1D.py ( 可視化用 )
└── tasks.py

```

### 実行コマンド

```terminal
$ invoke all
```

または、

```terminal
invoke clean
invoke ttf
invoke track
invoke post --snapshot --energyRange
```



## Quick start

```bash
# TTFテーブル作成
invoke ttf

# 縦方向トラッキング
invoke track

# セルごとの位相空間図とGIF
invoke post --snapshot

# 指定エネルギー範囲の位相空間図
invoke post --energyRange

# 両方を作成
invoke post --snapshot --energyRange
```

パラメータファイルを指定：

```bash
invoke track --parameterFile=inp/parameters_test.json
```



## 解析フローの注釈

1. `invoke ttf`
   - 電場マップ`eh_DTL.#01`から、エネルギー・開口ごとのTTFを計算
   - TTFテーブルと解析サマリーを出力
2. `invoke track`
   - 初期RF位相を走査して粒子を1次元縦方向に追跡
   - 各セルの粒子分布、同期粒子、セパラトリクス、捕獲効率をCSVへ保存
3. `invoke post`
   - 保存済みCSVを読み込み、再計算せずに図とGIFを作成


## 内部処理

- TTFは電場マップ (Ez(z)) から計算、径方向位置が１点の場合、内挿用にコピー
- RF電圧・同期位相をバンチング部から加速部へ連続的に遷移 ( transition function )
- TTFは一定値、または事前計算したCSVテーブルから補間 ( mode= csv or constant )
- 各セルで粒子のRF位相と運動エネルギーを更新 ( 1D cell by cell )
- 各セルの同期粒子条件から局所セパラトリクスを計算 ( 基準粒子に対する separatrix )
- 捕獲効率は、初期粒子数に対する局所セパラトリクス内粒子数の割合
- 空間電荷は、縦方向ディスク分割と円筒導体境界のBessel展開による簡易モデル
  - Baartmann 1986 Eq.(2) 参照
- `spaceCharge.enabled=false`で空間電荷を無効化
- 横方向のトラッキング等はない。


## Requirements

```bash
pip install invoke json5 numpy pandas scipy matplotlib pillow tqdm
```

