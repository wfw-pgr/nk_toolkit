
# trackv39 

### 実行

``` terminal
run-trackv39
```


### ディレクトリ構成

``` terminal
kent@maxwell ~/.../track/qm-using-DTL $ tree
.
├── track.dat              ( 入力 - シミュレーション設定 )
├── sclinac.dat            ( 入力 - ラティス設定 )
├── fi_in.dat              ( 入力 - 初期粒子設定等、カラでもあればよい )
├── Graph.cfg              ( 入力 - GUIグラフ表示設定 )
├── DTL.#01                ( 入力 - DTL要素使用時のみ - 読込み電磁場の強度等の設定 )
├── eh_DTL.#01             ( 入力 - DTL要素使用時のみ - 電場マップと読込み磁場ファイルの指定 )
├── eh_PMQ.#01             ( 入力 - DTL要素使用時のみ - 磁場マップ )
├── beam.out               ( 出力ファイル - 各点のstatisticalな量 )
├── coord.out              ( 出力ファイル - 最終地点の粒子分布     )
├── refp.out               ( 出力ファイル - 参照粒子情報 )
├── lost.out               ( 出力ファイル - 損失粒子情報 )
├── read_dis.out           ( 出力ファイル - 出口粒子 バイナリ )
├── ini_dis.dat            ( 起動時出力 - 初期粒子設定ログ情報 バイナリ )
├── linac.dat              ( 起動時出力 - ラティス設定ログ情報 テキスト )
├── log.out                ( 実行時出力 - ログ )
└── trackv38-analysis      ( 可視化ディレクトリ )
```

