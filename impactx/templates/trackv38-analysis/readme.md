
## trackv38-analysis

## ディレクトリ

``` terminal
.
├── dat
│   ├── beam.out   (input)
│   ├── coord.out  (input)
│   ├── reduced_beam_characteristics.0 (input)
│   └── visualize.json (config)
├── png
├── pyt
└── tasks.py
```


## コマンド

``` terminal
$ invoke clean
$ invoke plot --stat --poincare
$ invoke compare
```

### invoke plot

	- trackv38のstat (beam.out) または、poincare plot (coord.out)を出力
	- beam.out または、coord.out, visualize.json が必要


### invoke compare

	- impactx との比較
	- beam.out と reduced_beam_..., visualize.json が必要


