## やっていること

このディレクトリは，卓球動画から姿勢推定を行うために利用したもの．

## 実行環境

実行環境構築はもとのリポジトリのREADME.mdをもとに作成する
URL: [KAPAO](https://github.com/wmcnally/kapao)


## クローンもととの差分

基本的にはクローンもとと同じだが，卓球動画に対して実行するために`exp`と`notebooks`が追加されている

## expディレクトリについて

各フォルダの中にある`run.sh`を実行すると指定した動画に対して姿勢推定が行われる．
`run_deepsort.sh`を実行すると，[ deepsort ](https://github.com/nwojke/deep_sort)写っている人物トラッキングが行われる．
run.sh

## notebooks

kapaoやdeepsortで獲得した姿勢情報や人物トラッキング情報を動画内に描画するために利用．

