# Cloudera Data Science/Machine Learning Meetup #1 
# Deep Learning Hands-on: オルタナティブデータと自然言語処理(NLP)

## はじめに

この演習の環境は、下記の環境構築スクリプトを使って、AWS上に構築しています。

https://github.com/YoshiyukiKono/cloudera-demo-env

上記を参照して、ご自身で環境構築し、その環境で演習を実行することも可能です。

## Cloudera Data Science Workbench (CDSW) について

下記ページに、この演習を始めるに当たっての、CDSW操作の概要を記します。スクリーンショットを掲載していますので、適宜ご参照ください。

[CDSW操作](./docs/cdsw.md)

CDSW環境にログインし、このリポジトリを指定して新しいCDSWプロジェクトを作成します。

https://github.com/YoshiyukiKono/dsml_01_nlp.git

CDSWプロジェクトから、Jupyter notebookを使ったセッションを開始することで、演習を始めることができます。

## 演習について

Jupyter notebookが利用可能な環境が準備できたら、下記のファイルを開いてください。

./nlp_solution.ipynb

演習の主要な部分はこのJupyter notebookを使って実施されます。

最後に、Jupyter notebook上での操作で構築・保存したモデルを、CDSWの機能を使って、WEBサービス(Rest API)として公開します。

https://docs.cloudera.com/documentation/data-science-workbench/1-6-x/topics/cdsw_models.html

## トラブルシューティング


### セッションに割り当てるメモリ
トレーニング時には、十分なメモリを用意する必要があります。さもなければ、下記のエラーが発生します。

`The kernel appears to have died. It will restart automatically.`
