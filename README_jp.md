[![PyPI](https://img.shields.io/pypi/v/seg2link)](https://pypi.org/project/seg2link/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)

## 言語

- [English](README.md) | [中文](README_zh.md) | [日本語](README_jp.md)

# ![icon](docs/pics/icon.svg)

**Seg2Link**は、[napari](https://napari.org) ベースのソフトウェアであり、科学研究のために特別に設計されています。
このソフトウェアは、特に電子顕微鏡で取得された脳の画像に有用で、大規模な3D細胞画像で自動セグメンテーションを手動で素早く修正する効率的なツールボックスを提供することを目的としています。
私たちの詳細な [ドキュメント](https://wenchentao.github.io/Seg2Link/) では、段階的なチュートリアルが提供されており、
[学術論文](https://doi.org/10.1038/s41598-023-34232-6) ではソフトウェア背後の科学的方法論と検証について詳しく説明しています。

他のセグメンテーションソリューションとは異なり、Seg2Linkは、入力として細胞/非細胞領域の事前処理された予測が必要です。これらの予測は、
[Seg2linkUnet2d](https://github.com/WenChentao/seg2link_unet2d) （[ドキュメント](https://wenchentao.github.io/Seg2Link/seg2link-unet2d.html)) を使用して簡単に生成できます。この統合されたアプローチにより、セグメンテーションプロセスは正確かつ効率的になります。

#### 特長
- **深層学習の予測を利用** -- Seg2Linkは、深層学習の予測を入力として受け取り、半自動的なユーザー操作を通じて初期の不正確な予測を非常に正確な結果に洗練します。
  
- **ユーザーフレンドリー** -- Seg2Linkは、セグメンテーションの結果を自動生成するだけでなく、最小限のマウスとキーボードの操作で簡単に検査および手動修正が可能です。細胞の並べ替え、多段階の元に戻す・やり直しなどの機能がサポートされています。

- **効率** -- Seg2Linkは、数十億のボクセルを持つ大規模な3D画像を迅速に処理するように設計されています。

## 画像での紹介
![画像での紹介](docs/pics/Introduction.png)

## インストール
- [Anaconda](https://www.anaconda.com/products/individual) または [Miniconda](https://conda.io/miniconda.html) をインストールします。
- 新しい conda 環境を作成し、それを有効にします：
```console
$ conda create -n seg2link-env python=3.8 pip
$ conda activate seg2link-env
```
- seg2link をインストールします：
```console
$ pip install seg2link
```
- 最新バージョンに更新します：
```console
$ pip install --upgrade seg2link
```

## ソフトウェアの使用
- 作成した環境を有効にします：
```console
$ conda activate seg2link-env
```
- ソフトウェアを起動します：
```console
$ seg2link
```

## 引用
このパッケージを研究で使用した場合は、以下を引用してください：

- Wen, C., Matsumoto, M., Sawada, M. et al. Seg2Link: an efficient and versatile solution for semi-automatic cell segmentation in 3D image stacks. _Sci Rep_ **13**, 7109 (2023). https://doi.org/10.1038/s41598-023-34232-6
