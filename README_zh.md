[![PyPI](https://img.shields.io/pypi/v/seg2link)](https://pypi.org/project/seg2link/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)

## 语言

- [English](README.md) | [中文](README_zh.md) | [日本語](README_jp.md)

# ![图标](docs/pics/icon.svg)

**Seg2Link** 是一个基于 [napari](https://napari.org) 的软件，专门为科学研究设计。该软件致力于解决一个具体问题：为大尺寸的3D细胞图像提供一个高效的自动分割修正工具箱，特别适用于通过电子显微镜获得的大脑图像。

我们详尽的在线文档提供了逐步的 [教程](https://wenchentao.github.io/Seg2Link/) ，而我们的 [学术论文](https://doi.org/10.1038/s41598-023-34232-6) 则深入探讨了软件背后的科学方法和验证。

与其他分割解决方案不同，Seg2Link需要预处理的细胞/非细胞区域预测作为输入，这些预测可以方便地通过 [Seg2linkUnet2d](https://github.com/WenChentao/seg2link_unet2d) (参见 [教程](https://wenchentao.github.io/Seg2Link/seg2link-unet2d.html) ) 来创建。这一集成方式使得分割过程更加准确和高效。

## 特点
- **利用深度学习预测** -- Seg2Link使用深度学习的预测作为输入，并通过半自动的用户操作将不准确的初步预测转化为高度精确的结果。
- **简单易用** -- Seg2Link不仅能自动生成分割结果，还通过少量的鼠标和键盘操作提供了易于检查和手动校正的界面，并支持细胞排序、多步撤销和重做等功能。
- **高效性** -- Seg2Link专为快速处理具有数十亿体素的大型3D图像而设计。

## 图形摘要
![简介](docs/pics/Introduction.png)


## 安装
- 安装 [Anaconda](https://www.anaconda.com/products/individual) 或 [Miniconda](https://conda.io/miniconda.html)
- 创建并激活新的 conda 环境：
```console
$ conda create -n seg2link-env python=3.8 pip
$ conda activate seg2link-env
```
- 安装 seg2link：
```console
$ pip install seg2link
```
- 更新至最新版本：
```console
$ pip install --upgrade seg2link
```

## 使用软件
- 激活创建的环境：
```console
$ conda activate seg2link-env
```
- 启动软件：
```console
$ seg2link
```

## 引用
如果您在研究中使用了这个软件，请引用以下论文：
- Wen, C., Matsumoto, M., Sawada, M. et al. Seg2Link: an efficient and versatile solution for semi-automatic cell segmentation in 3D image stacks. _Sci Rep_ **13**, 7109 (2023). https://doi.org/10.1038/s41598-023-34232-6

