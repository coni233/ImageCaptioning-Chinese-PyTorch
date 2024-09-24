# 图像中文描述

图像中文描述 + 视觉注意力的 PyTorch 实现。

[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf) 这篇论文的复现，[这里是作者的原始实现](https://github.com/kelvinxu/arctic-captions)。

改进字自这个项目：https://github.com/foamliu/Image-Captioning-PyTorch/tree/master
原作者用的torch版本太低了，提供的模型也不是对应torch版本的，所以做了些更改

## 依赖
- Python 3.8.10
- PyTorch 1.13.1

## 数据集

使用 AI Challenger 2017 的图像中文描述数据集，包含30万张图片，150万句中文描述。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。

下载点这里：[图像中文描述数据集](https://challenger.ai/datasets/)，放在 data 目录下。


## 网络结构

 ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/net.png)

## 用法

### 数据预处理
提取210,000 张训练图片和30,000 张验证图片：
```bash
$ python pre_process.py
```

### 生成WORDMAP.json
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

### 演示
下载 [预训练模型](https://github.com/coni233/ImageCaptioning-Chinese-PyTorch/) 放在 models 目录，然后执行:

```bash
$ python demo.py
```


