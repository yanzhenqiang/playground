# GPT2 for Chinese chitchat

## UPDATE 2021.06.16
发布了[基于CPM模型的中文文本生成项目](https://github.com/yangjianxin1/CPM) ，可用于作文、小说、新闻、古诗等中文生成任务，并且训练和分享了中文作文生成模型，取得了不错的生成效果。
该项目提供了数据预处理、模型训练、文本生成、Http服务等代码模块。

## UPDATE 2021.05.26
- 调整项目结构，优化代码，修改部分bug。简化生成方法，加快生成速度，删除了MMI的做法
- 新增50w、100w的多轮对话的原始数据与预处理数据

## UPDATE 2020.01.09
添加50w闲聊语料与预训练模型的GoogleDrive的下载地址

## UPDATE 2019.12.17
~~基于微软的论文[DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)添加了MMI Model(maximum mutual information scoring function),对dialogue model生成的多个response进行筛选~~


## 项目描述
- 本项目是基于GPT2的中文闲聊机器人，模型实现基于HuggingFace的[transformers](https://github.com/huggingface/transformers)。
- 本项目受 [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)启发，精读作者的代码，获益匪浅。
- 在生成阶段，使用了Temperature、Top-k Sampling和Nucleus Sampling等，可参考论文[The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- ~~根据微软的DialoGPT的思想，在项目中添加了互信息。训练了两个模型:Dialogue Model与MMI Model(maximum mutual information scoring function)。首先使用Dialogue Model生成多个候选response，然后使用MMI Model从候选response中，选取loss最小的作为最终的response~~
- 代码中给出了许多详细的中文注释，方便大家更好地理解代码(部分代码或注释可能有误，望大家不吝赐教)
- **本项目被[微软的DialoGPT项目](https://github.com/microsoft/DialoGPT)引用**（为了简化生成方法，加快生成速度，删除了MMI的生成方法）

## 运行环境
python3.6、 transformers==4.2.0、pytorch==1.7.0

## 项目结构
- data
    - train.txt:默认的原始训练集文件，存放闲聊语料 
- model:存放对话生成的模型
    - epoch40:经过40轮训练之后得到的模型
      - config.json:模型参数的配置文件
      - pytorch_model.bin:模型文件
- vocab
    - vocab.txt:字典文件。默认的字典大小为13317。
- sample:人机闲聊聊天记录
- train.py
- interact.py
- preprocess.py


## 模型简介

### 模型参数简介(详见模型的config.json文件)
- initializer_range: 0.02
- layer_norm_epsilon: 1e-05
- n_ctx: 1024
- n_embd: 768
- n_head: 12
- n_layer: 12
- n_positions: 1024
- vocab_size: 21128

## 训练思路
多轮闲聊训练数据,在训练模型时，将训练数据进行如下拼接:"[CLS]想看你的美照[SEP]亲我一口就给你看[SEP]我亲两口[SEP]讨厌人家拿小拳拳捶你胸口[SEP]"。然后将上述拼接结果作为模型的输入，让模型进行自回归训练。

## 使用方法
### Quick Start
```
python interact.py --no_cuda --model_path model_epoch40_50w
或
python interact.py --model_path model_epoch40_50w --device 0
```

###  数据预处理
创建data文件夹，将原始训练语料命名为train.txt。
train.txt每段闲聊之间间隔一行，格式如下：
```
真想找你一起去看电影
突然很想你
我也很想你

想看你的美照
亲我一口就给你看
我亲两口
```
对data/train.txt对话语料进行tokenize，然后进行序列化保存到data/train.pkl。
train.pkl中序列化的对象的类型为List[List],记录对话列表中,每个对话包含的token。
```
python preprocess.py --train_path data/train.txt --save_path data/train.pkl
```

### 训练模型
当patience=n时，若连续n个epoch，在验证集上loss均没有下降，则进行early stop，停止训练。
当patience=0时，不进行early stop。
代码中默认关闭了early stop，因为在实践中，early stop得到的模型的生成效果不一定会更好。
```
python train.py --epochs 40 --batch_size 8 --device 0,1 --train_path data/train.pkl
```
### 人机交互
```
python interact.py --no_cuda --model_path path_to_your_model --max_history_len 3
```
可以尝试通过调整topk、topp、repetition_penalty、max_history_len等，调整生成的效果。
由于闲聊对话不长，因此CPU上跑也挺快
如果要使用GPU进行生成，则不要调用--no_cuda参数，并且通过--device gpu_id来指定使用哪块GPU。

## 闲聊语料分享
|中文闲聊语料 | 数据集地址 |语料描述|
|---------|--------|--------|
|常见中文闲聊|[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)|包含小黄鸡语料、豆瓣语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、PTT八卦语料、青云语料等|
|50w中文闲聊语料 | [百度网盘【提取码:4g5e】](https://pan.baidu.com/s/1M87Zf9e8iBqqmfTkKBWBWA) 或 [GoogleDrive](https://drive.google.com/drive/folders/1QFRsftLNTR_D3T55mS_FocPEZI7khdST?usp=sharing) |包含50w个多轮对话的原始语料、预处理数据|
|100w中文闲聊语料 | [百度网盘【提取码:s908】](https://pan.baidu.com/s/1TvCQgJWuOoK2f5D95nH3xg) 或 [GoogleDrive](https://drive.google.com/drive/folders/1NU4KLDRxdOGINwxoHGWfVOfP0wL05gyj?usp=sharing)|包含100w个多轮对话的原始语料、预处理数据|

<h2 id="model_share">模型分享</h2>

|模型 | 共享地址 |模型描述|
|---------|--------|--------|
|model_epoch40_50w | [百度网盘【提取码:ju6m】](https://pan.baidu.com/s/1iEu_-Avy-JTRsO4aJNiRiA) 或 [GoogleDrive](https://drive.google.com/drive/folders/1fJ6VuBp4wA1LSMpZgpe7Hgm9dbZT5bHS?usp=sharing) |使用50w多轮对话语料训练了40个epoch，loss降到2.0左右。|


## 生成样例
``` 
  user :你好啊
chatbot:你好，请问您是？
  user :我是你的主人啊 
chatbot:你是狗吗？
  user :你才是狗
chatbot:你不是狗我是你的小主人
  user :你这个大笨蛋
chatbot:你大笨蛋，你是想挨打了
  user :你全家都是大笨蛋
chatbot:我是你爸爸
  user :算了，不跟你吵了，你在干嘛呢
chatbot:我在上班啊，你还没下班呢啊你
  user :我周末要做作业啊
chatbot:不是才交了吗，早晚做完啊
  user :好多作业啊，还没做完呢
chatbot:我在抄，我妈让我周五交，还有事
  user :你就不怕被你妈妈骂吗
chatbot:不怕
  user :你周末有去哪里玩吗
chatbot:没时间啊，忙得很啊
  user :忙着干嘛呢
chatbot:不知道啊，我周末没有作业，我想睡觉，然后还得找人
``` 

## TODO
- 多卡并行训练负载不均衡的问题

## Reference
- [The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- [transformers](https://github.com/huggingface/transformers)
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)
